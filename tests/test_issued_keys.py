"""Tests for gateway-issued keys.

Covers the three parts of the design:
- storage: `issued_keys` table on the shared Base, created by the audit store's
  `create_all`; hashes only, verification is a per-request lookup (no cache).
- mint interface: `/admin/keys` mint/list/revoke behind the mandatory admin bearer;
  the mint response is the only place the plaintext appears.
- enforcement: `AGENTGATE_REQUIRE_ISSUED_KEYS` off = the default behavior, untouched;
  on = one opaque 401 (missing/unknown/revoked indistinguishable to the caller)
  before admission, and verified requests carry the mint record's canonical key_id
  regardless of header casing or credential channel.
"""

from __future__ import annotations

import httpx
import pytest
from prometheus_client import REGISTRY

from agentgate import keys as keys_mod
from agentgate.limits.spend import key_id_from_auth
from tests.support import sse_handler, wait_for_audit_row


@pytest.fixture(autouse=True)
def no_leftover_controller():
    """The FastAPI app is module-global — don't leave this file's armed admission
    controller behind for other files (test_admission.py carries the same guard)."""
    from agentgate.app import app as gateway_app

    if hasattr(gateway_app.state, "admission"):
        delattr(gateway_app.state, "admission")
    yield
    if hasattr(gateway_app.state, "admission"):
        delattr(gateway_app.state, "admission")

STREAMING = {"stream": True, "stream_options": {"include_usage": True}}
ADMIN_TOKEN = "admin-secret"


def _chat(content: str = "hello") -> dict:
    return {"model": "m", "messages": [{"role": "user", "content": content}], **STREAMING}


def _admin(gateway) -> dict:
    gateway.settings.admin_token = ADMIN_TOKEN
    return {"authorization": f"Bearer {ADMIN_TOKEN}"}


async def _mint(gateway, label: str | None = None) -> dict:
    body = {"label": label} if label is not None else {}
    r = await gateway.client.post("/admin/keys", json=body, headers=_admin(gateway))
    assert r.status_code == 200
    return r.json()


UPSTREAM_KEY = "upstream-cred"


def _give_cloud_provider_a_key(gateway, name: str = "gemini") -> None:
    """Give the mock cloud provider its own upstream credential.

    With issued keys required, the inbound credential is a gateway-minted key and
    never leaves the gateway — a non-local provider forwards only if it has an
    api_key of its own to inject (else the pipeline rejects 503, pinned below).
    """
    gateway.settings.providers[name] = gateway.settings.providers[name].model_copy(
        update={"api_key": UPSTREAM_KEY})


# --- store ---------------------------------------------------------------------------


async def test_mint_verify_roundtrip_and_canonical_key_id(gateway):
    store = keys_mod.KeyStore(gateway.store)
    minted = await store.mint("laptop")

    assert minted.secret.startswith("ag_")
    # The canonical id is exactly what key_id_from_auth yields for a well-behaved client.
    assert minted.key_id == key_id_from_auth(f"Bearer {minted.secret}", None)

    row = await store.verify(minted.secret)
    assert row is not None and row.key_id == minted.key_id and row.revoked_at is None
    assert await store.verify("ag_not-a-real-key") is None

    assert await store.revoke(minted.key_id) is True
    row = await store.verify(minted.secret)
    assert row is not None and row.revoked_at is not None  # revoked ≠ unknown internally
    assert await store.revoke(minted.key_id) is True  # idempotent
    assert await store.revoke("0000000000000000") is False
    assert await store.count_active() == 0


# --- enforcement off (default) -------------------------------------------------------


async def test_gate_off_is_untouched(gateway):
    """Any invented credential still forwards, and key_id stays the raw-header hash."""
    r = await gateway.client.post(
        "/v1/chat/completions", json=_chat(),
        headers={"authorization": "Bearer anything-goes"},
    )
    assert r.status_code == 200
    row = await wait_for_audit_row(gateway.store)
    assert row is not None
    assert row.key_id == key_id_from_auth("Bearer anything-goes", None)


# --- enforcement on ------------------------------------------------------------------


async def test_gate_on_one_opaque_401_for_missing_unknown_revoked(gateway):
    minted = await _mint(gateway)
    gateway.settings.require_issued_keys = True

    missing = await gateway.client.post("/v1/chat/completions", json=_chat())
    bogus = await gateway.client.post(
        "/v1/chat/completions", json=_chat(),
        headers={"authorization": "Bearer ag_bogus"},
    )
    await gateway.client.delete(f"/admin/keys/{minted['key_id']}", headers=_admin(gateway))
    revoked = await gateway.client.post(
        "/v1/chat/completions", json=_chat(),
        headers={"authorization": f"Bearer {minted['key']}"},
    )

    for r in (missing, bogus, revoked):
        assert r.status_code == 401
        assert r.headers["www-authenticate"] == "Bearer"
        assert r.json()["error"]["type"] == "invalid_api_key"
    # No oracle: all three rejections are byte-identical on the wire.
    assert missing.content == bogus.content == revoked.content
    # Header-stage rejection writes no audit row, like an admission shed.
    assert await wait_for_audit_row(gateway.store, tries=3) is None


async def test_gate_on_valid_key_forwards_with_canonical_key_id(gateway):
    minted = await _mint(gateway, label="cli")
    gateway.settings.require_issued_keys = True
    _give_cloud_provider_a_key(gateway)

    r = await gateway.client.post(
        "/v1/chat/completions", json=_chat(),
        headers={"authorization": f"Bearer {minted['key']}"},
    )
    assert r.status_code == 200
    row = await wait_for_audit_row(gateway.store)
    assert row is not None and row.key_id == minted["key_id"]


async def test_gate_on_collapses_presentation_aliases(gateway):
    """Casing and channel variants of one credential are one identity — the raw-header
    hash would hand each variant fresh per-key counters."""
    minted = await _mint(gateway)
    gateway.settings.require_issued_keys = True
    _give_cloud_provider_a_key(gateway)

    for headers in (
        {"authorization": f"bearer {minted['key']}"},          # lowercase scheme
        {"x-goog-api-key": minted["key"]},                     # other channel
    ):
        r = await gateway.client.post("/v1/chat/completions", json=_chat(), headers=headers)
        assert r.status_code == 200, headers
        row = await wait_for_audit_row(gateway.store)
        assert row is not None and row.key_id == minted["key_id"], headers
    # And the variants do NOT equal the raw hashes they would have had gate-off.
    assert minted["key_id"] != key_id_from_auth(f"bearer {minted['key']}", None)
    assert minted["key_id"] != key_id_from_auth(None, minted["key"])


async def test_gate_on_valid_key_passes_admission_and_forwards(gateway):
    """Happy path through both gates: a verified key gets an admission slot, forwards,
    and audits under its canonical key_id."""
    minted = await _mint(gateway)
    gateway.settings.require_issued_keys = True
    gateway.settings.admission_enabled = True
    _give_cloud_provider_a_key(gateway)

    r = await gateway.client.post(
        "/v1/chat/completions", json=_chat(),
        headers={"authorization": f"Bearer {minted['key']}"},
    )
    assert r.status_code == 200
    row = await wait_for_audit_row(gateway.store)
    assert row is not None and row.key_id == minted["key_id"]


async def test_gate_on_rejects_before_admission(gateway):
    """No queue slot for unauthenticated callers: with a full queue the auth 401 still
    wins over the admission 429 because it runs first."""
    gateway.settings.require_issued_keys = True
    gateway.settings.admission_enabled = True
    gateway.settings.admission_per_key = 1
    gateway.settings.admission_queue_depth = 0

    r = await gateway.client.post("/v1/chat/completions", json=_chat())
    assert r.status_code == 401
    assert r.json()["error"]["type"] == "invalid_api_key"


# --- forward-point invariant: a minted key never leaves the gateway -------------------


def _credentials_missing_count(provider: str = "gemini") -> float:
    return REGISTRY.get_sample_value(
        "agentgate_upstream_credentials_missing_total", {"provider": provider}) or 0.0


async def test_gate_on_keyless_cloud_provider_rejects_closed(gateway):
    """Issued keys on + the resolved provider is non-local with no api_key → the request
    is rejected 503 with the distinct counted error and NOTHING is forwarded: the minted
    gateway key must never land in a third party's request logs."""
    minted = await _mint(gateway)
    gateway.settings.require_issued_keys = True

    forwarded: list[httpx.Request] = []
    gateway.set_upstream(sse_handler(record=forwarded))
    before = _credentials_missing_count()

    r = await gateway.client.post(
        "/v1/chat/completions", json=_chat(),
        headers={"authorization": f"Bearer {minted['key']}"},
    )

    assert r.status_code == 503
    assert r.json()["error"]["type"] == "upstream_credentials_missing"
    assert not forwarded, "the minted key must never reach a cloud upstream"
    assert _credentials_missing_count() == before + 1
    row = await wait_for_audit_row(gateway.store)
    assert row is not None and row.status == 503


async def test_gate_on_cloud_provider_with_key_swaps_the_credential(gateway):
    """A cloud provider WITH its own api_key forwards normally — the forwarder injects
    that credential, and the minted key appears nowhere in the outbound request."""
    minted = await _mint(gateway)
    gateway.settings.require_issued_keys = True
    _give_cloud_provider_a_key(gateway)

    forwarded: list[httpx.Request] = []
    gateway.set_upstream(sse_handler(record=forwarded))

    r = await gateway.client.post(
        "/v1/chat/completions", json=_chat(),
        headers={"authorization": f"Bearer {minted['key']}"},
    )

    assert r.status_code == 200
    (out,) = forwarded
    assert out.headers["authorization"] == f"Bearer {UPSTREAM_KEY}"
    assert all(minted["key"] not in v for v in out.headers.values())


async def test_gate_on_local_route_is_unaffected(gateway):
    """The invariant is scoped to non-local providers: the local route forwards as
    before (zero egress — the minted key stays on-box)."""
    minted = await _mint(gateway)
    gateway.settings.require_issued_keys = True
    gateway.settings.routing.enabled = False
    gateway.settings.default_provider = "ollama"  # is_local, no api_key

    forwarded: list[httpx.Request] = []
    gateway.set_upstream(sse_handler(record=forwarded))

    r = await gateway.client.post(
        "/v1/chat/completions", json=_chat(),
        headers={"authorization": f"Bearer {minted['key']}"},
    )

    assert r.status_code == 200
    assert len(forwarded) == 1


async def test_gate_off_keyless_cloud_pass_through_is_unchanged(gateway):
    """Without issued keys the inbound credential is the caller's own upstream key —
    pass-through to a keyless cloud provider stays exactly as it was."""
    forwarded: list[httpx.Request] = []
    gateway.set_upstream(sse_handler(record=forwarded))

    r = await gateway.client.post(
        "/v1/chat/completions", json=_chat(),
        headers={"authorization": "Bearer caller-cloud-key"},
    )

    assert r.status_code == 200
    (out,) = forwarded
    assert out.headers["authorization"] == "Bearer caller-cloud-key"


# --- admin plane ---------------------------------------------------------------------


async def test_admin_keys_require_admin_bearer(gateway):
    gateway.settings.admin_token = ADMIN_TOKEN
    for method, path in (("POST", "/admin/keys"), ("GET", "/admin/keys"),
                         ("DELETE", "/admin/keys/deadbeef00000000")):
        r = await gateway.client.request(method, path,
                                         headers={"authorization": "Bearer wrong"})
        assert r.status_code == 401, (method, path)


async def test_admin_list_exposes_no_secret_material(gateway):
    minted = await _mint(gateway, label="laptop")
    r = await gateway.client.get("/admin/keys", headers=_admin(gateway))
    assert r.status_code == 200
    (entry,) = r.json()
    assert entry["key_id"] == minted["key_id"]
    assert entry["label"] == "laptop"
    assert entry["revoked_at"] is None
    assert "key" not in entry and "key_hash" not in entry
    assert minted["key"] not in r.text


async def test_admin_revoke_unknown_404s(gateway):
    r = await gateway.client.delete("/admin/keys/deadbeef00000000", headers=_admin(gateway))
    assert r.status_code == 404


async def test_admin_mint_rejects_non_string_label(gateway):
    r = await gateway.client.post("/admin/keys", json={"label": 7}, headers=_admin(gateway))
    assert r.status_code == 422
