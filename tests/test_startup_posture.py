"""Startup posture: loopback bind enforced, gateway tokens mandatory, Host guarded.

The tests cover these requirements:

- `validate_runtime_settings` refuses a non-loopback bind, with NO override flag — no
  config state makes a wide bind safe, because the proxy has no credential to arm.
- It refuses to serve without both dedicated tokens (`AGENTGATE_ADMIN_TOKEN`,
  `AGENTGATE_PDP_TOKEN`): conditional auth left unarmed deployments looking closed
  while open.
- `LoopbackHostGuard` rejects non-loopback Host headers: loopback excludes remote
  *sockets*, not remote *code* — a DNS-rebound page reaches 127.0.0.1 same-origin, and
  its Host header is the tell.

Known residual: a direct `uvicorn agentgate.app:app --host
0.0.0.0` never consults `settings.host`, so the bind check cannot see it. The lifespan
still enforces the token half there.
"""

from __future__ import annotations

import httpx
import pytest

from agentgate.app import app as gateway_app
from agentgate.app import lifespan
from agentgate.config import Settings, validate_runtime_settings
from agentgate.egress.policy import is_loopback_host
from tests.support import make_settings

# ---- the one loopback definition ----------------------------------------------------


@pytest.mark.parametrize("value", [
    "127.0.0.1", "localhost", "LOCALHOST", "::1", "[::1]",
    "127.0.0.1:4100", "localhost:4100", "[::1]:4100",
])
def test_loopback_forms_accepted(value):
    assert is_loopback_host(value)


@pytest.mark.parametrize("value", [
    "0.0.0.0", "::", "10.0.0.5", "evil.example", "evil.example:4100",
    "gw", "testserver", "",
    # Real loopback addresses outside the canonical set are deliberately treated as
    # remote: under-inclusion fails strict (a refused start, a rejected Host), never
    # open. One definition, shared with the egress policy's destination axis.
    "127.0.0.2", "127.0.0.53",
])
def test_everything_else_is_treated_as_remote(value):
    assert not is_loopback_host(value)


# ---- startup validation --------------------------------------------------------------


def _ok_settings(**overrides) -> Settings:
    base = dict(admin_token="admin-t", pdp_token="pdp-t")
    base.update(overrides)
    return make_settings(**base)


def test_valid_posture_is_accepted():
    validate_runtime_settings(_ok_settings())  # loopback default host + both tokens


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10"])
def test_non_loopback_bind_refuses_startup(host):
    with pytest.raises(RuntimeError, match="not a loopback address"):
        validate_runtime_settings(_ok_settings(host=host))


def test_there_is_no_override_flag():
    """No setting may bypass the loopback requirement while proxy routes lack auth."""
    fields = set(Settings.model_fields)
    assert not any("allow" in f and "loopback" in f for f in fields)
    assert "allow_non_loopback" not in fields


def test_missing_admin_token_refuses_startup():
    with pytest.raises(RuntimeError, match="AGENTGATE_ADMIN_TOKEN"):
        validate_runtime_settings(_ok_settings(admin_token=None))


def test_missing_pdp_token_refuses_startup():
    with pytest.raises(RuntimeError, match="AGENTGATE_PDP_TOKEN"):
        validate_runtime_settings(_ok_settings(pdp_token=None))


# Presence is not separation. The PEP loads the PDP token into the agent's own
# environment, so admin == pdp hands the governed agent the kill switch meant to halt it.

def test_admin_and_pdp_tokens_must_differ():
    with pytest.raises(RuntimeError, match="same value"):
        validate_runtime_settings(_ok_settings(admin_token="same-t", pdp_token="same-t"))


def test_admin_token_must_not_reuse_the_local_api_key():
    with pytest.raises(RuntimeError, match="same value"):
        validate_runtime_settings(_ok_settings(admin_token="k", local_api_key="k"))


def test_pdp_token_must_not_reuse_the_local_api_key():
    with pytest.raises(RuntimeError, match="same value"):
        validate_runtime_settings(_ok_settings(pdp_token="k", local_api_key="k"))


def test_distinct_tokens_alongside_a_local_api_key_are_accepted():
    """Non-vacuity: the check must not fire on a correctly-configured deployment."""
    validate_runtime_settings(_ok_settings(local_api_key="a-third-value"))


async def test_lifespan_enforces_the_posture(monkeypatch):
    """Every sanctioned launch path runs lifespan (uvicorn always does), so an unsafe
    posture must die there — before any state is assembled. Hermeticized inline: no
    ambient AGENTGATE_* variable or working-tree `.env` may arm the tokens and turn
    this refusal into a pass."""
    import os

    from agentgate import config as config_mod

    for key in list(os.environ):
        if key.startswith("AGENTGATE_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    config_mod.get_settings.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="unsafe posture"):
            async with lifespan(gateway_app):
                pass  # pragma: no cover — must not be reached
    finally:
        config_mod.get_settings.cache_clear()


# ---- Host guard (anti-DNS-rebinding) -------------------------------------------------


async def test_rebound_host_is_rejected_before_any_route():
    """A rebound page's request arrives on the loopback socket but says
    `Host: evil.example` — the middleware must 400 it without touching routes or
    state (this test assembles neither)."""
    transport = httpx.ASGITransport(app=gateway_app)
    async with httpx.AsyncClient(transport=transport,
                                 base_url="http://evil.example:4100") as client:
        r = await client.get("/healthz")
        assert r.status_code == 400
        assert "loopback" in r.json()["detail"]


@pytest.mark.parametrize("origin", ["http://127.0.0.1", "http://localhost:4100"])
async def test_loopback_host_passes(origin):
    transport = httpx.ASGITransport(app=gateway_app)
    async with httpx.AsyncClient(transport=transport, base_url=origin) as client:
        r = await client.get("/healthz")
        assert r.status_code == 200
