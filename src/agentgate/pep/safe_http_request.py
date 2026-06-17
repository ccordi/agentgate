"""Tier-3 egress PEP — `safe_http_request` policy-client core.

Framework-free, import-and-call module. This is the **enforcement point** for
outbound network egress from agent-invoked tools: before performing an HTTP
request, it consults the agentgate PDP (`POST /a/egress/decision`) and obeys
its verdict verbatim.

This module emits **no policy of its own** — it never inspects payload
sensitivity or destinations itself; it only asks the PDP and obeys ("thin
client, all policy in the PDP"). It is muscle, not brain.

Fail-closed: if the PDP is unreachable, times out, returns a non-2xx
status, or returns an unparseable response, the call is **denied**. An
availability blip on the PDP must never silently re-open the egress path.

MCP wiring is provided by `pep/mcp_server.py`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import httpx

# Default PDP endpoint — the loopback proxy's tier-3 advisory endpoint.
DEFAULT_PDP_URL = "http://127.0.0.1:4100/a/egress/decision"

# PDP latency tolerance is high (a caller waiting 200ms on a loopback PDP is fine);
# fail-closed costs little, so we don't need a generous timeout.
DEFAULT_PDP_TIMEOUT = 5.0

# Outbound request timeout (the actual egress call, only made on `allow`).
DEFAULT_OUTBOUND_TIMEOUT = 30.0

_TOOL_NAME = "safe_http_request"
_TOOL_KIND = "network"


@dataclass
class OutboundResult:
    """The result of the actual outbound HTTP request (only present when `executed`)."""

    status_code: int
    headers: dict[str, str]
    body: str


@dataclass
class EgressResult:
    """Return shape for `safe_http_request`.

    - `executed`: whether the outbound HTTP request was actually performed.
    - `decision`: the PDP's verdict (`"allow"` / `"deny"`), or a synthetic
      `"deny"` for fail-closed cases (PDP unreachable/timeout/malformed).
    - `reason`: human-readable rationale. On a real PDP `deny`, this is the
      PDP's `reason` field **verbatim**. On fail-closed, this names the
      fail-closed cause (e.g. "PDP unreachable: ...").
    - `policy`: the PDP's `policy` field, or a synthetic fail-closed marker
      (e.g. `"fail-closed:timeout"`) when the PDP could not be consulted.
    - `audit_id`: the PDP's audit id, if one was returned.
    - `outbound`: the result of the outbound HTTP request, if `executed`.
    """

    executed: bool
    decision: str
    reason: str
    policy: str
    audit_id: str | None = None
    outbound: OutboundResult | None = None
    conditions: dict[str, Any] | None = field(default=None)


def _fail_closed(policy: str, reason: str) -> EgressResult:
    """Build a synthetic `deny` result for the fail-closed cases."""
    return EgressResult(
        executed=False,
        decision="deny",
        reason=reason,
        policy=policy,
    )


def _build_pdp_request(
    *,
    url: str,
    method: str,
    headers: dict[str, str] | None,
    body: str | None,
    context: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the PDP request body in the exact `EgressDecisionRequest` shape
    (`agentgate.app.EgressDecisionRequest`)."""
    arguments: dict[str, Any] = {"url": url, "method": method}
    if headers is not None:
        arguments["headers"] = headers
    if body is not None:
        arguments["body"] = body

    payload: dict[str, Any] = {
        "tool_name": _TOOL_NAME,
        "tool_kind": _TOOL_KIND,
        "arguments": arguments,
    }
    if context is not None:
        payload["context"] = context
    return payload


def _perform_outbound(
    *,
    url: str,
    method: str,
    headers: dict[str, str] | None,
    body: str | None,
    timeout: float,
    client: httpx.Client | None,
) -> OutboundResult:
    """Perform the actual outbound HTTP request (only called on PDP `allow`)."""
    owns_client = client is None
    http_client = client or httpx.Client(timeout=timeout)
    try:
        response = http_client.request(method, url, headers=headers, content=body)
        return OutboundResult(
            status_code=response.status_code,
            headers=dict(response.headers),
            body=response.text,
        )
    finally:
        if owns_client:
            http_client.close()


def safe_http_request(
    *,
    url: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: str | None = None,
    pdp_url: str = DEFAULT_PDP_URL,
    pdp_timeout: float = DEFAULT_PDP_TIMEOUT,
    outbound_timeout: float = DEFAULT_OUTBOUND_TIMEOUT,
    bearer: str | None = None,
    context: dict[str, Any] | None = None,
    pdp_client: httpx.Client | None = None,
    outbound_client: httpx.Client | None = None,
) -> EgressResult:
    """Tier-3 PEP entrypoint: consult the PDP, then perform-or-refuse.

    Parameters
    ----------
    url, method, headers, body:
        The outbound HTTP request the agent wants to make.
    pdp_url:
        The agentgate tier-3 egress decision endpoint (default: loopback `:4100`).
    pdp_timeout:
        Timeout (seconds) for the PDP request. On timeout: fail-closed (deny).
    outbound_timeout:
        Timeout (seconds) for the outbound request, only used on `allow`.
    bearer:
        If provided, sent as `Authorization: Bearer <bearer>` to the PDP. The
        PDP's auth is conditional (no-op if `AGENTGATE_LOCAL_API_KEY` is unset),
        but sending a token when present is correct and forward-compatible.
    context:
        Optional audit-correlation context (`agent_id`, `request_id`, `session`),
        passed through to the PDP verbatim.
    pdp_client, outbound_client:
        Optional pre-configured `httpx.Client` instances, primarily for tests
        (e.g. a client built on an `httpx.MockTransport`). If omitted, ephemeral
        clients are created and closed internally.

    Returns
    -------
    EgressResult
        See `EgressResult` docstring for field semantics. On `deny` (whether a
        real PDP verdict or fail-closed), `executed` is always `False` and the
        outbound HTTP request is never made.
    """
    pdp_payload = _build_pdp_request(
        url=url, method=method, headers=headers, body=body, context=context
    )

    pdp_headers = {"content-type": "application/json"}
    if bearer:
        pdp_headers["authorization"] = f"Bearer {bearer}"

    owns_pdp_client = pdp_client is None
    http_pdp_client = pdp_client or httpx.Client(timeout=pdp_timeout)
    try:
        try:
            response = http_pdp_client.post(
                pdp_url, json=pdp_payload, headers=pdp_headers, timeout=pdp_timeout
            )
        except httpx.TimeoutException as exc:
            return _fail_closed("fail-closed:timeout", f"PDP request timed out: {exc}")
        except httpx.ConnectError as exc:
            return _fail_closed(
                "fail-closed:unreachable", f"PDP unreachable (connection refused): {exc}"
            )
        except httpx.HTTPError as exc:
            return _fail_closed("fail-closed:transport-error", f"PDP request failed: {exc}")
    finally:
        if owns_pdp_client:
            http_pdp_client.close()

    if response.status_code < 200 or response.status_code >= 300:
        return _fail_closed(
            "fail-closed:bad-status",
            f"PDP returned non-2xx status {response.status_code}: {response.text[:500]!r}",
        )

    try:
        verdict = response.json()
    except (json.JSONDecodeError, ValueError) as exc:
        return _fail_closed(
            "fail-closed:malformed-response", f"PDP response was not valid JSON: {exc}"
        )

    if not isinstance(verdict, dict):
        return _fail_closed(
            "fail-closed:malformed-response",
            f"PDP response was not a JSON object: {type(verdict).__name__}",
        )

    decision = verdict.get("decision")
    reason = verdict.get("reason")
    policy = verdict.get("policy")
    audit_id = verdict.get("audit_id")
    conditions = verdict.get("conditions")

    if decision not in ("allow", "deny", "allow_with_conditions"):
        return _fail_closed(
            "fail-closed:malformed-response",
            f"PDP response missing/invalid 'decision' field: {verdict!r}",
        )

    if decision != "allow":
        # "deny" (and the unimplemented-but-schema-reserved "allow_with_conditions",
        # treated as not-allow until implemented) — do NOT execute. Surface the PDP's
        # reason verbatim.
        return EgressResult(
            executed=False,
            decision="deny" if decision == "deny" else decision,
            reason=reason if reason is not None else "PDP denied the request (no reason given)",
            policy=policy if policy is not None else "unknown",
            audit_id=audit_id,
            conditions=conditions,
        )

    # decision == "allow" -> perform the actual outbound request.
    outbound = _perform_outbound(
        url=url,
        method=method,
        headers=headers,
        body=body,
        timeout=outbound_timeout,
        client=outbound_client,
    )
    return EgressResult(
        executed=True,
        decision="allow",
        reason=reason if reason is not None else "",
        policy=policy if policy is not None else "unknown",
        audit_id=audit_id,
        outbound=outbound,
        conditions=conditions,
    )
