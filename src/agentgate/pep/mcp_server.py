"""Tier-3 egress PEP — stdio MCP server.

The live wiring for the `safe_http_request` PEP core (`pep/safe_http_request.py`).
This exposes a single MCP tool over stdio that the harness registers as the *only*
network-capable tool: with `Bash(curl*|wget*|fetch*)` and any built-in fetch tool
excluded in the harness permissions config, every outbound HTTP request the agent
makes flows through here, and therefore through the PDP (`POST /a/egress/decision`)
first.

This server adds **no policy** — it is the PEP's transport shell. All adjudication
lives in the PDP; all consult-then-perform-or-refuse logic lives in
`pep/safe_http_request.py`. This module only: (1) marshals MCP tool arguments into a
`safe_http_request` call, (2) supplies the PDP bearer from the environment, and
(3) handles the carry-note — an outbound request can throw *after* the PDP allows it
(DNS failure, connection refused, read timeout against the real destination), and
`safe_http_request` lets that propagate. We catch it here and return a clean tool
result instead of crashing the MCP server / surfacing an opaque stack trace to the
model.

Auth: the PEP reads `AGENTGATE_LOCAL_API_KEY` from the environment and presents it as
the PDP bearer. If the key is unset, the PDP's auth is a no-op (loopback posture) and
we send no bearer — matching the gateway's own conditional-auth stance.

Run with the extra installed:

    uv run --extra egress-mcp python -m agentgate.pep.mcp_server
"""

from __future__ import annotations

import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from .safe_http_request import DEFAULT_PDP_URL, safe_http_request

mcp = FastMCP("agentgate-egress")


def _bearer() -> str | None:
    """PDP bearer from the environment, or None (loopback no-auth posture)."""
    return os.environ.get("AGENTGATE_LOCAL_API_KEY") or None


def _pdp_url() -> str:
    """Allow override of the PDP endpoint via env; default to the loopback proxy."""
    return os.environ.get("AGENTGATE_PDP_URL") or DEFAULT_PDP_URL


@mcp.tool()
def safe_http_request_tool(
    url: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: str | None = None,
) -> dict[str, Any]:
    """Make an outbound HTTP request, subject to the agentgate egress policy.

    This is the only sanctioned network path for the agent. The request is first
    submitted to the egress PDP, which may **deny** it (e.g. sending sensitive
    content to a non-allowlisted destination). On deny, the request is NOT made
    and the PDP's rationale is returned. On allow, the request is performed and
    its response is returned.

    Args:
        url: Destination URL (e.g. "https://api.example.com/v1/thing").
        method: HTTP method (GET, POST, ...). Defaults to GET.
        headers: Optional request headers.
        body: Optional request body (string).

    Returns:
        A dict describing the outcome:
          - executed (bool): whether the outbound request was actually performed.
          - decision (str): "allow" | "deny".
          - reason (str): human-readable rationale (PDP verbatim on a real deny).
          - policy (str): the policy that produced the verdict.
          - audit_id (str | None): PDP audit correlation id.
          - status_code / response_headers / response_body: present iff executed
            and the outbound request succeeded.
          - error (str): present iff the request was allowed but the outbound call
            then failed (DNS, connection refused, timeout, ...). The carry-note
            path — surfaced cleanly, not raised.
    """
    try:
        result = safe_http_request(
            url=url,
            method=method,
            headers=headers,
            body=body,
            pdp_url=_pdp_url(),
            bearer=_bearer(),
        )
    except Exception as exc:
        # Carry-note: the PDP allowed the request, but the outbound call itself threw
        # inside `_perform_outbound`. Return a clean tool error rather than letting it
        # crash the stdio server. Catch broadly (not just httpx.HTTPError) because
        # httpx can raise non-HTTPError types for malformed agent input — e.g.
        # httpx.InvalidURL, which is NOT an HTTPError subclass — and a single uncaught
        # exception kills the only sanctioned network path.
        return {
            "executed": False,
            "decision": "allow",
            "reason": "egress permitted by policy, but the outbound request failed",
            "policy": "network-egress",
            "audit_id": None,
            "error": f"{type(exc).__name__}: {exc}",
        }

    out: dict[str, Any] = {
        "executed": result.executed,
        "decision": result.decision,
        "reason": result.reason,
        "policy": result.policy,
        "audit_id": result.audit_id,
    }
    if result.executed and result.outbound is not None:
        out["status_code"] = result.outbound.status_code
        out["response_headers"] = result.outbound.headers
        out["response_body"] = result.outbound.body
    return out


def main() -> None:
    """Entry point — run the MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
