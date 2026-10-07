"""The MCP tool wrapper's path for an outbound call that fails after the PDP allowed it.

FastMCP tool functions are plain callables, so this imports and calls one directly —
no MCP runtime, no stdio server.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mcp")

from agentgate.egress import mcp_server  # noqa: E402


def test_import_surface():
    """Import smoke for the optional-dep module: the PEP core it wraps and the tool
    callable must both resolve under the `egress-mcp` extra. An import break here is
    otherwise invisible until the stdio server is launched."""
    assert callable(mcp_server.safe_http_request)
    assert callable(mcp_server.safe_http_request_tool)
    assert callable(mcp_server.main)
    assert isinstance(mcp_server.DEFAULT_PDP_URL, str)


def test_outbound_failure_returns_clean_error_dict(monkeypatch):
    """A throw from the outbound leg must not escape: the only sanctioned network path
    dies with the stdio server if it does."""
    def boom(**kw):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(mcp_server, "safe_http_request", boom)

    out = mcp_server.safe_http_request_tool(url="https://api.internal.example/x")

    assert out["executed"] is False
    assert out["decision"] == "allow"          # the PDP said yes; the *call* failed
    assert out["policy"] == "network-egress"
    assert out["audit_id"] is None
    assert out["error"] == "RuntimeError: connection reset"


def test_non_httperror_from_bad_url_is_also_caught(monkeypatch):
    """httpx.InvalidURL is not an HTTPError subclass — uncaught, malformed agent input
    would kill the server outright."""
    import httpx

    def boom(**kw):
        raise httpx.InvalidURL("no host")

    monkeypatch.setattr(mcp_server, "safe_http_request", boom)

    out = mcp_server.safe_http_request_tool(url="notaurl")
    assert out["executed"] is False
    assert out["error"].startswith("InvalidURL:")
