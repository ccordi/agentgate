"""Tier-3 egress PEP — policy-enforcement-point client core.

Thin clients that consult the agentgate PDP (`POST /a/egress/decision`) before
performing an outbound action, and obey its verdict.
"""

from __future__ import annotations

from .safe_http_request import EgressResult, safe_http_request

__all__ = ["EgressResult", "safe_http_request"]
