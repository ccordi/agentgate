"""Tier-3 egress PDP — policy logic for `POST /a/egress/decision`.

Implements the egress decision matrix. A thin composition of two
existing-or-trivial signals — **no new detection code**:

- In-scope test — is this tool call a network egress at all?
- Destination axis — allowlisted vs. untrusted (loopback always allowed).
- Payload axis — reuse `security.classifier.classify()` (regex, no LLM call).
- Decision matrix — allow / deny only (`allow_with_conditions` is reserved,
  unimplemented).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from .classifier import Sensitivity, classify

# Mirrors classify_request's bound (20k chars) — see security/classifier.py.
_MAX_PAYLOAD_CHARS = 20000

# URL/host shape matcher. Matches `scheme://host[...]` or a bare `host:port`
# (e.g. "internal.example:8443"). Deliberately small and explicit.
_URL_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", re.ASCII)
_HOST_PORT_RE = re.compile(r"^[A-Za-z0-9_.-]+:\d{1,5}(/.*)?$", re.ASCII)

# Loopback is always allowed regardless of the configured allowlist.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})


@dataclass
class EgressVerdict:
    decision: str  # "allow" | "deny" | "allow_with_conditions" (latter reserved, unimplemented)
    reason: str
    policy: str
    conditions: dict[str, Any] | None = None
    # Extra fields surfaced to the endpoint for the audit row (not part of the wire
    # schema's required fields, but convenient to carry alongside the verdict).
    destination: str | None = None
    sensitivity: Sensitivity | None = None
    hit_types: list[str] = field(default_factory=list)


def _looks_like_url_or_host(value: str) -> bool:
    """Small explicit matcher: `scheme://...` or bare `host:port`."""
    if _URL_RE.match(value):
        return True
    if _HOST_PORT_RE.match(value):
        return True
    return False


def _iter_strings(value: Any):
    """Yield every string leaf in a nested arguments structure."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _iter_strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _iter_strings(v)


def _is_in_scope(tool_kind: str | None, arguments: dict[str, Any]) -> tuple[bool, str | None]:
    """In-scope if `tool_kind == "network"` or any string arg looks like a URL/host.
    Returns (in_scope, first_url_like_string_found)."""
    if tool_kind == "network":
        # Still try to find a URL-ish arg for destination extraction, but scope is
        # already established.
        for s in _iter_strings(arguments):
            if _looks_like_url_or_host(s):
                return True, s
        return True, None

    for s in _iter_strings(arguments):
        if _looks_like_url_or_host(s):
            return True, s
    return False, None


def _extract_host(url_or_host: str) -> str:
    """Pull the host (no port, no scheme, no path) out of a URL or `host:port` string.

    For URL-form inputs (anything with a scheme) the host is parsed with **httpx** —
    the exact library the PEP uses to make the outbound request
    (`pep/safe_http_request.py:_perform_outbound`). This is a security invariant, not
    a convenience: the allowlist check and the actual connection MUST parse the
    destination identically, or a single crafted URL can make the PDP see an
    allowlisted host while httpx connects elsewhere. A hand-rolled parser that, e.g.,
    only terminates the authority at `/` is bypassable with `https://evil.com#@allowed.com`
    or `https://evil.com?x=@allowed.com` (RFC 3986 fragment/query terminate the
    authority before the userinfo `@`), so we defer to httpx's RFC-3986 parser and
    inherit its host normalisation (lowercasing, userinfo/port/path/query/fragment
    stripping, IPv6 de-bracketing, IDNA).
    """
    s = url_or_host.strip()
    if _URL_RE.match(s):
        try:
            host = httpx.URL(s).host
        except httpx.InvalidURL:
            host = ""
        if host:
            return host
        # httpx couldn't extract a host (e.g. "http://") — fall through to manual
        # parsing, which fails toward an untrusted (non-allowlisted) destination.
    # Bare `host:port` form (no scheme). This never reaches httpx as a real outbound
    # request, but the generic PDP endpoint may receive it. Authority terminates at
    # the first `/`, `?`, or `#` (RFC 3986), not just `/`.
    s = re.split(r"[/?#]", s, maxsplit=1)[0]
    # Strip userinfo.
    if "@" in s:
        s = s.rsplit("@", 1)[1]
    # IPv6 literal in brackets, e.g. "[::1]:8443".
    if s.startswith("["):
        end = s.find("]")
        if end != -1:
            return s[: end + 1]
        return s
    # Strip port.
    if ":" in s:
        s = s.split(":", 1)[0]
    return s.lower()


def _is_loopback(host: str) -> bool:
    return host in _LOOPBACK_HOSTS


def _build_payload_text(arguments: dict[str, Any]) -> str:
    """Concatenate the egressing args (url + headers + body, ...), bounded."""
    parts = list(_iter_strings(arguments))
    return "\n".join(parts)[:_MAX_PAYLOAD_CHARS]


def evaluate(
    tool_name: str,
    arguments: dict[str, Any],
    tool_kind: str | None = None,
    allowlist: list[str] | None = None,
    private_repo_markers: list[str] | None = None,
) -> EgressVerdict:
    """Evaluate the decision matrix for one tool call.

    Returns an `EgressVerdict` with `decision` ∈ {"allow", "deny"} (only these two
    are implemented; `allow_with_conditions` is reserved in the schema).
    """
    allowlist = allowlist or []
    arguments = arguments or {}

    # --- In-scope test ---
    in_scope, url_like = _is_in_scope(tool_kind, arguments)
    if not in_scope:
        return EgressVerdict(
            decision="allow",
            reason="tool call does not look like network egress; out of scope for the "
                   "egress policy (permitted by omission)",
            policy="out-of-scope",
        )

    destination = _extract_host(url_like) if url_like else None

    # --- Destination axis ---
    if destination is not None and (
        _is_loopback(destination) or destination in allowlist
    ):
        dest_axis = "safe"
    else:
        dest_axis = "untrusted"

    # --- Payload axis ---
    payload_text = _build_payload_text(arguments)
    result = classify(payload_text, markers=private_repo_markers or ())

    # --- Decision matrix ---
    if dest_axis == "safe":
        return EgressVerdict(
            decision="allow",
            reason=f"destination {destination!r} is allowlisted; sensitivity "
                   f"({result.sensitivity}) does not leave the trust boundary",
            policy="network-egress",
            destination=destination,
            sensitivity=result.sensitivity,
            hit_types=result.hit_types,
        )

    if result.sensitivity is Sensitivity.NONE:
        return EgressVerdict(
            decision="allow",
            reason=f"destination {destination!r} is not allowlisted, but payload "
                   f"carries no sensitive content",
            policy="network-egress",
            destination=destination,
            sensitivity=result.sensitivity,
            hit_types=result.hit_types,
        )

    return EgressVerdict(
        decision="deny",
        reason=(
            f"destination {destination!r} is not allowlisted and payload carries "
            f"{result.sensitivity} ({', '.join(result.hit_types) or 'unspecified'})"
        ),
        policy="network-egress",
        destination=destination,
        sensitivity=result.sensitivity,
        hit_types=result.hit_types,
    )
