"""Egress PDP — policy logic for `POST /a/egress/decision`.

Implements the egress decision matrix. A thin composition of two
signals, with **no detection code of its own**:

- In-scope test — is this tool call a network egress at all?
- Destination axis — allowlisted vs. untrusted (loopback always allowed).
- Payload axis — reuse `sensitivity.classify()` (regex, no LLM call).
- Decision matrix — allow / deny only (`allow_with_conditions` is reserved,
  unimplemented).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from agentgate.sensitivity import Sensitivity, classify

# The window the payload is classified over. A secret padded past it is invisible to the
# sensitivity axis, which flips deny to allow on a non-allowlisted destination — the
# case this PDP exists to enforce. So it is set as wide as the classifier can scan
# cheaply rather than to match sensitivity.py: the detection patterns are linear, so
# 1 MB costs ~95 ms of prose and ~460 ms of pathological input. Still bounded —
# classification is not free, and an unbounded window is its own denial of service.
#
# classify_request keeps its own 20k default: widening the *inbound* window changes
# what routes local vs cloud, which is a separate decision.
_MAX_PAYLOAD_CHARS = 1_000_000

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
    # Closed-vocabulary caveats for the audit row (`RequestRecord.caveats`). One kind on
    # this path: `truncated:egress_payload:<n>` — the payload axis classified the first n
    # characters of the joined arguments and stopped. Stamped on every decision-matrix
    # outcome; empty when the payload was never classified (out of scope, own-origin
    # deny). No `rejected:` value here — `finish_reason`/`status` already say why.
    caveats: list[str] = field(default_factory=list)


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
    (`egress/pep.py:_perform_outbound`). This is a security invariant, not
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


def is_loopback_host(value: str) -> bool:
    """Whether a bind address or `Host` header value names the loopback interface.

    The project-wide definition of "loopback" — the same `_LOOPBACK_HOSTS` set that
    drives the destination axis above. One definition on purpose: if the startup bind
    check and the egress policy ever disagreed about what counts as loopback (say,
    `127.0.0.2`), the two checks would silently drift apart.
    Anything outside the set is treated as non-loopback and handled strictly —
    under-inclusion fails closed for both callers (a bind refuses to start; a Host
    header is rejected).

    Accepts the bare forms a bind config uses ("127.0.0.1", "::1") and the
    authority forms a Host header uses ("127.0.0.1:4100", "[::1]:4100"). `0.0.0.0` is
    deliberately NOT accepted — it binds every interface, and refusing that bind is the
    entire point of the check (`validate_runtime_settings`). Do not "fix" this by adding
    it to `_LOOPBACK_HOSTS`: that one edit reopens the wide bind and the DNS-rebinding
    Host guard together. `test_startup_posture.py` pins the False.
    """
    s = value.strip().lower()
    if s in _LOOPBACK_HOSTS:
        return True
    return _is_loopback(_extract_host(s))


def _extract_port(url_or_host: str) -> int | None:
    """Best-effort port for the gateway-origin carve-out below. None when absent."""
    s = url_or_host.strip()
    if _URL_RE.match(s):
        try:
            return httpx.URL(s).port
        except httpx.InvalidURL:
            return None
    s = re.split(r"[/?#]", s, maxsplit=1)[0]
    if "@" in s:
        s = s.rsplit("@", 1)[1]
    if s.startswith("["):  # [::1]:8443
        end = s.find("]")
        s = s[end + 1:] if end != -1 else ""
        return int(s[1:]) if s.startswith(":") and s[1:].isdigit() else None
    if ":" in s:
        tail = s.split(":", 1)[1]
        return int(tail) if tail.isdigit() else None
    return None


def _is_own_origin(host: str, port: int | None, gateway_origin: tuple[str, int] | None) -> bool:
    """Whether this destination is the gateway's own listener.

    The loopback rule makes every loopback destination `dest_axis = "safe"`, which
    short-circuits the sensitivity axis entirely. Composed with the fact that the
    governed agent is on the host by construction, that would make the agent's ONLY
    sanctioned network path reach the gateway's own control plane: DELETE its own kill
    switch, or proxy to the cloud under the operator's pass-through credential.

    Loopback stays allowed — its blast radius just stops including the policy engine
    itself. An operator who genuinely wants the agent talking to the gateway can name
    the origin in `egress.allowlist` explicitly.
    """
    if gateway_origin is None or port is None:
        return False
    own_host, own_port = gateway_origin
    if port != own_port:
        return False
    # 0.0.0.0 binds every interface, so a gateway bound there owns every loopback alias.
    return _is_loopback(host) and (own_host in ("0.0.0.0", "::") or host == own_host
                                   or _is_loopback(own_host))


def _build_payload_text(arguments: dict[str, Any]) -> tuple[str, bool]:
    """Concatenate the egressing args (url + headers + body, ...), bounded.

    Returns the window and whether the join overran it. The full join exists before
    the slice, so the second is one length comparison.
    """
    text = "\n".join(_iter_strings(arguments))
    return text[:_MAX_PAYLOAD_CHARS], len(text) > _MAX_PAYLOAD_CHARS


def evaluate(
    tool_name: str,
    arguments: dict[str, Any],
    tool_kind: str | None = None,
    allowlist: list[str] | None = None,
    private_repo_markers: list[str] | None = None,
    gateway_origin: tuple[str, int] | None = None,
) -> EgressVerdict:
    """Evaluate the decision matrix for one tool call.

    Returns an `EgressVerdict` with `decision` ∈ {"allow", "deny"} (only these two
    are implemented; `allow_with_conditions` is reserved in the schema).

    Destination extraction takes the **first** URL-shaped string leaf in `arguments`,
    walking dict values in insertion order (`_iter_strings`). The PEP satisfies that by
    construction — `safe_http_request` builds the payload from the same local `url` it
    later contacts — but this endpoint is generic: **a non-PEP caller must order its
    arguments so the destination it will actually contact is the first URL-like leaf**,
    or the verdict describes a different host than the call. There is no capability token
    in this protocol, so the PDP cannot detect a client that consults it and then does
    something else; enforcement is entirely the PEP's control flow.

    `gateway_origin` is the gateway's own (host, port). Passed by the endpoint so the
    loopback allowance stops covering the policy engine itself — see `_is_own_origin`.
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
    dest_port = _extract_port(url_like) if url_like else None

    # --- Destination axis ---
    # Loopback is always allowed. The one exception is the gateway's OWN origin.
    # This has to be a hard deny rather than a demotion to "untrusted", because
    # the untrusted branch still allows a benign payload — and the calls that matter here
    # (DELETE /admin/kill/<id>, DELETE /admin/keys/<id>) carry no sensitive content at all, so
    # a demotion would let every one of them straight through. An operator who genuinely
    # wants the agent talking to the gateway names the origin in `egress.allowlist`.
    origin_str = f"{destination}:{dest_port}" if dest_port is not None else destination
    if (
        destination is not None
        and _is_own_origin(destination, dest_port, gateway_origin)
        and origin_str not in allowlist
        and destination not in allowlist
    ):
        return EgressVerdict(
            decision="deny",
            reason=(
                f"destination {origin_str!r} is the gateway's own control plane; the "
                f"loopback allowance does not extend to the policy engine itself "
                f"(add it to egress.allowlist to override)"
            ),
            policy="gateway-self-egress",
            destination=destination,
            sensitivity=None,
            hit_types=[],
        )

    if destination is not None and (
        _is_loopback(destination) or destination in allowlist
    ):
        dest_axis = "safe"
    else:
        dest_axis = "untrusted"

    # --- Payload axis ---
    payload_text, cut = _build_payload_text(arguments)
    result = classify(payload_text, markers=private_repo_markers or ())
    # Stamped ahead of the matrix so every outcome carries it: an allow past the window
    # is exactly the row that needs the caveat (test_classification_window_is_still_bounded
    # shows such a call is allowed; this makes it visible per request).
    caveats = [f"truncated:egress_payload:{_MAX_PAYLOAD_CHARS}"] if cut else []

    # --- Decision matrix ---
    if dest_axis == "safe":
        return EgressVerdict(
            decision="allow",
            reason=f"destination {destination!r} is loopback or allowlisted; sensitivity "
                   f"({result.sensitivity}) does not leave the trust boundary",
            policy="network-egress",
            destination=destination,
            sensitivity=result.sensitivity,
            hit_types=result.hit_types,
            caveats=caveats,
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
            caveats=caveats,
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
        caveats=caveats,
    )
