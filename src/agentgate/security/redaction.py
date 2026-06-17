"""PII/secret detection and egress redaction.

Single source of truth for detector primitives consumed by both the sensitivity
classifier and the cloud-egress redaction path.  No I/O, no model calls — regex +
entropy only, runs inline on the hot path.

Public surface:
  detect(text)  → [(hit_type, count)]          # shared primitive
  redact(text)  → RedactionResult              # rewrites spans in-place
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Detector patterns
# ---------------------------------------------------------------------------

# High-confidence secret shapes (prefix-anchored → very low false-positive rate).
_SECRET_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----"), "private_key"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "aws_access_key"),
    (re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"), "openai_key"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"), "github_token"),
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"), "slack_token"),
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"), "google_api_key"),
    (
        re.compile(r"(?i)\b(api[_-]?key|secret|password|token)\b\s*[:=]\s*['\"][^'\"]{8,}['\"]"),
        "assignment",
    ),
]

_PII_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Require an alphabetic TLD (>=2) so version pins / IPs like `cache@v5.0.5` or
    # `svc@10.0.0.1` aren't read as emails (2b over-fire fix; real `.com`/`.invalid` keep matching).
    (re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}\b"), "email"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "ssn"),
    (re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "phone"),
    # Reject a 13-16 digit run that is embedded in a longer numeric/decimal literal
    # (e.g. the fractional part of a float score `0.5842405849569906`); 2b over-fire fix.
    (re.compile(r"(?<![\d.])\b(?:\d[ -]?){13,16}\b(?![\d.])"), "card_number"),
]

# Generic high-entropy token → likely secret. Charset deliberately EXCLUDES `+ /` so
# that slash-delimited URL paths break into short sub-threshold segments and are not
# flagged (a measured false-positive guard — `…/assets/images/header-banner` must stay
# unflagged). Pure-alnum base64 and JWTs are caught here.
_TOKEN_RE = re.compile(r"\b[A-Za-z0-9_\-]{32,}\b")

# Base64 / base64url secret blobs that embed `+` or `/` — AWS secret access keys, GCP /
# base64 service-account material. `_TOKEN_RE` splits these on the slashes (e.g.
# `wJalr…/K7MD…/bPxR…` → three sub-32 fragments), so the secret would classify as NONE
# and egress unredacted. Match the whole blob, but bound the slash count: a real
# credential carries at most a couple of base64 `/`s, whereas a URL path is dominated by
# `/`-separated dictionary words (so URL paths stay unflagged). See `_is_b64_secret`.
_B64_BLOB_RE = re.compile(r"[A-Za-z0-9+/]{32,}={0,2}")
_B64_MAX_SLASHES = 3
_ENTROPY_BITS = 4.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _shannon_bits(s: str) -> float:
    if not s:
        return 0.0
    counts = {c: s.count(c) for c in set(s)}
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _is_b64_secret(tok: str) -> bool:
    """A `_B64_BLOB_RE` match is a likely secret only if it actually carries base64
    `+`/`/` (pure-alnum blobs are already handled by `_TOKEN_RE`), has few slashes (so
    `/`-heavy URL paths are excluded), and is high-entropy."""
    return (
        ("+" in tok or "/" in tok)
        and tok.count("/") <= _B64_MAX_SLASHES
        and _shannon_bits(tok) >= _ENTROPY_BITS
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class RedactionResult:
    redacted_text: str
    hit_count: int = 0
    hit_types: list[dict] = field(default_factory=list)  # [{"type": "email", "count": 2}, ...]

    @property
    def found(self) -> bool:
        return self.hit_count > 0


def detect(text: str) -> list[tuple[str, int]]:
    """Return [(hit_type, count)] across secret + PII patterns + high-entropy tokens.

    Secret patterns are tested first (secret > pii precedence).  Pure — no I/O.
    """
    hits: dict[str, int] = {}

    for pat, name in _SECRET_PATTERNS:
        matches = pat.findall(text)
        if matches:
            hits[name] = hits.get(name, 0) + len(matches)

    # High-entropy token check (one flag per text, not per token). Two passes: the
    # generic alnum/underscore/dash run, plus base64 blobs that embed `+`/`/`.
    if any(_shannon_bits(tok) >= _ENTROPY_BITS for tok in _TOKEN_RE.findall(text)) or any(
        _is_b64_secret(tok) for tok in _B64_BLOB_RE.findall(text)
    ):
        hits["high_entropy_token"] = hits.get("high_entropy_token", 0) + 1

    for pat, name in _PII_PATTERNS:
        matches = pat.findall(text)
        if matches:
            hits[name] = hits.get(name, 0) + len(matches)

    return list(hits.items())


def redact(text: str) -> RedactionResult:
    """Replace each matched span with '[REDACTED:<type>]'.

    Runs secret patterns before PII so a span that matches both gets the more
    specific secret label.  Returns the rewritten text + per-type hit counts.
    """
    if not text:
        return RedactionResult(redacted_text=text)

    counts: dict[str, int] = {}

    def _make_sub(name: str):
        placeholder = f"[REDACTED:{name}]"
        def _sub(m: re.Match) -> str:
            counts[name] = counts.get(name, 0) + 1
            return placeholder
        return _sub

    out = text

    # Secrets first
    for pat, name in _SECRET_PATTERNS:
        out = pat.sub(_make_sub(name), out)

    # PII second (some spans may already be gone)
    for pat, name in _PII_PATTERNS:
        out = pat.sub(_make_sub(name), out)

    # High-entropy tokens (replace each matching token)
    def _entropy_sub(m: re.Match) -> str:
        tok = m.group(0)
        if _shannon_bits(tok) >= _ENTROPY_BITS:
            counts["high_entropy_token"] = counts.get("high_entropy_token", 0) + 1
            return "[REDACTED:high_entropy_token]"
        return tok

    out = _TOKEN_RE.sub(_entropy_sub, out)

    # Base64 secret blobs embedding `+`/`/` (e.g. AWS secret keys), which `_TOKEN_RE`
    # splits and misses. The placeholder left by the pass above is not base64 (`:` `[` `]`),
    # so it cannot be re-matched here.
    def _b64_sub(m: re.Match) -> str:
        tok = m.group(0)
        if _is_b64_secret(tok):
            counts["high_entropy_token"] = counts.get("high_entropy_token", 0) + 1
            return "[REDACTED:high_entropy_token]"
        return tok

    out = _B64_BLOB_RE.sub(_b64_sub, out)

    hit_types = [{"type": t, "count": c} for t, c in counts.items()]
    total = sum(counts.values())
    return RedactionResult(redacted_text=out, hit_count=total, hit_types=hit_types)
