"""PII/secret detection and egress redaction.

Single source of truth for detector primitives consumed by both the sensitivity
classifier and the cloud-egress redaction path.  No I/O, no model calls — regex +
entropy only, runs inline on the hot path.

Public surface:
  detect(text)  → [(hit_type, count)]          # shared primitive
  redact(text)  → RedactionResult              # returns the text with matched spans replaced
  mask_url_password(url) → str                 # operator-facing log/exception text
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit, urlunsplit

# ---------------------------------------------------------------------------
# Detector patterns
# ---------------------------------------------------------------------------

# High-confidence secret shapes (prefix-anchored → very low false-positive rate).
SECRET_PATTERNS: list[tuple[re.Pattern, str]] = [
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

PII_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Require an alphabetic TLD (>=2) so version pins / IPs like `cache@v5.0.5` or
    # `svc@10.0.0.1` aren't read as emails (real `.com`/`.invalid` keep matching).
    # Every quantifier is bounded. `.` is inside the local-part class, so an unbounded `+`
    # hands the engine a fresh start position every other character of a dot-rich run —
    # quadratic, on the event loop, on attacker-chosen input. The bounds sit above anything
    # real (RFC 5321 caps a local part at 64, DNS labels at 63), so matches are unchanged.
    (re.compile(r"\b[\w.+-]{1,128}@[\w-]{1,63}(?:\.[\w-]{1,63}){0,12}\.[A-Za-z]{2,63}\b"), "email"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "ssn"),
    (re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "phone"),
    # Reject a 13-16 digit run that is embedded in a longer numeric/decimal literal
    # (e.g. the fractional part of a float score `0.5842405849569906`).
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

# The shortest bare number any detector above can match. The shared JSON lexeme walker
# (`content.map_json_lexemes` — the structural redactor and the sensitivity classifier
# both read through it) scans a JSON number as its lexeme — digits, sign, `.`,
# exponent — and nothing here matches one under 10 characters: phone needs 10 digits (11
# with a leading 1), card 13–16, SSN needs dashes, email an `@`, the secret and assignment
# shapes a word or prefix, `_TOKEN_RE` / `_B64_BLOB_RE` 32+ characters — and pure digits
# carry log2(10) ≈ 3.32 bits per character, under `_ENTROPY_BITS` at any length. So a
# caller may skip `redact()` for shorter numeric lexemes losslessly; the walker does,
# which is what keeps numeric-heavy JSON (an array of small ints) from paying the
# detectors' cost per element. Pinned by
# `test_short_numeric_lexemes_match_no_detector`: if a detector ever matches a shorter
# bare number, that test fails and this constant must drop with it.
MIN_SCANNABLE_NUMBER_CHARS = 10


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

    for pat, name in SECRET_PATTERNS:
        matches = pat.findall(text)
        if matches:
            hits[name] = hits.get(name, 0) + len(matches)

    # High-entropy token check (one flag per text, not per token). Two passes: the
    # generic alnum/underscore/dash run, plus base64 blobs that embed `+`/`/`.
    if any(_shannon_bits(tok) >= _ENTROPY_BITS for tok in _TOKEN_RE.findall(text)) or any(
        _is_b64_secret(tok) for tok in _B64_BLOB_RE.findall(text)
    ):
        hits["high_entropy_token"] = hits.get("high_entropy_token", 0) + 1

    for pat, name in PII_PATTERNS:
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
    for pat, name in SECRET_PATTERNS:
        out = pat.sub(_make_sub(name), out)

    # PII second (some spans may already be gone)
    for pat, name in PII_PATTERNS:
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


# ---------------------------------------------------------------------------
# Connection URLs in operator-facing text
# ---------------------------------------------------------------------------

def mask_url_password(url: str) -> str:
    """``url`` with the password in its userinfo replaced by ``***``.

    Startup announces which audit store and which limits backend came up, and the
    shared-limits refusal names the Redis it could not reach — all by interpolating the
    configured URL. A URL with credentials in it (`postgresql+asyncpg://user:pw@host/db`,
    `redis://user:pw@host:6379/0`) would therefore write the operator's password into the
    log file, and into whatever collects the traceback; a log file is readable by more
    people, for longer, than the config that holds the URL.

    Only the password is touched: scheme, user, host and path stay readable, because the
    line exists to tell the operator *which* backend this is. A URL with no userinfo —
    the default `sqlite+aiosqlite:///./data/agentgate.db` among them — comes back
    byte-identical, unparsed by anything that could mangle a filesystem path. For text a
    human reads only; the value used to connect is always the original string.
    """
    try:
        split = urlsplit(url)
    except ValueError:
        # An unparseable URL cannot be masked, and echoing it is the thing to avoid.
        return "<unparseable url>"
    if split.password is None:
        return url
    userinfo, _, host = split.netloc.rpartition("@")
    user = userinfo.partition(":")[0]
    return urlunsplit(split._replace(netloc=f"{user}:***@{host}"))
