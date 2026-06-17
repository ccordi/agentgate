"""Inbound prompt-injection / tool-output scanner.

Heuristic weighted-pattern scanner — a fast, always-available baseline that the
gateway uses when the model-based guard is unavailable or not configured. The same
``scan_text`` entry point is exercised by the red-team evaluation harness
(eval/redteam), so the interface is stable.

What it scans: the untrusted content in a request — the whole trailing batch of
tool-output messages (``role: "tool"`` and the legacy ``role: "function"``, so
parallel/batched tool calls are all covered) plus the newest user content.
System/assistant turns are trusted and skipped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# (compiled pattern, weight, label). Weights sum (capped at 1.0) into a score.
# Curated common injection signals — intentionally small and legible.
_PATTERNS: list[tuple[re.Pattern, float, str]] = [
    (re.compile(r"\bignore\s+(all\s+)?(the\s+|your\s+|prior\s+)?previous\s+(instructions|prompts?)\b", re.I), 0.6, "ignore-previous"),
    (re.compile(r"\bdisregard\s+(all\s+)?(your\s+|the\s+|previous\s+|prior\s+)?(instructions|rules|guidelines)\b", re.I), 0.6, "disregard-instructions"),
    (re.compile(r"\byou\s+are\s+now\s+(a|an|in)\b", re.I), 0.4, "role-reassign"),
    (re.compile(r"\b(reveal|print|repeat|show)\s+(your\s+)?(system\s+prompt|instructions|initial\s+(prompt|instructions))\b", re.I), 0.6, "prompt-exfil"),
    (re.compile(r"\b(ignore|override)\s+(your\s+)?(safety|guard)", re.I), 0.6, "safety-override"),
    (re.compile(r"\bnew\s+instructions?\s*:\s*", re.I), 0.4, "new-instructions"),
    (re.compile(r"<\s*\|?\s*/?\s*(system|im_start|im_end)\s*\|?\s*>", re.I), 0.5, "fake-control-tokens"),
    (re.compile(r"\bdeveloper\s+mode\b", re.I), 0.4, "developer-mode"),
    (re.compile(r"\b(send|exfiltrate|post|upload|email)\b.{0,40}\b(api[_\s-]?key|secret|password|token|credentials?)\b", re.I), 0.5, "exfil-secret"),
    (re.compile(r"\bcurl\b.{0,60}\|\s*(sh|bash)\b", re.I), 0.4, "pipe-to-shell"),
]

# Score at/above this blocks the request inbound (cheap, fully-buffered side).
HARD_THRESHOLD = 0.7
# Score at/above this flags-and-logs but forwards.
FLAG_THRESHOLD = 0.4


@dataclass
class Verdict:
    flagged: bool
    score: float
    reasons: list[str]
    hard: bool  # True -> block the request inbound

    @classmethod
    def clean(cls) -> Verdict:
        return cls(flagged=False, score=0.0, reasons=[], hard=False)


def scan_text(text: str) -> Verdict:
    """Score a single piece of untrusted text against the heuristic patterns."""
    if not text:
        return Verdict.clean()
    score = 0.0
    reasons: list[str] = []
    for pattern, weight, label in _PATTERNS:
        if pattern.search(text):
            score += weight
            reasons.append(label)
    score = min(score, 1.0)
    return Verdict(
        flagged=score >= FLAG_THRESHOLD,
        score=score,
        reasons=reasons,
        hard=score >= HARD_THRESHOLD,
    )


def _coerce_content(content) -> str:
    """OpenAI content is either a string or a list of typed parts; flatten to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and isinstance(p.get("text"), str):
                parts.append(p["text"])
        return "\n".join(parts)
    return ""


# Roles carrying tool output. Both the modern OpenAI `tool` role and the legacy
# `function` role (still emitted by many OpenAI-compatible clients/SDKs) are
# attacker-influenced channels and MUST be scanned — otherwise an injection delivered
# as `{"role": "function", ...}` reaches the model on an unscanned channel.
_TOOL_OUTPUT_ROLES = frozenset({"tool", "function"})


def trailing_tool_outputs(messages: list[dict]) -> list[dict]:
    """The contiguous run of tool-output messages at the tail of the conversation —
    i.e. the current turn's tool results, which may be a *parallel/batched* set
    (`assistant(tool_calls=[a,b,c]) → tool(a), tool(b), tool(c)`).

    Scanning only the single last tool message misses every result but the last when a
    harness emits parallel tool calls (the "A2 split/batch-payload" gap). Scanning the
    whole trailing run closes that, and each batch is still scanned exactly once: on the
    next turn these results sit behind an assistant message, so the trailing run becomes
    the *new* results, not the accumulated history.
    """
    block: list[dict] = []
    for m in reversed(messages):
        if m.get("role") in _TOOL_OUTPUT_ROLES:
            block.append(m)
        else:
            break
    block.reverse()
    return block


def extract_untrusted(messages: list[dict]) -> list[tuple[str, str]]:
    """Return [(source, text)] for the untrusted content in a request.

    Untrusted = the current turn's tool-output messages (`tool`/`function` role, often
    attacker-influenced — the whole trailing batch, so parallel tool calls are covered)
    and the newest user message. Earlier turns were already scanned on prior requests.
    """
    out: list[tuple[str, str]] = []
    for m in trailing_tool_outputs(messages):
        out.append(("tool_output", _coerce_content(m.get("content"))))
    last_user = next(
        (m for m in reversed(messages) if m.get("role") == "user"), None
    )
    if last_user is not None:
        out.append(("user", _coerce_content(last_user.get("content"))))
    return out


def scan_request(messages: list[dict]) -> Verdict:
    """Scan a request's untrusted content; return the strongest verdict."""
    worst = Verdict.clean()
    for source, text in extract_untrusted(messages):
        v = scan_text(text)
        if v.score > worst.score:
            worst = Verdict(v.flagged, v.score, [f"{source}:{r}" for r in v.reasons], v.hard)
    return worst
