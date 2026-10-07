"""Heuristic injection scanner — weighted regex patterns over one piece of text.

A fast, always-available baseline: no model, no optional deps, ~0.3 ms. It runs only
when an operator *chooses* it. No availability fallback: a backend whose model will not
load refuses startup (`app.lifespan`). ``scan_text`` is the interface every backend
shares, so it stays stable even though the internals may be swapped or augmented.

Which text it is handed is `guards.scan`'s job, not this module's.
"""

from __future__ import annotations

import re

from agentgate.content import bounded_gap_abandoned
from agentgate.guards import FLAG_THRESHOLD, HARD_THRESHOLD, Verdict

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

# The two bounded-gap patterns above, decomposed for the abandon probe: (label, prefix,
# suffix, bound) — the bound is each pattern's `.{0,N}`, and text carrying the prefix
# with the suffix further away than N is what the bound stopped seeing. Recorded on the
# verdict (`gap_abandoned`), never a reason and never in the score. Same construction
# as `tool_inspector._GAP_PROBES`, so the limit is visible on both bounded-gap surfaces.
# Pinned against the patterns by `test_gap_probe_bounds_match_the_patterns`.
_GAP_PROBES: list[tuple[str, re.Pattern, re.Pattern, int]] = [
    ("exfil-secret",
     re.compile(r"\b(send|exfiltrate|post|upload|email)\b", re.I),
     re.compile(r"\b(api[_\s-]?key|secret|password|token|credentials?)\b", re.I), 40),
    ("pipe-to-shell",
     re.compile(r"\bcurl\b", re.I), re.compile(r"\|\s*(sh|bash)\b", re.I), 60),
]


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
    # Gap probe, only where a bounded pattern missed. Two linear searches; the score
    # above is final before this runs and is not touched by it.
    gap_abandoned: list[str] = []
    for label, prefix, suffix, bound in _GAP_PROBES:
        if label in reasons:
            continue
        distance = bounded_gap_abandoned(text, prefix, suffix, bound)
        if distance is not None:
            gap_abandoned.append(f"gap_abandoned:{label}:{distance}")
    return Verdict(
        flagged=score >= FLAG_THRESHOLD,
        score=score,
        reasons=reasons,
        hard=score >= HARD_THRESHOLD,
        gap_abandoned=gap_abandoned,
    )
