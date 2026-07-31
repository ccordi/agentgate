"""Heuristic injection scanner — weighted regex patterns over one piece of text.

A fast, always-available baseline: no model, no optional deps, ~0.3 ms. It is what
the gateway falls back to when the model guard is unavailable, and it is the detector
the red-team harness (eval/redteam) drives by default, so ``scan_text`` is stable.

Which text it is handed is `guards.scan`'s job, not this module's.
"""

from __future__ import annotations

import re

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
