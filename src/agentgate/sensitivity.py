"""Sensitivity classifier — `{none, pii, secret, private_repo}`.

Single source of truth for **both** routing (sensitive content stays local, zero cloud
egress) and audit-retention policy.  Regex/entropy detection via `redaction.detect()`;
no LLM, no egress, runs inline on the hot path.

Precedence (most-sensitive wins): secret > private_repo > pii > none.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum

from agentgate.content import coerce_content, coerce_tool_call_args
from agentgate.redaction import detect


class Sensitivity(StrEnum):
    NONE = "none"
    PII = "pii"
    SECRET = "secret"
    PRIVATE_REPO = "private_repo"


# Secret hit_type names produced by redaction.detect() that map to SECRET sensitivity.
_SECRET_TYPES = frozenset({
    "private_key", "aws_access_key", "openai_key", "github_token",
    "slack_token", "google_api_key", "assignment", "high_entropy_token",
})


@dataclass
class ClassifyResult:
    sensitivity: Sensitivity
    hit_types: list[str] = field(default_factory=list)

    @property
    def is_sensitive(self) -> bool:
        return self.sensitivity is not Sensitivity.NONE


def classify(text: str, markers: Iterable[str] = ()) -> ClassifyResult:
    """Classify a single text blob. ``markers`` are configured private-repo indicators."""
    if not text:
        return ClassifyResult(Sensitivity.NONE)

    detected = detect(text)  # [(hit_type, count)]
    hit_type_names = [t for t, _ in detected]

    secret_hits = [t for t in hit_type_names if t in _SECRET_TYPES]
    if secret_hits:
        return ClassifyResult(Sensitivity.SECRET, secret_hits)

    repo_hits = [m for m in markers if m and m in text]
    if repo_hits:
        return ClassifyResult(Sensitivity.PRIVATE_REPO, ["marker"])

    pii_hits = [t for t in hit_type_names if t not in _SECRET_TYPES]
    if pii_hits:
        return ClassifyResult(Sensitivity.PII, pii_hits)

    return ClassifyResult(Sensitivity.NONE)


_SENSITIVITY_RANK = {
    Sensitivity.NONE: 0,
    Sensitivity.PII: 1,
    Sensitivity.PRIVATE_REPO: 2,
    Sensitivity.SECRET: 3,
}


def _merge_results(left: ClassifyResult, right: ClassifyResult) -> ClassifyResult:
    """Return the more-sensitive result, combining same-tier evidence without duplicates."""
    left_rank = _SENSITIVITY_RANK[left.sensitivity]
    right_rank = _SENSITIVITY_RANK[right.sensitivity]
    if left_rank > right_rank:
        return left
    if right_rank > left_rank:
        return right
    return ClassifyResult(left.sensitivity, list(dict.fromkeys(left.hit_types + right.hit_types)))


def classify_request(messages: list[dict], markers: Iterable[str] = (), *, max_chars: int = 20000) -> ClassifyResult:
    """Classify the content that would egress (all message text, bounded).

    Tool-call arguments count as message text: they egress with the request like
    anything else, and on an assistant tool-call message they are the only text there.

    They are appended *after* all message content rather than interleaved with it, and
    only into what the window has left. Interleaved, one ordinary 20 KB `write_file` call
    pushed every later message past `max_chars`, so a secret or a private-repo marker that
    used to classify stopped being seen — a widening that produced misses. Content keeps
    the whole window it had before tool calls were read at all; arguments get the
    remainder. A marker matters most here: redaction is not a backstop for it, so a miss
    routes proprietary content to cloud.
    """
    marker_tuple = tuple(markers)
    content = "\n".join(coerce_content(m.get("content")) for m in messages)[:max_chars]
    result = classify(content, marker_tuple)
    remaining = max_chars - len(content)
    if remaining > 0:
        args = "\n".join(t for t in (coerce_tool_call_args(m) for m in messages) if t)
        if args:
            # Classify the two surfaces separately, then merge by the documented
            # precedence. A synthetic separator cannot consume one character of the
            # payload budget, the regex input never exceeds `max_chars` in total, and no
            # detector can manufacture a match by straddling the content/arguments edge.
            result = _merge_results(result, classify(args[:remaining], marker_tuple))
    return result
