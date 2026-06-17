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

from .injection import _coerce_content
from .redaction import detect


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


def classify_request(messages: list[dict], markers: Iterable[str] = (), *, max_chars: int = 20000) -> ClassifyResult:
    """Classify the content that would egress (all message text, bounded)."""
    joined = "\n".join(_coerce_content(m.get("content")) for m in messages)
    return classify(joined[:max_chars], markers)
