"""Corpus item schema — the on-disk JSONL contract shared across the harness.

One JSON object per line. Deliberately file-based (not a DB): the gold set must be
versioned, diffable, and human-reviewable, which is git's job.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class LabelOrigin(StrEnum):
    """Where an item's label came from — the provenance half of the labeling chain."""

    KNOWN = "known"      # source guarantees the label (authored payloads, labeled datasets)
    JUDGE = "judge"      # independent LLM judge assigned it
    HUMAN = "human"      # human gold-set label (validates the judge)
    UNLABELED = ""       # captured but not yet labeled (e.g. raw FP capture)


# 1 = injection / adversarial, 0 = benign. ``None`` = not yet labeled.
Label = int


class CorpusItem(BaseModel):
    """A single evaluation example.

    ``id`` is a stable content hash so the same text always maps to the same id across
    sources/runs (dedup + judge-cache keying). ``category`` tags positives with their
    attack family (mirrors ``injection.py``'s pattern labels) so we can report
    per-category recall — including the evasions the heuristics are expected to miss.
    """

    id: str = ""
    source: str
    text: str
    label: Label | None = None
    label_origin: LabelOrigin = LabelOrigin.UNLABELED
    category: str | None = None
    meta: dict = Field(default_factory=dict)
    sensitivity: str | None = None
    expected_action: str | None = None
    planted_secret: str | None = None
    secret_span: list[int] | None = None

    @field_validator("label")
    @classmethod
    def _label_is_binary(cls, v: Label | None) -> Label | None:
        if v is not None and v not in (0, 1):
            raise ValueError(f"label must be 0, 1, or None; got {v!r}")
        return v

    def model_post_init(self, _ctx) -> None:
        if not self.id:
            # Bypass validation-on-assignment cost; id is derived, not user input.
            object.__setattr__(self, "id", stable_id(self.text))

    @property
    def is_labeled(self) -> bool:
        return self.label is not None


def stable_id(text: str) -> str:
    """Deterministic short id from item text (dedup + cache key)."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
