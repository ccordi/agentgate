"""Data-integrity tests for the committed sensitivity corpus.

Pins the agreement between the committed JSONL and the generator that produces it — tier
sizes, per-tier field shape, and the planted-secret spans. It does NOT re-run the
generator (that is `gen sensitivity`'s own byte-diff check); it asserts that what is on
disk still matches what the generator's constants say should be there, so an edited corpus
or a drifted generator shows up as a test failure rather than a silently wrong eval.
"""

from __future__ import annotations

from eval.redteam.gen.sensitivity_corpus import TIER_SIZE, TIERS
from eval.redteam.loader import SENSITIVITY_CORPUS, load_jsonl


def test_sensitivity_corpus_exists_and_loads():
    assert SENSITIVITY_CORPUS.exists(), f"Sensitivity corpus file {SENSITIVITY_CORPUS} does not exist"
    assert SENSITIVITY_CORPUS.is_file()

    items = list(load_jsonl(SENSITIVITY_CORPUS))
    expected_total = TIER_SIZE * len(TIERS)
    assert len(items) == expected_total, f"Expected {expected_total} items in corpus, got {len(items)}"

    by_tier = {t: [i for i in items if i.sensitivity == t] for t in TIERS}
    for tier, tier_items in by_tier.items():
        assert len(tier_items) == TIER_SIZE, (
            f"Expected {TIER_SIZE} {tier} items, got {len(tier_items)}"
        )
    public_items, sensitive_doc_items, secret_bearing_items = (by_tier[t] for t in TIERS)

    for i in items:
        assert i.source == "sensitivity_corpus"
        assert i.label == 0
        assert i.label_origin == "known"

    for i in public_items:
        assert i.category == "sensitivity_public"
        assert i.expected_action == "route_cloud"
        assert i.planted_secret is None
        assert i.secret_span is None

    for i in sensitive_doc_items:
        assert i.category == "sensitivity_sensitive_doc"
        assert i.expected_action == "route_local"
        assert i.planted_secret is None
        assert i.secret_span is None

    for i in secret_bearing_items:
        assert i.category == "sensitivity_secret_bearing"
        assert i.expected_action == "redact_and_route_cloud"
        assert i.planted_secret is not None
        assert i.secret_span is not None
        assert len(i.secret_span) == 2
        
        # Verify span extraction matches planted secret
        start, end = i.secret_span
        extracted = i.text[start:end]
        assert extracted == i.planted_secret, (
            f"Span mismatch for item {i.id}: "
            f"text[{start}:{end}] = {extracted!r} vs planted = {i.planted_secret!r}"
        )
