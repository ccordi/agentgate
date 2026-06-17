"""Unit tests for the sensitivity corpus."""

from __future__ import annotations

from eval.redteam.loader import SENSITIVITY_CORPUS, load_jsonl


def test_sensitivity_corpus_exists_and_loads():
    assert SENSITIVITY_CORPUS.exists(), f"Sensitivity corpus file {SENSITIVITY_CORPUS} does not exist"
    assert SENSITIVITY_CORPUS.is_file()

    items = list(load_jsonl(SENSITIVITY_CORPUS))
    assert len(items) == 150, f"Expected 150 items in corpus, got {len(items)}"

    public_items = [i for i in items if i.sensitivity == "public"]
    sensitive_doc_items = [i for i in items if i.sensitivity == "sensitive_doc"]
    secret_bearing_items = [i for i in items if i.sensitivity == "secret_bearing"]

    assert len(public_items) == 50, f"Expected 50 public items, got {len(public_items)}"
    assert len(sensitive_doc_items) == 50, f"Expected 50 sensitive_doc items, got {len(sensitive_doc_items)}"
    assert len(secret_bearing_items) == 50, f"Expected 50 secret_bearing items, got {len(secret_bearing_items)}"

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
