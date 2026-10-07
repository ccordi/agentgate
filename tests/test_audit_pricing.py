"""Audit store + pricing tests."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import logging
import subprocess
import sys
import uuid
from datetime import timedelta
from pathlib import Path

import pytest

from agentgate import pricing
from agentgate.audit.models import ContentSample, RequestRecord
from agentgate.audit.store import AuditStore, ContentSampleAudit, RequestAudit, utcnow
from agentgate.observability import metrics
from agentgate.pricing import estimate_cost_usd
from tests.support import make_audit

REPO_ROOT = Path(pricing.__file__).resolve().parents[2]
REFRESH_SCRIPT = REPO_ROOT / "scripts" / "refresh_price_map.py"


def _unknown_count() -> float:
    return metrics.price_unknown_model_total._value.get()


def test_pricing_known_and_unknown():
    # Prices come from the vendored LiteLLM map (USD per token).
    # 1M prompt + 1M completion on gemini-2.5-flash-lite = 0.10 + 0.40
    assert estimate_cost_usd("gemini-2.5-flash-lite", 1_000_000, 1_000_000) == pytest.approx(0.50)
    assert estimate_cost_usd("gemini-3-flash-preview", 1_000_000, 0) == pytest.approx(0.50)
    assert estimate_cost_usd("gemini-3-flash-preview", 0, 1_000_000) == pytest.approx(3.00)
    # A model priced straight from the map.
    assert estimate_cost_usd("gpt-4o", 1_000_000, 1_000_000) == pytest.approx(12.50)
    # Unknown / local models are free ($0; the loud-path details are covered below).
    assert estimate_cost_usd("llama3.2", 1_000_000, 1_000_000) == 0.0
    assert estimate_cost_usd(None, 10, 10) == 0.0


def test_pricing_prefix_tolerates_version_suffixes():
    # A versioned name the map doesn't list resolves via the longest bare-key
    # prefix — the -lite entry (0.50/2M), not the shorter, pricier
    # "gemini-2.5-flash" (2.80/2M).
    assert estimate_cost_usd(
        "gemini-2.5-flash-lite-zzz", 1_000_000, 1_000_000
    ) == pytest.approx(0.50)


def test_pricing_provider_prefixed_keys():
    # "codestral-2508" exists only as "mistral/codestral-2508" in the map —
    # the suffix is still resolvable (3e-07 + 9e-07 per token).
    assert estimate_cost_usd("codestral-2508", 1_000_000, 1_000_000) == pytest.approx(1.20)
    # Bare key wins over a provider-prefixed sibling: "gpt-4o-mini" prices at
    # the bare/OpenAI row (0.75/2M), not azure/gpt-4o-mini's +10% (0.825/2M).
    assert estimate_cost_usd("gpt-4o-mini", 1_000_000, 1_000_000) == pytest.approx(0.75)


def test_price_alias_collision_prefers_the_priced_entry():
    """A $0 provider row must not claim an alias a paid row also spells.

    The vendored map lists `codestral/codestral-latest` at 0/0 ahead of
    `mistral/codestral-latest`, so first-in-file suffix indexing would price
    `codestral-latest` at $0 on every cloud request, and silently: a resolved
    alias takes no unknown-model warning or counter. That is spend the USD cap cannot
    see. Over-pricing an alias only trips the cap early; under-pricing lets spend past
    it, so the priced row wins the collision.
    """
    pricing._price_table.cache_clear()  # the vendored map, whatever a prior test loaded
    try:
        assert pricing._price_table()["codestral-latest"][2] == "mistral/codestral-latest"
        assert estimate_cost_usd(
            "codestral-latest", 1_000_000, 1_000_000) == pytest.approx(4.0)
    finally:
        pricing._price_table.cache_clear()


def test_pricing_unknown_model_is_loud(caplog):
    pricing._warned_unknown.discard("no-such-model-r1-test")
    before = _unknown_count()
    with caplog.at_level(logging.WARNING, logger="agentgate"):
        assert estimate_cost_usd("no-such-model-r1-test", 10, 10) == 0.0
        assert estimate_cost_usd("no-such-model-r1-test", 10, 10) == 0.0
    # Counted on every $0 lookup; warned once per model name per process.
    assert _unknown_count() == before + 2
    warnings = [r for r in caplog.records if "no-such-model-r1-test" in r.message]
    assert len(warnings) == 1 and warnings[0].levelno == logging.WARNING
    # The empty-model path stays silent — it is not a pricing gap.
    before = _unknown_count()
    assert estimate_cost_usd(None, 10, 10) == 0.0
    assert estimate_cost_usd("", 10, 10) == 0.0
    assert _unknown_count() == before


def test_price_table_survives_malformed_entries(tmp_path, monkeypatch):
    crafted = {
        "sample_spec": {"input_cost_per_token": 0.0, "output_cost_per_token": 0.0},
        "good-model": {"input_cost_per_token": 1e-06, "output_cost_per_token": 2e-06},
        "no-price-model": {"litellm_provider": "x", "mode": "image_generation"},
        "half-price-model": {"input_cost_per_token": 1e-06},
        "weird-price-model": {"input_cost_per_token": "cheap", "output_cost_per_token": 2e-06},
        "not-a-dict-model": "surprise",
        "prov/good-model": {"input_cost_per_token": 9e-06, "output_cost_per_token": 9e-06},
        "prov/only-prefixed": {"input_cost_per_token": 2e-06, "output_cost_per_token": 4e-06},
    }
    path = tmp_path / "model_prices.json"
    path.write_text(json.dumps(crafted))
    monkeypatch.setattr(pricing, "_PRICE_MAP_PATH", path)
    pricing._price_table.cache_clear()
    try:
        assert estimate_cost_usd("good-model", 1_000_000, 1_000_000) == pytest.approx(3.0)
        # Suffix-indexed entry; bare key still wins for the colliding name above.
        assert estimate_cost_usd("only-prefixed", 1_000_000, 1_000_000) == pytest.approx(6.0)
        # Reserved / unpriced / malformed entries fall through to $0, no crash.
        for model in ("sample_spec", "no-price-model", "half-price-model",
                      "weird-price-model", "not-a-dict-model"):
            assert estimate_cost_usd(model, 1_000, 1_000) == 0.0
    finally:
        pricing._price_table.cache_clear()


def test_refresh_script_idempotent(tmp_path):
    """Refreshing from the same source twice: second run is a byte-for-byte no-op."""
    out = tmp_path / "model_prices.json"
    cmd = [
        sys.executable, str(REFRESH_SCRIPT),
        "--from-file", str(pricing._PRICE_MAP_PATH), "--out", str(out),
    ]
    first = subprocess.run(cmd, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    written = out.read_bytes()
    assert written == pricing._PRICE_MAP_PATH.read_bytes()
    second = subprocess.run(cmd, capture_output=True, text=True)
    assert second.returncode == 0, second.stderr
    assert "no changes" in second.stdout
    assert out.read_bytes() == written


@pytest.mark.skipif(
    importlib.util.find_spec("litellm") is None,
    reason="litellm-plugin extra not installed (default refresh source is its bundled map)",
)
def test_refresh_script_default_source_dry_run():
    result = subprocess.run(
        [sys.executable, str(REFRESH_SCRIPT), "--dry-run"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "installed litellm==" in result.stdout
    assert "dry run" in result.stdout


def test_request_audit_fields_match_columns():
    """`AuditStore.write` passes every RequestAudit field straight to the ORM by name —
    each one must be a real column, or writes silently start failing."""
    fields = {f.name for f in dataclasses.fields(RequestAudit)}
    columns = set(RequestRecord.__table__.columns.keys())
    assert fields <= columns, f"RequestAudit fields with no column: {fields - columns}"


async def test_audit_write_roundtrip(audit_db_url):
    store = AuditStore(audit_db_url)
    await store.init()
    try:
        await store.write(make_audit())
        rows = await store.fetch_requests()
        assert len(rows) == 1
        assert rows[0].tokens_prompt == 11
        assert rows[0].finish_reason == "stop"
        assert rows[0].route_provider == "gemini"
        # An explicit id round-trips and is fetchable on its own.
        rid = uuid.uuid4()
        await store.write(make_audit(id=rid, agent_id="a2"))
        fetched = await store.fetch_request(rid)
        assert fetched is not None and fetched.agent_id == "a2"
        assert await store.fetch_request(uuid.uuid4()) is None
    finally:
        await store.close()


async def test_fetch_requests_filters_and_order(audit_db_url):
    store = AuditStore(audit_db_url)
    await store.init()
    try:
        t0 = utcnow()
        await store.write(make_audit(ts=t0, agent_id="old"))
        await store.write(make_audit(ts=t0 + timedelta(seconds=1), agent_id="new"))
        await store.write(
            make_audit(ts=t0 + timedelta(seconds=2), agent_id="flagged", injection_flagged=True)
        )
        await store.write(
            make_audit(ts=t0 + timedelta(seconds=3), agent_id="tool_def", tool_def_flagged=True)
        )

        newest_first = [r.agent_id for r in await store.fetch_requests()]
        assert newest_first == ["tool_def", "flagged", "new", "old"]
        assert len(await store.fetch_requests(limit=2)) == 2
        assert [r.agent_id for r in await store.fetch_requests(agent_id="new")] == ["new"]
        # flagged_only covers both injection and tool-definition flags.
        assert {r.agent_id for r in await store.fetch_requests(flagged_only=True)} == {
            "flagged", "tool_def",
        }
    finally:
        await store.close()


async def test_summary_and_content_samples(audit_db_url):
    store = AuditStore(audit_db_url)
    await store.init()
    try:
        assert await store.summary() == {
            "requests": 0, "by_provider": {}, "by_status": {}, "injection_flagged": 0,
            "injection_hard": 0, "tool_def_flagged": 0, "redaction_hits": 0, "cost_usd": 0.0,
            "tokens_prompt": 0, "tokens_completion": 0,
        }

        t0 = utcnow()
        rid = uuid.uuid4()
        await store.write(make_audit(id=rid, ts=t0, redaction_hit_count=2, cost_usd=0.25))
        await store.write(make_audit(
            ts=t0 + timedelta(seconds=1), route_provider="local", status=400,
            injection_flagged=True, injection_hard=True, tool_def_flagged=True, cost_usd=0.75,
        ))

        s = await store.summary()
        assert s["requests"] == 2
        assert s["by_provider"] == {"gemini": 1, "local": 1}
        assert s["by_status"] == {"200": 1, "400": 1}
        assert (s["injection_flagged"], s["injection_hard"], s["tool_def_flagged"]) == (1, 1, 1)
        assert s["redaction_hits"] == 2
        assert s["cost_usd"] == pytest.approx(1.0)
        assert s["tokens_prompt"] == 22 and s["tokens_completion"] == 12

        # `since` scopes every aggregate, not just the count.
        recent = await store.summary(since=t0 + timedelta(seconds=1))
        assert recent["requests"] == 1
        assert recent["by_provider"] == {"local": 1}
        assert recent["cost_usd"] == pytest.approx(0.75)

        assert await store.fetch_content_samples(rid) == []
        for reason in ("flagged", "random"):
            await store.write_content_sample(ContentSampleAudit(
                request_id=rid, ts=utcnow(), role="user", redacted_content_enc=b"tok",
                sampled_reason=reason, expires_at=utcnow() + timedelta(days=1),
            ))
        samples = await store.fetch_content_samples(rid)
        assert [s.sampled_reason for s in samples] == ["flagged", "random"]
        assert all(isinstance(s, ContentSample) for s in samples)
        assert await store.fetch_content_samples(uuid.uuid4()) == []
    finally:
        await store.close()
