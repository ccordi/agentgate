"""Audit store + pricing tests."""

from __future__ import annotations

import dataclasses
import uuid
from datetime import timedelta

import pytest

from agentgate.audit.models import ContentSample, RequestRecord
from agentgate.audit.store import AuditStore, ContentSampleAudit, RequestAudit, utcnow
from agentgate.pricing import estimate_cost_usd
from tests.support import make_audit


def test_pricing_known_and_unknown():
    # 1M prompt + 1M completion on gemini-2.5-flash-lite = 0.10 + 0.40
    assert estimate_cost_usd("gemini-2.5-flash-lite", 1_000_000, 1_000_000) == pytest.approx(0.50)
    # Prefix match tolerates version suffixes.
    assert estimate_cost_usd("gemini-3-flash-preview", 1_000_000, 0) == pytest.approx(0.30)
    # Unknown / local models are free.
    assert estimate_cost_usd("llama3.2", 1_000_000, 1_000_000) == 0.0
    assert estimate_cost_usd(None, 10, 10) == 0.0


def test_request_audit_fields_match_columns():
    """`AuditStore.write` passes every RequestAudit field straight to the ORM by name —
    each one must be a real column, or writes silently start failing."""
    fields = {f.name for f in dataclasses.fields(RequestAudit)}
    columns = set(RequestRecord.__table__.columns.keys())
    assert fields <= columns, f"RequestAudit fields with no column: {fields - columns}"


async def test_audit_write_roundtrip(tmp_path):
    db = tmp_path / "audit.db"
    store = AuditStore(f"sqlite+aiosqlite:///{db}")
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


async def test_fetch_requests_filters_and_order(tmp_path):
    store = AuditStore(f"sqlite+aiosqlite:///{tmp_path / 'audit.db'}")
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


async def test_summary_and_content_samples(tmp_path):
    store = AuditStore(f"sqlite+aiosqlite:///{tmp_path / 'audit.db'}")
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
