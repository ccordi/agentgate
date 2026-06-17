"""Audit store + pricing tests."""

from __future__ import annotations

import pytest
from sqlalchemy import func, select

from agentgate.audit.models import RequestRecord
from agentgate.audit.store import AuditStore, RequestAudit, utcnow
from agentgate.pricing import estimate_cost_usd


def test_pricing_known_and_unknown():
    # 1M prompt + 1M completion on gemini-2.5-flash-lite = 0.10 + 0.40
    assert estimate_cost_usd("gemini-2.5-flash-lite", 1_000_000, 1_000_000) == pytest.approx(0.50)
    # Prefix match tolerates version suffixes.
    assert estimate_cost_usd("gemini-3-flash-preview", 1_000_000, 0) == pytest.approx(0.30)
    # Unknown / local models are free.
    assert estimate_cost_usd("llama3.2", 1_000_000, 1_000_000) == 0.0
    assert estimate_cost_usd(None, 10, 10) == 0.0


async def test_audit_write_roundtrip(tmp_path):
    db = tmp_path / "audit.db"
    store = AuditStore(f"sqlite+aiosqlite:///{db}")
    await store.init()
    try:
        await store.write(
            RequestAudit(
                ts=utcnow(), agent_id="a1", key_id=None, model_requested="gemini-3-flash",
                route_provider="gemini", route_is_local=False, upstream_model="gemini-3-flash",
                sensitivity_class=None, tokens_prompt=11, tokens_completion=6, cost_usd=0.0001,
                latency_total_ms=12.3, latency_upstream_ms=10.0, injection_flagged=False,
                injection_score=None, redaction_hit_count=0, redaction_hit_types=None,
                tool_call_count=0, finish_reason="stop", status=200,
            )
        )
        # Read it back through a fresh session.
        async with store._sessionmaker() as session:  # type: ignore[attr-defined]
            count = await session.scalar(select(func.count()).select_from(RequestRecord))
            row = await session.scalar(select(RequestRecord))
        assert count == 1
        assert row.tokens_prompt == 11
        assert row.finish_reason == "stop"
        assert row.route_provider == "gemini"
    finally:
        await store.close()
