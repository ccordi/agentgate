"""Audit-row helpers: a builder for the write path, polling for the read path."""

from __future__ import annotations

import asyncio

from agentgate.audit.models import RequestRecord
from agentgate.audit.store import AuditStore, RequestAudit, utcnow


def make_audit(**overrides) -> RequestAudit:
    """A complete, unremarkable `RequestAudit`; override what a test cares about.

    One definition because `RequestAudit` keeps growing columns (`injection_hard`,
    `guard_backend`): a per-file copy means every new field is a multi-file edit,
    and copies that start identical drift apart silently.
    """
    base = dict(
        ts=utcnow(), agent_id="a1", key_id=None, model_requested="gemini-3-flash",
        route_provider="gemini", route_is_local=False, upstream_model="gemini-3-flash",
        sensitivity_class=None, tokens_prompt=11, tokens_completion=6, cost_usd=0.0001,
        latency_total_ms=12.3, latency_upstream_ms=10.0, injection_flagged=False,
        injection_score=None, redaction_hit_count=0, redaction_hit_types=None,
        tool_call_count=0, finish_reason="stop", status=200,
    )
    return RequestAudit(**{**base, **overrides})


async def wait_for_audit_row(
    store: AuditStore, *, tries: int = 40, delay: float = 0.02
) -> RequestRecord | None:
    """Poll briefly for the newest audit row to land, or return None.

    Audit writes are scheduled off the hot path after the stream finishes, so a test
    that asserts on the row has to wait for it. Uses the public read API — no reaching
    into `store._sessionmaker`.
    """
    for _ in range(tries):
        rows = await store.fetch_requests(limit=1)
        if rows:
            return rows[0]
        await asyncio.sleep(delay)
    return None
