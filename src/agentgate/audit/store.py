"""Audit write path (async SQLAlchemy over SQLite).

Writes happen **off the request hot path** — the endpoint schedules a fire-and-forget
write after the stream finishes, so audit latency never enters the client's p99.
This is also what keeps SQLite's serialized writes from mattering for the benchmark.
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from agentgate.audit.models import Base, ContentSample, RequestRecord

log = logging.getLogger("agentgate.audit")


@dataclass
class RequestAudit:
    """A completed request's metadata, assembled by the endpoint then persisted."""

    ts: datetime
    agent_id: str | None
    key_id: str | None
    model_requested: str | None
    route_provider: str | None
    route_is_local: bool
    upstream_model: str | None
    sensitivity_class: str | None
    tokens_prompt: int
    tokens_completion: int
    cost_usd: float
    latency_total_ms: float | None
    latency_upstream_ms: float | None
    injection_flagged: bool
    injection_score: float | None
    redaction_hit_count: int
    redaction_hit_types: list | None
    tool_call_count: int
    finish_reason: str | None
    status: int | None
    # Optional/trailing so existing constructions stay valid; populated by the endpoint.
    latency_inject_ms: float | None = None
    latency_redact_ms: float | None = None
    skill_flagged: bool = False
    skill_hard: bool = False
    skill_reasons: list | None = None
    # True when the injection verdict crossed the hard threshold (would-block). Trailing
    # default so existing constructions stay valid; in observe-mode the row is still status 200.
    injection_hard: bool = False
    # Explicit request UUID lets content samples share the same id.
    id: uuid.UUID | None = None
    # Effective injection-guard backend for this request — the resolved backend after
    # per-key override + deberta-availability fallback, i.e. what _scan_request actually
    # ran. None for rows written before the scan (e.g. skill_blocked rejections).
    guard_backend: str | None = None


@dataclass
class ContentSampleAudit:
    """One message's redacted+encrypted content to persist as a content sample."""

    request_id: uuid.UUID
    ts: datetime
    role: str
    redacted_content_enc: bytes
    sampled_reason: str
    expires_at: datetime


class AuditStore:
    def __init__(self, database_url: str) -> None:
        self._url = database_url
        self._engine: AsyncEngine | None = None
        self._sessionmaker: async_sessionmaker | None = None

    async def init(self) -> None:
        # Ensure the sqlite directory exists for file-based URLs.
        if self._url.startswith("sqlite") and "///" in self._url:
            path = self._url.split("///", 1)[1]
            if path and path not in (":memory:",):
                os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self._engine = create_async_engine(self._url, future=True)
        self._sessionmaker = async_sessionmaker(self._engine, expire_on_commit=False)
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        log.info("audit store ready: %s", self._url)

    async def close(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()

    async def write(self, audit: RequestAudit) -> None:
        """Persist one metadata row. Swallows errors — auditing must never break a
        proxied request."""
        if self._sessionmaker is None:
            log.warning("audit store not initialized; dropping record")
            return
        try:
            async with self._sessionmaker() as session:
                kwargs: dict = dict(
                    ts=audit.ts,
                    agent_id=audit.agent_id,
                    key_id=audit.key_id,
                    model_requested=audit.model_requested,
                    route_provider=audit.route_provider,
                    route_is_local=audit.route_is_local,
                    upstream_model=audit.upstream_model,
                    sensitivity_class=audit.sensitivity_class,
                    tokens_prompt=audit.tokens_prompt,
                    tokens_completion=audit.tokens_completion,
                    cost_usd=audit.cost_usd,
                    latency_total_ms=audit.latency_total_ms,
                    latency_upstream_ms=audit.latency_upstream_ms,
                    latency_inject_ms=audit.latency_inject_ms,
                    latency_redact_ms=audit.latency_redact_ms,
                    injection_flagged=audit.injection_flagged,
                    injection_hard=audit.injection_hard,
                    injection_score=audit.injection_score,
                    redaction_hit_count=audit.redaction_hit_count,
                    redaction_hit_types=audit.redaction_hit_types,
                    tool_call_count=audit.tool_call_count,
                    finish_reason=audit.finish_reason,
                    status=audit.status,
                    skill_flagged=audit.skill_flagged,
                    skill_hard=audit.skill_hard,
                    skill_reasons=audit.skill_reasons,
                    guard_backend=audit.guard_backend,
                )
                if audit.id is not None:
                    kwargs["id"] = audit.id
                session.add(RequestRecord(**kwargs))
                await session.commit()
        except Exception:  # noqa: BLE001 — auditing must never break forwarding
            log.exception("audit write failed")

    async def write_content_sample(self, sample: ContentSampleAudit) -> None:
        """Persist one content sample row. Swallows errors — must never break forwarding."""
        if self._sessionmaker is None:
            return
        try:
            async with self._sessionmaker() as session:
                session.add(ContentSample(
                    request_id=sample.request_id,
                    ts=sample.ts,
                    role=sample.role,
                    redacted_content_enc=sample.redacted_content_enc,
                    sampled_reason=sample.sampled_reason,
                    expires_at=sample.expires_at,
                ))
                await session.commit()
        except Exception:  # noqa: BLE001
            log.exception("content sample write failed")

    async def purge_expired_content(self) -> int:
        """Delete content_samples rows past their TTL. Returns number of rows deleted."""
        if self._sessionmaker is None:
            return 0
        try:
            now = datetime.now(UTC)
            async with self._sessionmaker() as session:
                result = await session.execute(
                    delete(ContentSample).where(ContentSample.expires_at < now)
                )
                await session.commit()
                return result.rowcount
        except Exception:  # noqa: BLE001
            log.exception("content purge failed")
            return 0

    async def count_content_samples(self) -> int:
        """Return total content_samples rows (for tests)."""
        if self._sessionmaker is None:
            return 0
        async with self._sessionmaker() as session:
            result = await session.execute(select(ContentSample))
            return len(result.scalars().all())


def utcnow() -> datetime:
    return datetime.now(UTC)
