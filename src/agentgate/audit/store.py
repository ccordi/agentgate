"""Audit write path (async SQLAlchemy over SQLite).

Writes happen **off the request hot path** — the endpoint schedules a fire-and-forget
write after the stream finishes, so audit latency never enters the client's p99.
This is also what keeps SQLite's serialized writes from mattering for the benchmark.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import Integer, delete, func, select
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
    tool_def_flagged: bool = False
    tool_def_hard: bool = False
    tool_def_reasons: list | None = None
    # True when the injection verdict crossed the hard threshold (would-block). Trailing
    # default so existing constructions stay valid; in observe-mode the row is still status 200.
    injection_hard: bool = False
    # Explicit request UUID lets content samples share the same id.
    id: uuid.UUID | None = None
    # Effective injection-guard backend for this request — the resolved backend after
    # per-key override + deberta-availability fallback, i.e. what run_injection_scan
    # actually ran. None for rows written before the scan (e.g. tool_def_blocked rejections).
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
                # RequestAudit field names are the RequestRecord column names — pinned by
                # test_request_audit_fields_match_columns so this shortcut can't drift.
                # Shallow by field, not `asdict`: `asdict` deep-copies every value on the
                # way to a constructor that discards the copy, ~10x the cost per request.
                kwargs = {f.name: getattr(audit, f.name) for f in dataclasses.fields(audit)}
                if kwargs.get("id") is None:
                    kwargs.pop("id")
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

    # ---- read path -------------------------------------------------------------
    # Unlike the write methods above, these do NOT swallow exceptions: they back the
    # `agentgate audit` CLI and tests, where a silent empty result is worse than a
    # traceback.

    def _require_sessionmaker(self) -> async_sessionmaker:
        if self._sessionmaker is None:
            raise RuntimeError("audit store not initialized; call init() first")
        return self._sessionmaker

    async def fetch_requests(
        self,
        *,
        limit: int = 20,
        agent_id: str | None = None,
        flagged_only: bool = False,
    ) -> list[RequestRecord]:
        """Newest-first metadata rows, optionally filtered by agent or flagged status."""
        stmt = select(RequestRecord).order_by(RequestRecord.ts.desc()).limit(limit)
        if agent_id is not None:
            stmt = stmt.where(RequestRecord.agent_id == agent_id)
        if flagged_only:
            stmt = stmt.where(
                RequestRecord.injection_flagged.is_(True) | RequestRecord.tool_def_flagged.is_(True)
            )
        async with self._require_sessionmaker()() as session:
            return list((await session.execute(stmt)).scalars().all())

    async def fetch_request(self, request_id: uuid.UUID) -> RequestRecord | None:
        """One metadata row by exact id, or None."""
        async with self._require_sessionmaker()() as session:
            return await session.scalar(
                select(RequestRecord).where(RequestRecord.id == request_id)
            )

    async def fetch_content_samples(self, request_id: uuid.UUID) -> list[ContentSample]:
        """Content-tier rows belonging to one request, oldest first."""
        stmt = (
            select(ContentSample)
            .where(ContentSample.request_id == request_id)
            .order_by(ContentSample.ts)
        )
        async with self._require_sessionmaker()() as session:
            return list((await session.execute(stmt)).scalars().all())

    async def summary(self, *, since: datetime | None = None) -> dict:
        """Aggregate counts over the metadata tier, optionally from `since` onwards."""
        def scoped(stmt):
            return stmt.where(RequestRecord.ts >= since) if since is not None else stmt

        async with self._require_sessionmaker()() as session:
            totals = (
                await session.execute(scoped(select(
                    func.count(RequestRecord.id),
                    func.sum(func.cast(RequestRecord.injection_flagged, Integer)),
                    func.sum(func.cast(RequestRecord.injection_hard, Integer)),
                    func.sum(func.cast(RequestRecord.tool_def_flagged, Integer)),
                    func.sum(RequestRecord.redaction_hit_count),
                    func.sum(RequestRecord.cost_usd),
                    func.sum(RequestRecord.tokens_prompt),
                    func.sum(RequestRecord.tokens_completion),
                )))
            ).one()
            by_provider = (
                await session.execute(scoped(
                    select(RequestRecord.route_provider, func.count(RequestRecord.id))
                    .group_by(RequestRecord.route_provider)
                ))
            ).all()
            by_status = (
                await session.execute(scoped(
                    select(RequestRecord.status, func.count(RequestRecord.id))
                    .group_by(RequestRecord.status)
                ))
            ).all()

        n, inj_flagged, inj_hard, tool_def_flagged, red_hits, cost, tok_p, tok_c = totals
        return {
            "requests": n,
            "by_provider": {str(p): c for p, c in by_provider},
            "by_status": {str(s): c for s, c in by_status},
            "injection_flagged": inj_flagged or 0,
            "injection_hard": inj_hard or 0,
            "tool_def_flagged": tool_def_flagged or 0,
            "redaction_hits": red_hits or 0,
            "cost_usd": float(cost or 0.0),
            "tokens_prompt": tok_p or 0,
            "tokens_completion": tok_c or 0,
        }


def utcnow() -> datetime:
    return datetime.now(UTC)
