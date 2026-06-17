"""Audit store schema (SQLAlchemy).

Two tiers:
- **Metadata tier** — one row per request, always written, retained indefinitely.
- **Content tier** — sampled/flagged raw text, redacted-at-rest, with TTL.

Types are deliberately portable — JSON instead of JSONB, generic Uuid, no ARRAY —
so swapping SQLite for Postgres is a connection-URL change plus a migration regen.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, LargeBinary, String, Uuid
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(UTC)


class RequestRecord(Base):
    """Metadata tier — always written, retained. Drives telemetry + cost analytics.

    Mirrors the `requests` audit table. All latency fields are populated;
    stages that did not run record None.
    """

    __tablename__ = "requests"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)

    agent_id: Mapped[str | None] = mapped_column(String, nullable=True)
    key_id: Mapped[str | None] = mapped_column(String, nullable=True)

    model_requested: Mapped[str | None] = mapped_column(String, nullable=True)
    route_provider: Mapped[str | None] = mapped_column(String, nullable=True)
    route_is_local: Mapped[bool] = mapped_column(Boolean, default=False)
    upstream_model: Mapped[str | None] = mapped_column(String, nullable=True)

    sensitivity_class: Mapped[str | None] = mapped_column(String, nullable=True)

    tokens_prompt: Mapped[int] = mapped_column(Integer, default=0)
    tokens_completion: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)

    latency_total_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_inject_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_classify_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_redact_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_upstream_ms: Mapped[float | None] = mapped_column(Float, nullable=True)

    injection_flagged: Mapped[bool] = mapped_column(Boolean, default=False)
    # True when the verdict crossed the hard threshold (would-block). In normal mode this
    # always coincides with a 400; in guard_observe_mode the request is forwarded anyway, so
    # this column is how would-have-blocked events (live FP candidates) stay countable.
    injection_hard: Mapped[bool] = mapped_column(Boolean, default=False)
    injection_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    redaction_hit_count: Mapped[int] = mapped_column(Integer, default=0)
    # JSON (not Postgres ARRAY) for portability — list of hit type strings.
    redaction_hit_types: Mapped[list | None] = mapped_column(JSON, nullable=True)

    tool_call_count: Mapped[int] = mapped_column(Integer, default=0)
    finish_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[int | None] = mapped_column(Integer, nullable=True)

    skill_flagged: Mapped[bool] = mapped_column(Boolean, default=False)
    skill_hard: Mapped[bool] = mapped_column(Boolean, default=False)
    skill_reasons: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # Effective injection-guard backend that ran for this request — "deberta" | "llm" |
    # "combined" | "heuristic", or null for rows written before the scan ran
    # (e.g. skill_blocked rejections).
    guard_backend: Mapped[str | None] = mapped_column(String, nullable=True)


class ContentSample(Base):
    """Content tier — redacted+encrypted message text, sampled from non-sensitive cloud requests.

    Never written for local routes or sensitive content (those stay off-box by design).
    Rows expire after ``content_retention_days``; the sweeper deletes them.
    """

    __tablename__ = "content_samples"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    # Logical FK to requests.id — not a DB-level FK so SQLite stays simple.
    request_id: Mapped[uuid.UUID] = mapped_column(Uuid, index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    role: Mapped[str] = mapped_column(String)                   # "user" | "tool" | "assistant"
    redacted_content_enc: Mapped[bytes] = mapped_column(LargeBinary)  # Fernet token
    sampled_reason: Mapped[str] = mapped_column(String)         # "flagged" | "random"
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
