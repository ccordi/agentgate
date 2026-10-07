"""Audit store schema (SQLAlchemy).

The **metadata tier**: one best-effort row attempted per handled request,
retained indefinitely. The content tier (sampled/flagged raw text, redact-at-rest,
TTL) is `ContentSample` below.

Types are deliberately portable — JSON instead of JSONB, generic Uuid, no ARRAY —
so the same schema runs on SQLite and on Postgres (the `pg` extra and a
`postgresql+asyncpg://` URL).
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


class IssuedKey(Base):
    """Gateway-minted API keys — what makes `key_id` a verified identity.

    Only the sha256 of the secret is stored; the plaintext leaves mint exactly once.
    `key_id` is precomputed at mint from the canonical presentation form
    (`Bearer <secret>`), so every verified request collapses to one downstream
    identity regardless of how the credential was presented — a raw-header hash
    would let a valid key holder vary header casing to obtain fresh per-key
    counters. Lives in the gateway DB beside the audit tables, but it is
    configuration state, not audit data.
    """

    __tablename__ = "issued_keys"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    label: Mapped[str | None] = mapped_column(String, nullable=True)
    key_hash: Mapped[str] = mapped_column(String, unique=True, index=True)
    key_id: Mapped[str] = mapped_column(String, index=True)


class RequestRecord(Base):
    """One metadata-tier row per handled request (see the module docstring). Drives
    telemetry and cost analytics."""

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
    # Not written by the current pipeline; always null.
    latency_classify_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_redact_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_upstream_ms: Mapped[float | None] = mapped_column(Float, nullable=True)

    injection_flagged: Mapped[bool] = mapped_column(Boolean, default=False)
    # True when the verdict crossed the hard threshold (would-block). In normal mode this
    # always coincides with a 400; in guard_observe_mode the request is forwarded anyway, so
    # this column is how would-have-blocked events (possible false positives on live
    # traffic) stay countable.
    injection_hard: Mapped[bool] = mapped_column(Boolean, default=False)
    injection_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    redaction_hit_count: Mapped[int] = mapped_column(Integer, default=0)
    # JSON (not Postgres ARRAY) for portability — a list of `{"type", "count"}` objects,
    # one per hit type.
    redaction_hit_types: Mapped[list | None] = mapped_column(JSON, nullable=True)

    tool_call_count: Mapped[int] = mapped_column(Integer, default=0)
    finish_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Tool-definition screening verdicts.
    tool_def_flagged: Mapped[bool] = mapped_column(Boolean, default=False)
    tool_def_hard: Mapped[bool] = mapped_column(Boolean, default=False)
    tool_def_reasons: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # Effective injection-guard backend that ran for this request —
    # "deberta" | "llm" | "combined" | "heuristic", or null for rows
    # written before the scan (e.g. tool_def_blocked rejections).
    guard_backend: Mapped[str | None] = mapped_column(String, nullable=True)

    # How many untrusted items the guard examined. Distinguishes
    # "scanned nothing" from "scanned and found nothing" — both otherwise read
    # injection_score=0.0, injection_flagged=False. 0 on a
    # forwarded request is the signal worth alerting on: it means the scan surface was
    # empty, which on guard_backend="llm" happens whenever the request carries no tool
    # results the assistant has not already answered. Null (not 0) for rows written before
    # any scan ran, e.g. tool_def_blocked rejections — the same convention guard_backend
    # uses above.
    scanned_item_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Caveats on what the rest of the row means — a list of closed-vocabulary tags,
    # `<kind>:<detail>[:<n>]`: `truncated:classify:<n>` (the sensitivity classifier
    # exhausted its window at n chars), `truncated:egress_payload:<n>` (the egress PDP
    # did), `rejected:<wire error type>` (why a pre-forward rejection happened, the same
    # string the client was answered with; the set is `pipeline.REJECTION_TYPES`) and
    # `gap_abandoned:<pattern>:<distance>` (a bounded-gap pattern's prefix was seen with
    # its suffix past the bound — recorded, never flagged). Every detail comes from a
    # fixed set and every n is an integer: no value carries free or attacker-controlled
    # text. Null means nothing to report OR the row predates the column; an empty list is
    # never written, so `WHERE caveats IS NOT NULL` finds every affected row.
    caveats: Mapped[list | None] = mapped_column(JSON, nullable=True)


class ContentSample(Base):
    """Content tier — redacted+encrypted message text, sampled from non-sensitive cloud requests.

    Never written for local routes or sensitive content: that content is never stored.
    Rows expire after ``content_retention_days``; the sweeper deletes them.
    """

    __tablename__ = "content_samples"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    # Logical FK to requests.id — not a DB-level FK so SQLite stays simple.
    request_id: Mapped[uuid.UUID] = mapped_column(Uuid, index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    # "user" | "tool" | "function" | "assistant". `function` is the legacy tool-output
    # role: the tier samples the guards' scan surface (`content.trailing_tool_outputs`),
    # which covers both. Unconstrained String — a new value needs no migration.
    role: Mapped[str] = mapped_column(String)
    redacted_content_enc: Mapped[bytes] = mapped_column(LargeBinary)  # Fernet token
    sampled_reason: Mapped[str] = mapped_column(String)         # "flagged" | "random"
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
