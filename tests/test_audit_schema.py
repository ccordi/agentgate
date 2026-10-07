"""Audit schema migration safety.

Reproduces the failure mode: an audit DB created before a model column existed
(``injection_hard``, ``guard_backend``) silently drops every insert, because
``create_all`` no-ops on existing tables and ``write()`` swallows errors by design.
``init()`` must repair such a DB additively.

DDL and introspection go through SQLAlchemy (not raw sqlite3) so the same test runs
against both backends via ``audit_db_url``.
"""

from __future__ import annotations

from sqlalchemy import func, inspect, select, text
from sqlalchemy.ext.asyncio import create_async_engine

from agentgate.audit.models import RequestRecord
from agentgate.audit.store import AuditStore
from tests.support import make_audit


async def _run_ddl(url: str, *statements: str) -> None:
    engine = create_async_engine(url)
    try:
        async with engine.begin() as conn:
            for stmt in statements:
                await conn.execute(text(stmt))
    finally:
        await engine.dispose()


async def _column_names(url: str, table: str) -> set[str]:
    engine = create_async_engine(url)
    try:
        async with engine.connect() as conn:
            return await conn.run_sync(
                lambda c: {col["name"] for col in inspect(c).get_columns(table)}
            )
    finally:
        await engine.dispose()


async def test_init_adds_missing_columns_to_existing_db(audit_db_url):
    # Build a DB with the current schema, then drop the four additive columns to recreate
    # an audit DB from before they existed.
    store = AuditStore(audit_db_url)
    await store.init()
    await store.close()
    await _run_ddl(
        audit_db_url,
        "ALTER TABLE requests DROP COLUMN guard_backend",
        "ALTER TABLE requests DROP COLUMN injection_hard",  # NOT NULL DEFAULT case
        "ALTER TABLE requests DROP COLUMN scanned_item_count",
        # The newest column, and therefore the one an existing DB file is most likely
        # to be missing — the exact shape of the failure this test exists for.
        "ALTER TABLE requests DROP COLUMN caveats",  # nullable JSON: the plain ADD branch
    )

    # Re-init against the old DB: all four columns must come back, and an insert that
    # references them must persist instead of being silently dropped.
    store = AuditStore(audit_db_url)
    await store.init()
    try:
        cols = await _column_names(audit_db_url, "requests")
        assert {"guard_backend", "injection_hard", "scanned_item_count", "caveats"} <= cols

        await store.write(make_audit(
            guard_backend="llm", injection_hard=True, scanned_item_count=0,
            caveats=["rejected:guard_unavailable"],
        ))
        async with store._sessionmaker() as session:  # type: ignore[attr-defined]
            count = await session.scalar(select(func.count()).select_from(RequestRecord))
            row = await session.scalar(select(RequestRecord))
        assert count == 1, "insert was silently dropped"
        assert row.guard_backend == "llm"
        assert row.injection_hard is True
        # 0 must round-trip as 0, not as NULL: "scanned nothing" is the value this column
        # was added to make countable, so falsiness must not erase it.
        assert row.scanned_item_count == 0
        assert row.caveats == ["rejected:guard_unavailable"]
    finally:
        await store.close()


async def test_rows_written_before_the_caveats_column_read_none(audit_db_url):
    """A pre-column DB is repaired additively and its old rows read `caveats=None` —
    indistinguishable from "nothing to report" by design, the `guard_backend`
    convention. A new row's list round-trips through the JSON column, and None stays
    None (the write path never turns it into `[]`).
    """
    store = AuditStore(audit_db_url)
    await store.init()
    await store.write(make_audit(agent_id="old"))
    await store.close()
    await _run_ddl(audit_db_url, "ALTER TABLE requests DROP COLUMN caveats")
    assert "caveats" not in await _column_names(audit_db_url, "requests")

    store = AuditStore(audit_db_url)
    await store.init()
    try:
        assert "caveats" in await _column_names(audit_db_url, "requests")
        tags = ["truncated:classify:20000", "rejected:spend_exceeded"]
        await store.write(make_audit(agent_id="new", caveats=tags))
        await store.write(make_audit(agent_id="plain", caveats=None))
        rows = {r.agent_id: r for r in await store.fetch_requests(limit=10)}
        assert set(rows) == {"old", "new", "plain"}, "an insert was silently dropped"
        assert rows["old"].caveats is None
        assert rows["new"].caveats == tags
        assert rows["plain"].caveats is None
    finally:
        await store.close()


async def test_init_noop_on_current_schema(audit_db_url):
    # A DB already at the current schema must pass through init() unchanged and writable.
    store = AuditStore(audit_db_url)
    await store.init()
    await store.close()

    store = AuditStore(audit_db_url)
    await store.init()
    try:
        await store.write(make_audit())
        async with store._sessionmaker() as session:  # type: ignore[attr-defined]
            count = await session.scalar(select(func.count()).select_from(RequestRecord))
        assert count == 1
    finally:
        await store.close()
