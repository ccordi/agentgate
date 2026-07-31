"""`agentgate audit` — read the audit trail the gateway writes.

Three views over the same store the gateway writes to (`AGENTGATE_DATABASE_URL`):

    agentgate audit tail [-n 20] [--agent ID] [--flagged] [--json]
    agentgate audit stats [--since 24h] [--json]
    agentgate audit show <request-id-prefix> [--json]

`show` is the one that demonstrates the content tier: metadata rows are always readable,
but the sampled message text is Fernet-encrypted at rest, so it only decrypts when
AGENTGATE_CONTENT_ENC_KEY is set. Open → query → close per invocation; no daemon.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import uuid
from datetime import UTC, datetime, timedelta

from agentgate.audit.crypto import ContentCipher
from agentgate.audit.store import AuditStore
from agentgate.config import get_settings

_SINCE_RE = re.compile(r"^(\d+)([hd])$")


def parse_since(spec: str | None) -> datetime | None:
    """`24h` / `7d` → an aware UTC cutoff. None means all time."""
    if spec is None:
        return None
    m = _SINCE_RE.match(spec.strip())
    if not m:
        raise SystemExit(f"--since: expected <N>h or <N>d, got {spec!r}")
    n, unit = int(m.group(1)), m.group(2)
    delta = timedelta(hours=n) if unit == "h" else timedelta(days=n)
    return datetime.now(UTC) - delta


def _as_utc(ts: datetime) -> datetime:
    """Re-attach UTC to a timestamp SQLite handed back naive.

    Rows are written with aware-UTC values, but SQLite has no timezone type and
    SQLAlchemy round-trips `DateTime(timezone=True)` as naive. A naive value is not
    "unknown zone" to the stdlib — `astimezone()` assumes it is *local*, so converting
    it directly relabels UTC as local time instead of converting it.
    """
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=UTC)


def _json_default(obj):
    if isinstance(obj, datetime):
        return _as_utc(obj).isoformat()
    if isinstance(obj, uuid.UUID):
        return str(obj)
    return str(obj)


def _dump_json(payload) -> None:
    print(json.dumps(payload, indent=2, default=_json_default))


def _row_dict(row) -> dict:
    return {c.name: getattr(row, c.name) for c in row.__table__.columns}


def _db_path(url: str) -> str:
    return url.split("///", 1)[1] if "///" in url else url


async def _open_store(settings) -> AuditStore:
    url = settings.database_url
    path = _db_path(url)
    if url.startswith("sqlite") and path not in (":memory:", "") and not _exists(path):
        raise SystemExit(f"no audit database at {path} — has the gateway run?")
    store = AuditStore(url)
    await store.init()
    return store


def _exists(path: str) -> bool:
    from pathlib import Path
    return Path(path).exists()


# --- commands -------------------------------------------------------------------------

async def cmd_tail(args, store: AuditStore) -> None:
    rows = await store.fetch_requests(
        limit=args.n, agent_id=args.agent, flagged_only=args.flagged
    )
    if args.json:
        _dump_json([_row_dict(r) for r in rows])
        return
    if not rows:
        print("no matching requests")
        return
    fmt = "{:<8} {:<8} {:<10} {:<8} {:<14} {:<6} {:<7} {:<6} {:<4} {:>8}"
    print(fmt.format("TIME", "ID", "AGENT", "PROVIDER", "MODEL", "STATUS",
                     "SENS", "INJ", "RED", "COST"))
    for r in rows:
        inj = "-"
        if r.injection_score is not None:
            inj = f"{r.injection_score:.2f}" + ("!" if r.injection_hard else "")
        print(fmt.format(
            _as_utc(r.ts).astimezone().strftime("%H:%M:%S") if r.ts else "-",
            str(r.id)[:8],
            (r.agent_id or "-")[:10],
            (r.route_provider or "-")[:8],
            (r.upstream_model or r.model_requested or "-")[:14],
            str(r.status if r.status is not None else "-"),
            (r.sensitivity_class or "-")[:7],
            inj,
            str(r.redaction_hit_count),
            f"${r.cost_usd:.5f}",
        ))


async def cmd_stats(args, store: AuditStore) -> None:
    summary = await store.summary(since=parse_since(args.since))
    if args.json:
        _dump_json(summary)
        return
    print(f"since: {args.since or 'all time'}")
    for key, value in summary.items():
        if isinstance(value, dict):
            rendered = ", ".join(f"{k}={v}" for k, v in sorted(value.items())) or "-"
        elif key == "cost_usd":
            rendered = f"${value:.5f}"
        else:
            rendered = str(value)
        print(f"  {key:<20} {rendered}")


async def cmd_show(args, store: AuditStore) -> None:
    prefix = args.request_id.lower()
    # No LIKE on a UUID column across dialects — scan the recent window instead.
    candidates = [
        r for r in await store.fetch_requests(limit=1000) if str(r.id).startswith(prefix)
    ]
    if not candidates:
        raise SystemExit(f"no request id starting with {prefix!r} in the last 1000 rows")
    if len(candidates) > 1:
        ids = ", ".join(str(r.id)[:12] for r in candidates[:5])
        raise SystemExit(f"ambiguous prefix {prefix!r} — matches {len(candidates)}: {ids}…")

    row = candidates[0]
    samples = await store.fetch_content_samples(row.id)
    cipher = ContentCipher(get_settings().content_enc_key)

    def decrypt(blob: bytes) -> str:
        if not cipher.enabled:
            return "<encrypted — set AGENTGATE_CONTENT_ENC_KEY to view>"
        try:
            return cipher.decrypt(blob)
        except Exception as exc:  # noqa: BLE001 — a wrong key is a normal outcome here
            return f"<undecryptable: {type(exc).__name__}>"

    if args.json:
        _dump_json({
            "request": _row_dict(row),
            "content_samples": [
                {"role": s.role, "sampled_reason": s.sampled_reason, "ts": s.ts,
                 "expires_at": s.expires_at, "content": decrypt(s.redacted_content_enc)}
                for s in samples
            ],
        })
        return

    for key, value in _row_dict(row).items():
        print(f"{key:<22} {value}")
    print(f"\ncontent samples: {len(samples)}")
    for s in samples:
        print(f"\n  [{s.sampled_reason}] role={s.role} expires={s.expires_at}")
        print(f"  {decrypt(s.redacted_content_enc)}")


_COMMANDS = {"tail": cmd_tail, "stats": cmd_stats, "show": cmd_show}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentgate audit", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="audit_command", required=True)

    tail = sub.add_parser("tail", help="newest requests, one per line")
    tail.add_argument("-n", type=int, default=20, help="how many rows (default 20)")
    tail.add_argument("--agent", default=None, help="only this agent_id")
    tail.add_argument("--flagged", action="store_true",
                      help="only rows the injection guard or tool inspector flagged")

    stats = sub.add_parser("stats", help="aggregate counts")
    stats.add_argument("--since", default=None, metavar="Nh|Nd",
                       help="window, e.g. 24h or 7d (default: all time)")

    show = sub.add_parser("show", help="one request in full, with its content samples")
    show.add_argument("request_id", help="request id, or a unique prefix of one")

    for p in (tail, stats, show):
        p.add_argument("--json", action="store_true", help="machine-readable output")
    return parser


async def _run(args) -> None:
    store = await _open_store(get_settings())
    try:
        await _COMMANDS[args.audit_command](args, store)
    finally:
        await store.close()


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    asyncio.run(_run(args))
