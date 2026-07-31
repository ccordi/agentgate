"""Tests for `agentgate audit`.

The command functions are called in-process (no subprocess) against a temp store, so a
failure points at a line rather than at an exit code.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from cryptography.fernet import Fernet

from agentgate.audit import cli
from agentgate.audit.crypto import ContentCipher
from agentgate.audit.store import AuditStore, ContentSampleAudit, RequestAudit, utcnow
from tests.support import FAKE_EMAIL, make_audit


def _audit(**overrides) -> RequestAudit:
    """The shared row, populated the way the CLI tests need it — they filter and group on
    agent_id/key_id/sensitivity_class, which the shared defaults leave empty."""
    return make_audit(**{"agent_id": "continue", "key_id": "k1",
                         "sensitivity_class": "none", **overrides})


@pytest.fixture
async def store(tmp_path, monkeypatch):
    """A seeded temp store, with get_settings() pointed at it."""
    from agentgate import config

    db = tmp_path / "audit.db"
    url = f"sqlite+aiosqlite:///{db}"
    s = AuditStore(url)
    await s.init()

    settings = config.Settings(_env_file=None, database_url=url, content_enc_key=None)
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    try:
        yield s
    finally:
        await s.close()


def _args(**kw):
    return type("Args", (), {"json": False, "n": 20, "agent": None, "flagged": False,
                             "since": None, **kw})()


async def test_tail_renders_seeded_rows(store, capsys):
    await store.write(_audit())
    await store.write(_audit(
        agent_id="capture", injection_flagged=True, injection_hard=True,
        injection_score=0.9, redaction_hit_count=2, status=400,
    ))

    await cli.cmd_tail(_args(), store)
    out = capsys.readouterr().out
    assert "TIME" in out and "PROVIDER" in out
    assert "continue" in out and "capture" in out
    assert "0.90!" in out            # hard verdicts carry the ! suffix
    assert "$0.00010" in out


async def test_tail_filters(store, capsys):
    await store.write(_audit(agent_id="a"))
    await store.write(_audit(agent_id="b", injection_flagged=True))

    await cli.cmd_tail(_args(agent="a"), store)
    out = capsys.readouterr().out
    assert "a" in out and " b " not in out

    await cli.cmd_tail(_args(flagged=True), store)
    assert "b" in capsys.readouterr().out


async def test_tail_json_is_machine_readable(store, capsys):
    rid = uuid.uuid4()
    await store.write(_audit(id=rid))
    await cli.cmd_tail(_args(json=True), store)
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 1
    assert rows[0]["id"] == str(rid)          # UUIDs stringify
    assert rows[0]["ts"].startswith("20")     # timestamps are ISO


async def test_timestamps_are_converted_from_utc_not_relabelled(store, capsys):
    """SQLite hands `ts` back naive, and `astimezone()` treats naive as *local*.

    Calling it directly relabels a UTC clock reading as local time rather than
    converting it, so every row in `tail` was off by the operator's UTC offset —
    invisibly, since only %H:%M:%S is shown, so the date rollover doesn't show either.
    """
    ts = datetime(2026, 7, 31, 3, 4, 5, tzinfo=UTC)
    await store.write(_audit(ts=ts))

    # JSON keeps the offset rather than emitting a zone-less string.
    await cli.cmd_tail(_args(json=True), store)
    assert json.loads(capsys.readouterr().out)[0]["ts"] == "2026-07-31T03:04:05+00:00"

    # The human view is that instant in local time — whatever the local zone is.
    await cli.cmd_tail(_args(), store)
    assert ts.astimezone().strftime("%H:%M:%S") in capsys.readouterr().out


async def test_tail_says_so_when_empty(store, capsys):
    await cli.cmd_tail(_args(), store)
    assert "no matching requests" in capsys.readouterr().out


async def test_stats_renders_and_scopes(store, capsys):
    t0 = utcnow()
    await store.write(_audit(ts=t0 - timedelta(days=3), cost_usd=1.0))
    await store.write(_audit(ts=t0, cost_usd=0.5, injection_flagged=True))

    await cli.cmd_stats(_args(), store)
    out = capsys.readouterr().out
    assert "requests" in out and "gemini=2" in out
    assert "$1.50000" in out

    await cli.cmd_stats(_args(since="24h"), store)
    out = capsys.readouterr().out
    assert "$0.50000" in out


def test_parse_since():
    assert cli.parse_since(None) is None
    assert cli.parse_since("24h") is not None
    assert cli.parse_since("7d") is not None
    with pytest.raises(SystemExit):
        cli.parse_since("yesterday")


async def test_show_decrypts_content_when_key_is_set(store, capsys, monkeypatch):
    from agentgate import config

    key = Fernet.generate_key().decode()
    cipher = ContentCipher(key)
    rid = uuid.uuid4()
    await store.write(_audit(id=rid))
    await store.write_content_sample(ContentSampleAudit(
        request_id=rid, ts=utcnow(), role="user",
        redacted_content_enc=cipher.encrypt(f"reach me at {FAKE_EMAIL}"),
        sampled_reason="flagged", expires_at=utcnow() + timedelta(days=30),
    ))

    settings = config.Settings(
        _env_file=None, database_url=store._url, content_enc_key=key
    )
    monkeypatch.setattr(cli, "get_settings", lambda: settings)

    await cli.cmd_show(_args(request_id=str(rid)[:8]), store)
    out = capsys.readouterr().out
    assert str(rid) in out
    assert "content samples: 1" in out
    assert f"reach me at {FAKE_EMAIL}" in out


async def test_show_without_key_says_encrypted(store, capsys):
    rid = uuid.uuid4()
    cipher = ContentCipher(Fernet.generate_key().decode())
    await store.write(_audit(id=rid))
    await store.write_content_sample(ContentSampleAudit(
        request_id=rid, ts=utcnow(), role="user",
        redacted_content_enc=cipher.encrypt("secret text"),
        sampled_reason="random", expires_at=utcnow() + timedelta(days=30),
    ))

    # The `store` fixture pins content_enc_key=None.
    await cli.cmd_show(_args(request_id=str(rid)[:8]), store)
    out = capsys.readouterr().out
    assert "secret text" not in out
    assert "set AGENTGATE_CONTENT_ENC_KEY to view" in out


async def test_show_rejects_unknown_and_ambiguous_prefixes(store):
    with pytest.raises(SystemExit, match="no request id starting with"):
        await cli.cmd_show(_args(request_id="ffffffff"), store)

    # Two ids sharing a prefix: '' matches everything.
    await store.write(_audit())
    await store.write(_audit())
    with pytest.raises(SystemExit, match="ambiguous prefix"):
        await cli.cmd_show(_args(request_id=""), store)


def test_parser_accepts_the_documented_forms():
    p = cli.build_parser()
    assert p.parse_args(["tail", "-n", "5", "--flagged"]).n == 5
    assert p.parse_args(["stats", "--since", "7d"]).since == "7d"
    assert p.parse_args(["show", "abc123", "--json"]).json is True
    with pytest.raises(SystemExit):
        p.parse_args([])                      # a subcommand is required


async def test_missing_database_is_a_clean_error(tmp_path):
    from agentgate import config

    settings = config.Settings(
        _env_file=None, database_url=f"sqlite+aiosqlite:///{tmp_path / 'nope.db'}"
    )
    with pytest.raises(SystemExit, match="has the gateway run"):
        await cli._open_store(settings)


# --- entry-point dispatch --------------------------------------------------------------
# `agentgate` with no args must still start the server, and `agentgate audit …` must reach
# the audit CLI with the rest of the argv intact.

def test_bare_invocation_serves(monkeypatch):
    from agentgate import __main__ as entry

    called = []
    monkeypatch.setattr(entry, "serve", lambda: called.append("serve"))
    entry.main([])
    assert called == ["serve"]


def test_explicit_serve_command(monkeypatch):
    from agentgate import __main__ as entry

    called = []
    monkeypatch.setattr(entry, "serve", lambda: called.append("serve"))
    entry.main(["serve"])
    assert called == ["serve"]


def test_audit_command_forwards_remaining_argv(monkeypatch):
    from agentgate import __main__ as entry

    seen = []
    monkeypatch.setattr(entry, "serve", lambda: seen.append("serve"))
    monkeypatch.setattr(cli, "main", lambda argv: seen.append(argv))
    entry.main(["audit", "tail", "-n", "5", "--flagged"])
    assert seen == [["tail", "-n", "5", "--flagged"]]


def test_unknown_command_is_rejected():
    from agentgate import __main__ as entry

    with pytest.raises(SystemExit):
        entry.main(["frobnicate"])
