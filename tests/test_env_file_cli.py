"""Explicit configuration selection through the public CLI, without loading models."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import uvicorn

from agentgate import __main__ as entry
from agentgate.audit.store import AuditStore
from agentgate.config import Settings, get_settings
from agentgate.guards import local_llm
from agentgate.guards.deberta import configured_model_dir
from agentgate.guards.local_llm import live_cache_path
from agentgate.observability.otel import otlp_endpoint
from tests.support import make_audit

ROOT = Path(__file__).parents[1]
STARTER = """\
AGENTGATE_ADMIN_TOKEN=admin-token
AGENTGATE_PDP_TOKEN=tool-token
AGENTGATE_GUARD_BACKEND=heuristic
AGENTGATE_DEFAULT_PROVIDER=mock
AGENTGATE_ROUTING__ENABLED=false
"""


@pytest.fixture(autouse=True)
def isolate_judge_file(monkeypatch):
    monkeypatch.setitem(local_llm.JudgeConfig.model_config, "env_file", None)


@pytest.mark.parametrize("command", [[], ["serve"]])
@pytest.mark.parametrize("exported_port", [None, "4301"])
def test_startup_uses_selected_file_and_environment_precedence(
    tmp_path, monkeypatch, command, exported_port,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(Settings.model_config, "env_file", ".env")
    (tmp_path / ".env").write_text("AGENTGATE_PORT=4300\n")
    # Selection must discard a cached Settings and replace the default file.
    assert get_settings().port == 4300
    config_dir = tmp_path / "config dir"
    config_dir.mkdir()
    selected = config_dir / "custom.env"
    selected.write_text(
        STARTER
        + "AGENTGATE_PORT=4302\n"
        + "AGENTGATE_GUARD_MODEL_DIR=models/chosen\n"
        + "AGENTGATE_GUARD_CACHE_PATH=data/chosen.json\n"
        + "AGENTGATE_OTLP_ENDPOINT=http://127.0.0.1:4318/v1/traces\n"
    )
    if exported_port is not None:
        monkeypatch.setenv("AGENTGATE_PORT", exported_port)
    # Point-of-use readers must preserve environment precedence too, including empty.
    monkeypatch.setenv("AGENTGATE_GUARD_CACHE_PATH", "data/exported.json")
    monkeypatch.setenv("AGENTGATE_OTLP_ENDPOINT", "")

    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: calls.append((a, kw)))
    entry.main(["--env-file", str(selected.relative_to(tmp_path)), *command])

    assert len(calls) == 1
    assert calls[0][0] == ("agentgate.app:app",)
    assert calls[0][1]["port"] == int(exported_port or "4302")
    assert get_settings().admin_token == "admin-token"
    assert get_settings().default_provider == "mock"
    assert get_settings().guard_backend == "heuristic"
    assert configured_model_dir() == Path("models/chosen")
    assert live_cache_path() == Path("data/exported.json")
    assert otlp_endpoint() == ""
    assert Path.cwd() == tmp_path
    assert "AGENTGATE_ADMIN_TOKEN" not in os.environ


def test_startup_without_option_still_reads_working_directory_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(Settings.model_config, "env_file", ".env")
    (tmp_path / ".env").write_text(STARTER + "AGENTGATE_PORT=4303\n")
    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: calls.append(kw))
    entry.main([])
    assert calls[0]["port"] == 4303


@pytest.mark.parametrize("exported_model", [None, "exported-judge"])
def test_live_judge_uses_selected_file(tmp_path, monkeypatch, exported_model):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(local_llm.JudgeConfig.model_config, "env_file", ".env")
    (tmp_path / ".env").write_text("AGENTGATE_JUDGE_MODEL=wrong-judge\n")
    selected = tmp_path / "judge.env"
    selected.write_text(
        STARTER.replace("GUARD_BACKEND=heuristic", "GUARD_BACKEND=llm")
        + "AGENTGATE_JUDGE_MODEL=selected-judge\n"
        + "AGENTGATE_JUDGE_BASE_URL=http://127.0.0.1:4304/v1\n"
        + "AGENTGATE_JUDGE_API_KEY=judge-token\n"
        + "AGENTGATE_JUDGE_TIMEOUT_S=2\n"
        + "AGENTGATE_GUARD_CACHE_PATH=data/selected-guard.json\n"
    )
    if exported_model is not None:
        monkeypatch.setenv("AGENTGATE_JUDGE_MODEL", exported_model)
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: None)
    local_llm._guard.cache_clear()
    guard = None
    try:
        entry.main(["--env-file", str(selected)])
        assert get_settings().guard_backend == "llm"
        guard = local_llm._guard()  # Constructs the client; makes no model request.
        assert guard.cfg.model == (exported_model or "selected-judge")
        assert guard.cfg.base_url == "http://127.0.0.1:4304/v1"
        assert guard.cfg.api_key == "judge-token"
        assert guard.cfg.timeout_s == 2
        assert local_llm.JudgeConfig().model == guard.cfg.model
    finally:
        if guard is not None:
            guard.client.close()
        local_llm._guard.cache_clear()


@pytest.mark.parametrize("kind", ["missing", "directory", "unreadable"])
@pytest.mark.parametrize("command", [[], ["audit", "tail"]])
def test_bad_explicit_file_is_rejected_before_dispatch(
    tmp_path, monkeypatch, capsys, kind, command,
):
    selected = tmp_path / "settings.env"
    if kind == "directory":
        selected.mkdir()
    elif kind == "unreadable":
        if os.geteuid() == 0:
            pytest.skip("root can read a mode-000 file")
        selected.write_text(STARTER)
        selected.chmod(0)
    calls = []
    monkeypatch.setattr(entry, "serve", lambda: calls.append("serve"))
    from agentgate.audit import cli

    monkeypatch.setattr(cli, "main", lambda args: calls.append("audit"))
    try:
        with pytest.raises(SystemExit) as exc:
            entry.main(["--env-file", str(selected), *command])
        assert exc.value.code == 2
    finally:
        if kind == "unreadable":
            selected.chmod(0o600)
    assert calls == []
    assert "env file" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [
    ["--env-file"],
    ["serve", "--env-file", "settings.env"],
    ["serve", "--unknown-option"],
    ["--env-file", "settings.env", "init"],
])
def test_bad_options_never_start_or_initialize(tmp_path, monkeypatch, argv):
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr(entry, "serve", lambda: calls.append("serve"))
    monkeypatch.setattr(entry, "initialize", lambda path: calls.append("init"))
    with pytest.raises(SystemExit) as exc:
        entry.main(argv)
    assert exc.value.code == 2
    assert calls == []


async def test_audit_subprocess_reads_selected_database_from_original_workdir(tmp_path):
    workdir = tmp_path / "working"
    workdir.mkdir()
    config_dir = tmp_path / "configuration"
    config_dir.mkdir()
    selected = config_dir / "audit.env"
    # No serving tokens: audit needs only its database settings.
    selected.write_text("AGENTGATE_DATABASE_URL=sqlite+aiosqlite:///./chosen.db\n")
    (workdir / ".env").write_text(
        "AGENTGATE_DATABASE_URL=sqlite+aiosqlite:///./wrong.db\n"
    )
    store = AuditStore(f"sqlite+aiosqlite:///{workdir / 'chosen.db'}")
    await store.init()
    row = make_audit(agent_id="selected-file")
    await store.write(row)
    await store.close()

    result = subprocess.run(
        [sys.executable, "-m", "agentgate", "--env-file", str(selected),
         "audit", "tail", "--json", "-n", "1"],
        cwd=workdir,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    rows = json.loads(result.stdout)
    assert [record["agent_id"] for record in rows] == ["selected-file"]
    assert not (config_dir / "chosen.db").exists()
    assert not (workdir / "wrong.db").exists()
