"""The settings read at their point of use honour `.env` the way `Settings` does.

`config.env_setting` serves the values that are not fields on `Settings` — the classifier's
model directory, the LLM judge's cache path, the tracing endpoint and the MCP server's PDP
token and URL — so one `.env` works for the gateway, the MCP server and the LiteLLM
guardrail. Precedence is the same as for every other setting: the process
environment wins, even when the value is empty; the file fills what the environment
leaves unset; nothing is written into the environment.

The hermetic conftest fixture points `Settings.model_config["env_file"]` at nothing, so
each test here re-enables the file inside its own temporary working directory.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agentgate.config import Settings, env_setting

NAMES = (
    "AGENTGATE_PDP_TOKEN",
    "AGENTGATE_PDP_URL",
    "AGENTGATE_OTLP_ENDPOINT",
    "AGENTGATE_GUARD_MODEL_DIR",
    "AGENTGATE_GUARD_CACHE_PATH",
    "AGENTGATE_GUARD_INTRA_OP_THREADS",
)


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(Settings.model_config, "env_file", ".env")
    for name in NAMES:
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def test_file_value_is_used_when_nothing_is_exported(workdir: Path) -> None:
    (workdir / ".env").write_text(
        'AGENTGATE_PDP_TOKEN="from file"\n'
        "export AGENTGATE_OTLP_ENDPOINT=http://127.0.0.1:4318/v1/traces  # comment\n"
    )
    assert env_setting("AGENTGATE_PDP_TOKEN") == "from file"
    assert env_setting("AGENTGATE_OTLP_ENDPOINT") == "http://127.0.0.1:4318/v1/traces"


def test_exported_value_wins_even_when_empty(workdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (workdir / ".env").write_text("AGENTGATE_PDP_TOKEN=from-file\n")
    monkeypatch.setenv("AGENTGATE_PDP_TOKEN", "from-env")
    assert env_setting("AGENTGATE_PDP_TOKEN") == "from-env"
    # Same rule as pydantic-settings: set-but-empty is a value, not "unset".
    monkeypatch.setenv("AGENTGATE_PDP_TOKEN", "")
    assert env_setting("AGENTGATE_PDP_TOKEN") == ""


def test_missing_file_and_missing_key_fall_back_to_the_default(workdir: Path) -> None:
    assert env_setting("AGENTGATE_PDP_TOKEN") is None
    assert env_setting("AGENTGATE_GUARD_INTRA_OP_THREADS", "0") == "0"
    (workdir / ".env").write_text("AGENTGATE_PDP_TOKEN=tok\n")
    assert env_setting("AGENTGATE_GUARD_CACHE_PATH") is None
    assert env_setting("AGENTGATE_GUARD_CACHE_PATH", "data/x.json") == "data/x.json"


def test_the_file_is_read_not_sourced(workdir: Path) -> None:
    (workdir / ".env").write_text("AGENTGATE_PDP_TOKEN=tok\n")
    assert env_setting("AGENTGATE_PDP_TOKEN") == "tok"
    assert "AGENTGATE_PDP_TOKEN" not in os.environ


def test_hermetic_fixture_also_cuts_off_this_reader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The suite-wide isolation (`Settings.model_config["env_file"] = None`) must cover
    these reads too, or a developer's live `.env` would leak into unrelated tests."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AGENTGATE_PDP_TOKEN", raising=False)
    (tmp_path / ".env").write_text("AGENTGATE_PDP_TOKEN=leak\n")
    assert Settings.model_config.get("env_file") is None  # the conftest fixture's doing
    assert env_setting("AGENTGATE_PDP_TOKEN") is None


def test_mcp_server_reads_its_token_and_pdp_url_from_the_file(workdir: Path) -> None:
    pytest.importorskip("mcp")
    from agentgate.egress import mcp_server

    (workdir / ".env").write_text(
        "AGENTGATE_PDP_TOKEN=tok\n"
        "AGENTGATE_PDP_URL=http://127.0.0.1:4999/a/egress/decision\n"
    )
    assert mcp_server._bearer() == "tok"
    assert mcp_server._pdp_url() == "http://127.0.0.1:4999/a/egress/decision"


def test_mcp_server_fails_closed_without_a_token(workdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No token anywhere → None → the PDP 401s. An empty exported token also counts as
    set (OpenCode substitutes an unset `{env:...}` as ""), so it must not fall through
    to the file's token: the OpenCode guide (docs/opencode.md) therefore relies on `.env`
    rather than that mapping."""
    pytest.importorskip("mcp")
    from agentgate.egress import mcp_server

    assert mcp_server._bearer() is None
    assert mcp_server._pdp_url() == mcp_server.DEFAULT_PDP_URL
    (workdir / ".env").write_text("AGENTGATE_PDP_TOKEN=tok\n")
    monkeypatch.setenv("AGENTGATE_PDP_TOKEN", "")
    assert mcp_server._bearer() is None


def test_tracing_endpoint_and_guard_paths_read_the_file(workdir: Path) -> None:
    from agentgate.guards import deberta, local_llm
    from agentgate.observability.otel import otlp_endpoint

    assert otlp_endpoint() is None
    assert deberta.configured_model_dir() == Path(deberta._DEFAULT_DIR)
    assert local_llm.live_cache_path() == local_llm._LIVE_CACHE_PATH
    (workdir / ".env").write_text(
        "AGENTGATE_OTLP_ENDPOINT=http://127.0.0.1:4318/v1/traces\n"
        "AGENTGATE_GUARD_MODEL_DIR=/models/piguard-onnx\n"
        "AGENTGATE_GUARD_CACHE_PATH=data/other_cache.json\n"
    )
    assert otlp_endpoint() == "http://127.0.0.1:4318/v1/traces"
    assert deberta.configured_model_dir() == Path("/models/piguard-onnx")
    assert local_llm.live_cache_path() == Path("data/other_cache.json")
