"""Subprocess coverage for the dependency-free `agentgate init` path."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def run_init(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, "-S", "-m", "agentgate", "init", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def read_settings(path: Path) -> dict[str, str]:
    return dict(
        line.split("=", 1)
        for line in path.read_text().splitlines()
        if line and not line.startswith("#")
    )


def test_init_creates_private_model_free_configuration(tmp_path):
    result = run_init(tmp_path)

    env_path = tmp_path / ".env"
    assert result.returncode == 0, result.stderr
    assert str(env_path.resolve()) in result.stdout
    assert "docs/getting-started.md" in result.stdout
    assert result.stderr == ""
    assert stat.S_IMODE(env_path.stat().st_mode) == 0o600

    settings = read_settings(env_path)
    assert set(settings) == {
        "AGENTGATE_ADMIN_TOKEN",
        "AGENTGATE_PDP_TOKEN",
        "AGENTGATE_GUARD_BACKEND",
        "AGENTGATE_DEFAULT_PROVIDER",
        "AGENTGATE_ROUTING__ENABLED",
    }
    assert settings["AGENTGATE_ADMIN_TOKEN"] != settings["AGENTGATE_PDP_TOKEN"]
    assert len(settings["AGENTGATE_ADMIN_TOKEN"]) >= 40
    assert len(settings["AGENTGATE_PDP_TOKEN"]) >= 40
    assert settings["AGENTGATE_GUARD_BACKEND"] == "heuristic"
    assert settings["AGENTGATE_DEFAULT_PROVIDER"] == "mock"
    assert settings["AGENTGATE_ROUTING__ENABLED"] == "false"
    assert settings["AGENTGATE_ADMIN_TOKEN"] not in result.stdout
    assert settings["AGENTGATE_PDP_TOKEN"] not in result.stdout
    assert "Model-free starter configuration" in env_path.read_text()


def test_directory_option_writes_to_existing_directory(tmp_path):
    output_dir = tmp_path / "config"
    output_dir.mkdir()

    result = run_init(tmp_path, "--directory", str(output_dir))

    assert result.returncode == 0, result.stderr
    assert (output_dir / ".env").is_file()
    assert not (tmp_path / ".env").exists()


@pytest.mark.parametrize("target_kind", ["missing", "file"])
def test_directory_option_rejects_invalid_target(tmp_path, target_kind):
    target = tmp_path / target_kind
    if target_kind == "file":
        target.write_text("not a directory")

    result = run_init(tmp_path, "--directory", str(target))

    assert result.returncode == 2
    assert "agentgate init:" in result.stderr
    assert not (target / ".env").exists()
    assert not (tmp_path / ".env").exists()


def test_existing_env_is_preserved(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("keep=this\n")

    result = run_init(tmp_path)

    assert result.returncode == 2
    assert "refusing to overwrite existing" in result.stderr
    assert env_path.read_text() == "keep=this\n"


def test_existing_env_symlink_is_not_followed(tmp_path):
    destination = tmp_path / "destination"
    destination.write_text("keep=this\n")
    (tmp_path / ".env").symlink_to(destination)

    result = run_init(tmp_path)

    assert result.returncode == 2
    assert "refusing to overwrite existing" in result.stderr
    assert (tmp_path / ".env").is_symlink()
    assert destination.read_text() == "keep=this\n"


@pytest.mark.parametrize("args", [("unexpected",), ("--unknown-option",)])
def test_init_rejects_unrecognized_input_without_writing(tmp_path, args):
    result = run_init(tmp_path, *args)

    assert result.returncode == 2
    assert "usage: agentgate init" in result.stderr
    assert not (tmp_path / ".env").exists()
