"""Deployment posture: the compose file publishes to host loopback only.

Inside its container the gateway listens on all interfaces, which AGENTGATE_CONTAINER_BIND
permits, so what keeps it local is the compose file publishing each port to 127.0.0.1
only. A wildcard publish would expose the port and, because Docker writes its own iptables
rules, also bypass host firewalls. Every published port in deploy/docker-compose.yml must
name 127.0.0.1.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

COMPOSE = Path(__file__).resolve().parent.parent / "deploy" / "docker-compose.yml"


def published_ports(text: str) -> list[str]:
    """Every entry of every short-syntax `ports:` list in the file.

    Deliberately a 20-line scanner, not a YAML dependency: it only needs to find
    `ports:` blocks and their `- "..."` items, and it must fail loudly (empty
    result) if the file's shape drifts to something it can't read.
    """
    entries: list[str] = []
    in_ports = False
    ports_indent = 0
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        stripped = line.strip()
        if in_ports:
            if stripped.startswith("- ") and indent > ports_indent:
                entries.append(stripped[2:].strip().strip("\"'"))
                continue
            in_ports = False
        if stripped == "ports:":
            in_ports = True
            ports_indent = indent
    return entries


def test_every_publish_is_host_loopback():
    entries = published_ports(COMPOSE.read_text())
    # Non-vacuity: the gateway, grafana, redis, collector and tempo all publish.
    assert len(entries) >= 5, f"parser found only {entries} — compose layout drifted?"
    offenders = [e for e in entries if not e.startswith("127.0.0.1:")]
    assert not offenders, f"non-loopback publishes: {offenders}"


def service_blocks(text: str) -> dict[str, list[str]]:
    """Service name -> its stripped body lines, same textual style as above."""
    blocks: dict[str, list[str]] = {}
    current: list[str] | None = None
    in_services = False
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        stripped = line.strip()
        if indent == 0:
            in_services = stripped == "services:"
            current = None
            continue
        if in_services and indent == 2 and stripped.endswith(":"):
            current = blocks.setdefault(stripped[:-1], [])
            continue
        if current is not None:
            current.append(stripped)
    return blocks


def test_locally_built_tag_is_never_pulled():
    """Services reusing the gateway-built tag must pin `pull_policy: never`.

    The tag exists only locally; without the pin, a fresh `up` asks the registry
    for it — a squattable public name that would then run inside the gateway's
    network namespace.
    """
    blocks = service_blocks(COMPOSE.read_text())
    reusers = {
        name: lines
        for name, lines in blocks.items()
        if "image: agentgate:dev" in lines
        and not any(entry.startswith("build:") for entry in lines)
    }
    assert reusers, "parser found no service reusing the local tag — compose layout drifted?"
    offenders = [name for name, lines in reusers.items() if "pull_policy: never" not in lines]
    assert not offenders, f"local-tag services that may pull from a registry: {offenders}"


# The profile combinations the compose header documents. `obs` is not standalone:
# prometheus joins the gateway service's network namespace, so its documented
# command enables both profiles.
PROFILE_SETS = [("gateway",), ("gateway", "obs"), ("otel",)]


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker CLI not available")
@pytest.mark.parametrize("profiles", PROFILE_SETS)
def test_documented_profile_sets_compose(profiles):
    """`docker compose config` must validate every documented profile set.

    `config` is client-side — no daemon required — so this pins that no service
    references another outside its active profiles (the failure mode: a netns
    join onto a service the profile set doesn't enable).
    """
    cmd = ["docker", "compose", "-f", str(COMPOSE)]
    for profile in profiles:
        cmd += ["--profile", profile]
    cmd += ["config", "--quiet"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"profiles {profiles}: {proc.stderr}"


_REPO_ROOT = Path(__file__).resolve().parent.parent
_PRICE_MAP = Path("src") / "agentgate" / "data" / "model_prices.json"


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git to evaluate ignore rules")
def test_vendored_price_map_is_not_ignorable():
    """The price map ships inside the package, so no ignore rule may match it.

    hatchling honors .gitignore when it selects wheel contents, and an unanchored
    ``data/`` rule (meant for the audit-store directory) also matches
    ``src/agentgate/data/`` — the wheel then installs a gateway with no price map,
    and every cost lookup fails. ``git check-ignore --no-index`` asks the rules
    directly, so the file being tracked does not mask the defect.
    """
    assert (_REPO_ROOT / _PRICE_MAP).is_file()
    proc = subprocess.run(
        ["git", "check-ignore", "--no-index", "-v", str(_PRICE_MAP)],
        cwd=_REPO_ROOT, capture_output=True, text=True, check=False,
    )
    if proc.returncode == 128:
        pytest.skip("not a git checkout")
    # 1 = not ignored; 0 = ignored (the verbose line names the offending rule).
    assert proc.returncode == 1, f"price map is ignorable: {proc.stdout.strip()}"


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker CLI not available")
@pytest.mark.parametrize("real_upstream", [False, True])
def test_saved_setup_reaches_container_and_audit_volume(tmp_path, monkeypatch, real_upstream):
    """The settings `agentgate init` writes, plus those for a real model server
    (docs/docker.md), reach the container intact through `--env-file`, and the audit
    volume is mounted."""
    import json
    import os

    from agentgate.config import Settings, validate_runtime_settings

    for name in list(os.environ):
        if name.startswith(("AGENTGATE_", "COMPOSE_")):
            monkeypatch.delenv(name)
    deploy = tmp_path / "deploy"
    deploy.mkdir()
    compose = deploy / "docker-compose.yml"
    compose.write_text(COMPOSE.read_text())
    # Outside the project directory to pin the documented --env-file behavior.
    env_file = tmp_path / "saved-settings.env"
    lines = [
        "AGENTGATE_ADMIN_TOKEN=admin-test-token",
        "AGENTGATE_PDP_TOKEN=pdp-test-token",
        "AGENTGATE_GUARD_BACKEND=heuristic",
        "AGENTGATE_ROUTING__ENABLED=false",
        f"AGENTGATE_DEFAULT_PROVIDER={'local' if real_upstream else 'mock'}",
    ]
    if real_upstream:
        lines += [
            'AGENTGATE_PROVIDERS=\'{"local":{"name":"local","base_url":"http://host.docker.internal:8000","is_local":true}}\'',
            "AGENTGATE_LOCAL_MODEL_OVERRIDE=my-model",
            "AGENTGATE_LOCAL_API_KEY=upstream-test-key",
            'AGENTGATE_EGRESS__ALLOWLIST=\'["example.com"]\'',
        ]
    env_file.write_text("\n".join(lines) + "\n")
    result = subprocess.run(
        ["docker", "compose", "--env-file", str(env_file), "-f", str(compose),
         "--profile", "gateway", "config", "--format", "json"],
        capture_output=True, text=True, check=True,
    )
    service = json.loads(result.stdout)["services"]["agentgate"]
    for name, value in service["environment"].items():
        if value is not None:
            monkeypatch.setenv(name, value)
    settings = Settings(_env_file=None)
    validate_runtime_settings(settings)
    assert settings.admin_token == "admin-test-token"
    assert settings.pdp_token == "pdp-test-token"
    assert settings.guard_backend == "heuristic"
    assert not settings.routing.enabled
    assert settings.container_bind and settings.host == "0.0.0.0"
    if real_upstream:
        assert settings.default_provider == "local"
        assert settings.provider().base_url == "http://host.docker.internal:8000"
        assert settings.provider().is_local
        assert settings.provider().api_key == "upstream-test-key"
        assert settings.local_model_override == "my-model"
        assert settings.egress.allowlist == ["example.com"]
    else:
        assert settings.default_provider == "mock"
        assert settings.provider().base_url == "http://127.0.0.1:4200"
        assert settings.local_api_key is None
    volumes = service["volumes"]
    assert any(v["type"] == "volume" and v["source"] == "agentgate-data"
               and v["target"] == "/app/data" for v in volumes)


@pytest.mark.parametrize(
    ("bindings", "passes"),
    [
        ("4100/tcp -> 127.0.0.1:14100\n9090/tcp -> 127.0.0.1:9090", True),
        ("4100/tcp -> 0.0.0.0:14100", False),
        ("4100/tcp -> 127.0.0.1:14100\n9090/tcp -> 0.0.0.0:9090", False),
        ("4100/tcp -> [::]:14100", False),
        ("4100/tcp -> 127x0x0x1:14100", False),
        ("", False),
    ],
)
def test_smoke_checks_docker_port_output(bindings, passes):
    """Run the published-port check from scripts/container_smoke.sh against sample
    `docker port` output."""
    import os

    script = (_REPO_ROOT / "scripts" / "container_smoke.sh").read_text()
    start = script.index('cid="$(')
    success = "printf '%s\\n' 'PASS: loopback-only published ports'"
    stop = script.index(success, start) + len(success)
    # Replace only Docker; the assertion runs unchanged under its real shell options.
    stub = '''set -euo pipefail
COMPOSE=(docker compose)
docker() {
  case "$1" in
    compose) printf '%s\\n' test-container ;;
    port) printf '%s\\n' "$TEST_BINDINGS" ;;
    *) return 1 ;;
  esac
}
'''
    result = subprocess.run(
        ["bash", "-c", stub + script[start:stop]],
        env={**os.environ, "TEST_BINDINGS": bindings}, capture_output=True, text=True,
    )
    assert (result.returncode == 0) is passes, result.stdout + result.stderr
    assert ("PASS: loopback-only published ports" in result.stdout) is passes
