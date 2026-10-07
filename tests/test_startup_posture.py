"""Startup posture: loopback bind enforced, gateway tokens mandatory, Host guarded.

The loopback binding is a load-bearing control: the threat model's single operator on
a single host, with the gateway on loopback, justifies every unauthenticated surface
(the proxy routes have no inbound auth by design), so `AGENTGATE_HOST=0.0.0.0` must not
flip the premise with one env var and no warning (containerizing makes that the path of
least resistance). These tests pin:

- `validate_runtime_settings` refuses a non-loopback bind. The one exception is
  `AGENTGATE_CONTAINER_BIND` — the container contract (bind wide inside an isolated
  namespace whose ingress is published to host loopback only), which touches the bind
  check and nothing else; there is still no generic override.
- It refuses to serve without both dedicated tokens (`AGENTGATE_ADMIN_TOKEN`,
  `AGENTGATE_PDP_TOKEN`): conditional auth would leave deployments without them
  looking closed while open.
- `LoopbackHostGuard` rejects non-loopback Host headers: loopback excludes remote
  *sockets*, not remote *code* — a DNS-rebound page reaches 127.0.0.1 same-origin, and
  its Host header is the tell.

Known gap (documented in the threat model): a direct `uvicorn agentgate.app:app --host
0.0.0.0` never consults `settings.host`, so the bind check cannot see it. The lifespan
still enforces the token half there.
"""

from __future__ import annotations

import logging

import httpx
import pytest

from agentgate.app import app as gateway_app
from agentgate.app import lifespan
from agentgate.config import Provider, Settings, validate_runtime_settings
from agentgate.egress.policy import is_loopback_host

# ---- the one loopback definition ---------------------------------------------------

@pytest.mark.parametrize("value", [
    "127.0.0.1", "localhost", "LOCALHOST", "::1", "[::1]",
    "127.0.0.1:4100", "localhost:4100", "[::1]:4100",
])
def test_loopback_forms_accepted(value):
    assert is_loopback_host(value)


@pytest.mark.parametrize("value", [
    "0.0.0.0", "::", "10.0.0.5", "evil.example", "evil.example:4100",
    "gw", "testserver", "",
    # Real loopback addresses outside the canonical set are deliberately treated as
    # remote: under-inclusion fails strict (a refused start, a rejected Host), never
    # open. One definition, shared with the egress policy's loopback axis.
    "127.0.0.2", "127.0.0.53",
])
def test_everything_else_is_treated_as_remote(value):
    assert not is_loopback_host(value)


# ---- startup validation --------------------------------------------------------------

def _ok_settings(**overrides) -> Settings:
    base = dict(admin_token="admin-t", pdp_token="pdp-t")
    base.update(overrides)
    return Settings(**base)


def test_valid_posture_is_accepted():
    validate_runtime_settings(_ok_settings())  # loopback default host + both tokens


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10"])
def test_non_loopback_bind_refuses_startup(host):
    with pytest.raises(RuntimeError, match="not a loopback address"):
        validate_runtime_settings(_ok_settings(host=host))


def test_the_only_bind_exception_is_the_container_contract():
    """The one bind exception is container deployments, where bind address and
    exposure decouple and ingress is published to host loopback only. The narrow flag
    exists; a generic allow-non-loopback escape hatch must not."""
    fields = set(Settings.model_fields)
    assert "container_bind" in fields
    assert not any("allow" in f and "loopback" in f for f in fields)
    assert "allow_non_loopback" not in fields


def test_container_bind_defaults_off():
    assert Settings(_env_file=None).container_bind is False


def test_container_bind_permits_a_wide_bind():
    validate_runtime_settings(_ok_settings(host="0.0.0.0", container_bind=True))


def test_container_bind_relaxes_nothing_but_the_bind():
    """The flag's whole reach is the bind check: tokens stay mandatory and distinct
    under it."""
    with pytest.raises(RuntimeError, match="AGENTGATE_ADMIN_TOKEN"):
        validate_runtime_settings(
            Settings(container_bind=True, host="0.0.0.0", pdp_token="pdp-t")
        )
    with pytest.raises(RuntimeError, match="same value"):
        validate_runtime_settings(
            _ok_settings(container_bind=True, admin_token="same-t", pdp_token="same-t")
        )


def test_missing_admin_token_refuses_startup():
    with pytest.raises(RuntimeError, match="AGENTGATE_ADMIN_TOKEN"):
        validate_runtime_settings(_ok_settings(admin_token=None))


def test_missing_pdp_token_refuses_startup():
    with pytest.raises(RuntimeError, match="AGENTGATE_PDP_TOKEN"):
        validate_runtime_settings(_ok_settings(pdp_token=None))


# Presence is not separation. The PEP loads the PDP token into the agent's own
# environment, so admin == pdp hands the governed agent the kill switch meant to halt it.

def test_admin_and_pdp_tokens_must_differ():
    with pytest.raises(RuntimeError, match="same value"):
        validate_runtime_settings(_ok_settings(admin_token="same-t", pdp_token="same-t"))


def test_admin_token_must_not_reuse_the_local_api_key():
    with pytest.raises(RuntimeError, match="same value"):
        validate_runtime_settings(_ok_settings(admin_token="k", local_api_key="k"))


def test_pdp_token_must_not_reuse_the_local_api_key():
    with pytest.raises(RuntimeError, match="same value"):
        validate_runtime_settings(_ok_settings(pdp_token="k", local_api_key="k"))


def test_distinct_tokens_alongside_a_local_api_key_are_accepted():
    """Non-vacuity: the check must not fire on a correctly-configured deployment."""
    validate_runtime_settings(_ok_settings(local_api_key="a-third-value"))


@pytest.mark.asyncio
async def test_lifespan_enforces_the_posture():
    """Every sanctioned launch path runs lifespan (uvicorn always does), so an unsafe
    posture must die there — before any state is assembled. The hermetic conftest
    guarantees get_settings() sees no tokens here."""
    with pytest.raises(RuntimeError, match="unsafe posture"):
        async with lifespan(gateway_app):
            pass  # pragma: no cover — must not be reached


# ---- a configured scanner that cannot load is a refused start -----------------------
# Warning and falling back to the heuristic scanner when the classifier fails to warm would
# leave a deployment that asked for the classifier running the heuristic scanner for its
# whole life, with audit rows recording a scan that had happened — by a scanner nobody
# chose. So the gateway refuses to start. These pin each way a backend can *reach*
# deberta, and that a configuration which never reaches it still boots with no model files
# present at all.

def _no_deberta(monkeypatch) -> None:
    """Make the warmup fail exactly the way a missing model file does."""
    from agentgate.guards import deberta

    def boom() -> None:
        raise FileNotFoundError("guard model not found at models/…/model.onnx")

    monkeypatch.setattr(deberta, "warmup", boom)


def _lifespan_settings(tmp_path, **overrides) -> Settings:
    """Settings that pass the startup checks and that lifespan can assemble without
    touching the host: a temp audit DB, and a Redis URL nothing listens on (memory
    backend, immediately)."""
    return _ok_settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'audit.db'}",
        redis_url=UNREACHABLE_REDIS,
        **overrides,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("overrides", [
    pytest.param({"guard_backend": "deberta"}, id="global-default"),
    pytest.param({"guard_backend": "heuristic",
                  "guard_backend_overrides": {"key-a": "deberta"}}, id="per-key-override"),
    pytest.param({"guard_backend": "combined"}, id="combined"),
])
async def test_unloadable_deberta_refuses_startup(monkeypatch, caplog, tmp_path, overrides):
    """Any backend this process can *resolve to* counts: the global default, a per-key
    override while the default needs no model, and `combined`, which needs both halves."""
    _no_deberta(monkeypatch)
    settings = _lifespan_settings(tmp_path, **overrides)
    monkeypatch.setattr("agentgate.app.get_settings", lambda: settings)

    with caplog.at_level(logging.CRITICAL, logger="agentgate"):
        with pytest.raises(RuntimeError, match="refuses to start") as exc:
            async with lifespan(gateway_app):
                pass  # pragma: no cover — must not be reached

    # The refusal names the cause and the two ways out, because it is the only thing the
    # operator gets: uvicorn exits non-zero and there is no process left to ask.
    message = str(exc.value)
    assert "guard model not found" in message
    assert "AGENTGATE_GUARD_MODEL_DIR" in message and "`guard` extra" in message
    assert any("DEBERTA GUARD UNAVAILABLE" in r.getMessage() for r in caplog.records)


def test_missing_model_names_the_build_command_for_its_directory(monkeypatch, tmp_path):
    """The build hint writes where the gateway looks: the plain command for the default
    directory, `--out` for a configured one (the plain command would build elsewhere)."""
    pytest.importorskip("onnxruntime")
    from agentgate.guards import deberta

    monkeypatch.chdir(tmp_path)  # no models/ here, so the default directory is missing too
    with pytest.raises(FileNotFoundError) as exc:
        deberta.DebertaGuard()
    assert "`uv run --script scripts/convert_piguard_onnx.py` or set" in str(exc.value)

    configured = tmp_path / "my models"
    with pytest.raises(FileNotFoundError) as exc:
        deberta.DebertaGuard(str(configured))
    assert f"convert_piguard_onnx.py --out '{configured}'`" in str(exc.value)  # quoted


@pytest.mark.asyncio
async def test_a_posture_that_never_resolves_to_deberta_still_starts(monkeypatch, tmp_path):
    """`heuristic` default plus an `llm` override: nothing can resolve to deberta, so the
    model is never loaded and its absence is not an error. The model files are genuinely
    absent here — the warmup is patched to fail, and is never called."""
    _no_deberta(monkeypatch)
    settings = _lifespan_settings(tmp_path, guard_backend="heuristic",
                                  guard_backend_overrides={"key-a": "llm"})
    monkeypatch.setattr("agentgate.app.get_settings", lambda: settings)

    async with lifespan(gateway_app):
        assert gateway_app.state.deberta_available is False
        assert gateway_app.state.settings is settings


# ---- issued keys + keyless non-local providers: warn loudly, never refuse ------------
# The forward point enforces the invariant per request (a minted key never leaves the
# gateway — pipeline.screen_upstream_credentials, 503 upstream_credentials_missing).
# Boot only warns: a keyless cloud entry is valid as long as routing never selects it,
# so a blanket boot refusal would break forced-local deployments.

def _credential_warnings(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if "upstream_credentials_missing" in r.getMessage()]


def test_issued_keys_with_keyless_cloud_providers_warns_at_boot(caplog):
    with caplog.at_level(logging.WARNING, logger="agentgate"):
        validate_runtime_settings(_ok_settings(require_issued_keys=True))
    (record,) = _credential_warnings(caplog)
    assert record.levelno == logging.WARNING
    # The default registry's non-local entries are all keyless — each is named.
    for name in ("gemini", "openai", "mock"):
        assert name in record.getMessage()


def test_no_warning_when_issued_keys_are_off(caplog):
    with caplog.at_level(logging.WARNING, logger="agentgate"):
        validate_runtime_settings(_ok_settings())
    assert not _credential_warnings(caplog)


def test_no_warning_when_non_local_providers_carry_their_own_keys(caplog):
    providers = {
        "gemini": Provider(name="gemini", base_url="https://cloud.example",
                           api_key="upstream-cred"),
        "local": Provider(name="local", base_url="http://127.0.0.1:8000", is_local=True),
    }
    with caplog.at_level(logging.WARNING, logger="agentgate"):
        validate_runtime_settings(
            _ok_settings(require_issued_keys=True, providers=providers))
    assert not _credential_warnings(caplog)


# ---- Host guard (anti-DNS-rebinding) -------------------------------------------------

@pytest.mark.asyncio
async def test_rebound_host_is_rejected_before_any_route():
    """A rebound page's request arrives on the loopback socket but says
    `Host: evil.example` — the middleware must 400 it without touching routes or
    state (this test assembles neither)."""
    transport = httpx.ASGITransport(app=gateway_app)
    async with httpx.AsyncClient(transport=transport,
                                 base_url="http://evil.example:4100") as client:
        r = await client.get("/healthz")
        assert r.status_code == 400
        assert "loopback" in r.json()["detail"]


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", ["http://127.0.0.1", "http://localhost:4100"])
async def test_loopback_host_passes(origin):
    transport = httpx.ASGITransport(app=gateway_app)
    async with httpx.AsyncClient(transport=transport, base_url=origin) as client:
        r = await client.get("/healthz")
        assert r.status_code == 200


# ---- AGENTGATE_REQUIRE_SHARED_LIMITS: no in-process fallback when running replicas ----
# `make_backend` falls back to in-process counters when Redis is unreachable — right
# for a single process, wrong across replicas, where each one would get its own spend
# counters and kill switch. AGENTGATE_REQUIRE_SHARED_LIMITS turns the fallback into a
# refused boot.

# Nothing listens on port 1, so the connection fails immediately.
UNREACHABLE_REDIS = "redis://127.0.0.1:1/0"


async def test_require_shared_limits_refuses_boot_without_redis():
    from agentgate.limits.backend import make_backend

    with pytest.raises(RuntimeError, match="REQUIRE_SHARED_LIMITS"):
        await make_backend(UNREACHABLE_REDIS, require=True)


async def test_default_still_falls_back_to_memory():
    """Single-process deployments keep working with no Redis at all."""
    from agentgate.limits.backend import MemoryBackend, make_backend

    backend = await make_backend(UNREACHABLE_REDIS)
    assert isinstance(backend, MemoryBackend)


def test_require_shared_limits_defaults_off():
    assert Settings(_env_file=None).require_shared_limits is False


# ---- connection URLs never reach a log or a traceback verbatim -----------------------
# Startup names the audit store and the limits backend by URL, and the shared-limits
# refusal names the Redis it could not reach. `postgresql://user:pw@host/db` and
# `redis://user:pw@host` are ordinary spellings, so logging them verbatim would write the
# operator's password into the log file and into whatever collects the traceback.

SYNTHETIC_PASSWORD = "not-a-real-password"


def test_only_the_url_password_is_masked():
    from agentgate.redaction import mask_url_password

    assert mask_url_password(f"redis://ops:{SYNTHETIC_PASSWORD}@redis.invalid:6379/0") == (
        "redis://ops:***@redis.invalid:6379/0")
    assert mask_url_password(
        f"postgresql+asyncpg://ops:{SYNTHETIC_PASSWORD}@db.invalid/agentgate") == (
        "postgresql+asyncpg://ops:***@db.invalid/agentgate")
    # A password with no user is still a password.
    assert mask_url_password(f"redis://:{SYNTHETIC_PASSWORD}@redis.invalid:6379/0") == (
        "redis://:***@redis.invalid:6379/0")
    # No userinfo: byte-identical, so the default sqlite path URL cannot be mangled on
    # the way to the log line that exists to tell the operator which store came up.
    for url in ("sqlite+aiosqlite:///./data/agentgate.db", "sqlite+aiosqlite:///:memory:",
                "redis://127.0.0.1:6379/0", "redis://ops@redis.invalid:6379/0"):
        assert mask_url_password(url) == url


async def test_audit_store_ready_line_masks_the_password(caplog):
    from unittest.mock import AsyncMock, MagicMock, patch

    from agentgate.audit.store import AuditStore

    url = f"postgresql+asyncpg://ops:{SYNTHETIC_PASSWORD}@db.invalid/agentgate"
    engine = MagicMock()  # mocked all the way down: nothing connects to anything
    engine.begin.return_value.__aenter__.return_value.run_sync = AsyncMock()
    with caplog.at_level(logging.INFO, logger="agentgate"), \
            patch("agentgate.audit.store.create_async_engine", return_value=engine), \
            patch("agentgate.audit.store.async_sessionmaker"):
        await AuditStore(url).init()
    lines = [r.getMessage() for r in caplog.records if "audit store ready" in r.getMessage()]
    assert len(lines) == 1
    assert SYNTHETIC_PASSWORD not in lines[0]
    assert "postgresql+asyncpg://ops:***@db.invalid/agentgate" in lines[0]


async def test_limits_backend_line_masks_the_password(caplog):
    from unittest.mock import AsyncMock, MagicMock, patch

    from agentgate.limits.backend import make_backend

    url = f"redis://ops:{SYNTHETIC_PASSWORD}@redis.invalid:6379/0"
    client = MagicMock()
    client.ping = AsyncMock(return_value=True)
    with caplog.at_level(logging.INFO, logger="agentgate"), \
            patch("redis.asyncio.from_url", return_value=client):
        await make_backend(url)
    lines = [r.getMessage() for r in caplog.records if "limits backend" in r.getMessage()]
    assert len(lines) == 1
    assert SYNTHETIC_PASSWORD not in lines[0]
    assert "redis://ops:***@redis.invalid:6379/0" in lines[0]


async def test_require_shared_limits_refusal_masks_the_password():
    """The refusal is a RuntimeError that aborts boot, so its text is the one most
    likely to be pasted into an issue or picked up by a crash reporter."""
    from agentgate.limits.backend import make_backend

    url = f"redis://ops:{SYNTHETIC_PASSWORD}@127.0.0.1:1/0"
    with pytest.raises(RuntimeError) as excinfo:
        await make_backend(url, require=True)
    message = str(excinfo.value)
    assert SYNTHETIC_PASSWORD not in message
    assert "redis://ops:***@127.0.0.1:1/0" in message
