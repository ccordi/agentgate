"""Shared fixtures + suite-wide isolation from the working tree's live configuration.

Two things live here.

**`hermetic_settings` (autouse).** `Settings` is a `BaseSettings` carrying
`env_prefix="AGENTGATE_"` and `env_file=".env"`, so every `Settings()` a test constructs
would silently inherit whatever the developer's deployment happens to be configured
with — arming an opt-in gate in the live `.env` (issued keys, the admin token) can
flip tests that assert the unarmed defaults, with no source change at all.

The autouse fixture closes that: inside a test, no `AGENTGATE_*` variable and no `.env`
file reaches a `Settings()`. A test that depends on a setting has to name it at
construction, which is also how it documents the dependency.

**`gateway`.** One fixture for the app-state bootstrap. It populates `app.state` the way
`lifespan` does, but with a mocked upstream and a temp SQLite audit store, and it
guarantees teardown, closing its clients even when the test fails.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable, Iterator

import httpx
import pytest

from agentgate.app import app as gateway_app
from agentgate.audit.crypto import ContentCipher
from agentgate.audit.store import AuditStore
from agentgate.config import Provider, Settings, get_settings
from agentgate.limits.backend import MemoryBackend
from agentgate.limits.spend import SpendConfig, SpendTracker
from tests.support import make_settings, sse_handler

# The ASGI client's base_url has to be a loopback *name*: `LoopbackHostGuard` rejects
# any request whose Host header is not one (anti-DNS-rebinding), and httpx
# derives Host from base_url. A cosmetic hostname like "http://gw" 400s every request.
GATEWAY_BASE_URL = "http://127.0.0.1"

# Postgres opt-in for the suite, captured at import time: `hermetic_settings` below
# deletes every AGENTGATE_* variable for the duration of each test, so a fixture that
# read the environment lazily would never see it.
TEST_DATABASE_URL = os.environ.get("AGENTGATE_TEST_DATABASE_URL")


@pytest.fixture
async def audit_db_url(tmp_path):
    """Per-test audit DB URL — a temp SQLite file by default.

    With AGENTGATE_TEST_DATABASE_URL set (see scripts/test_postgres.sh), each test
    instead gets a freshly created database on that server, dropped at teardown —
    the same isolation a tmp_path file gives SQLite.
    """
    if TEST_DATABASE_URL is None:
        yield f"sqlite+aiosqlite:///{tmp_path / 'audit.db'}"
        return

    from sqlalchemy import text
    from sqlalchemy.engine import make_url
    from sqlalchemy.ext.asyncio import create_async_engine

    name = f"agentgate_test_{uuid.uuid4().hex[:12]}"
    # CREATE/DROP DATABASE cannot run inside a transaction, hence AUTOCOMMIT.
    admin = create_async_engine(TEST_DATABASE_URL, isolation_level="AUTOCOMMIT")
    try:
        async with admin.connect() as conn:
            await conn.execute(text(f'CREATE DATABASE "{name}"'))
        yield make_url(TEST_DATABASE_URL).set(database=name).render_as_string(
            hide_password=False
        )
        async with admin.connect() as conn:
            # FORCE terminates any connection a failed test left behind.
            await conn.execute(text(f'DROP DATABASE "{name}" WITH (FORCE)'))
    finally:
        await admin.dispose()


@pytest.fixture(autouse=True)
def hermetic_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Cut every `Settings()` built during a test off from the working tree."""
    for key in list(os.environ):
        if key.startswith("AGENTGATE_"):
            monkeypatch.delenv(key, raising=False)

    # `env_file` is read out of `model_config` at construction time, so overriding the
    # entry is enough; monkeypatch restores the dict after each test.
    monkeypatch.setitem(Settings.model_config, "env_file", None)

    # `get_settings` is `@lru_cache`d: a Settings built before the patch would otherwise
    # outlive it and leak into whichever test happens to call it first.
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class Gateway:
    """Handle for one test's gateway: the app, an ASGI client, its store and settings.

    Settings are per-test, so a test mutates `gw.settings.<field>` freely and never has
    to restore anything — the next test builds a fresh object.
    """

    def __init__(self, app, client: httpx.AsyncClient, store: AuditStore, settings: Settings):
        self.app = app
        self.client = client
        self.store = store
        self.settings = settings
        self.handler: Callable[[httpx.Request], httpx.Response] = sse_handler()

    def set_upstream(self, handler: Callable[[httpx.Request], httpx.Response]) -> None:
        """Swap the upstream handler the gateway's outbound client dispatches to."""
        self.handler = handler

    def set_cipher(self, key: str | None) -> None:
        """Enable (or disable) the content tier with an explicit Fernet key."""
        self.settings.content_enc_key = key
        self.app.state.cipher = ContentCipher(key)


@pytest.fixture
async def gateway(audit_db_url):
    """Fresh app state on the shared FastAPI app, plus an ASGI client.

    Defaults mirror what the endpoint tests all wanted: the heuristic scanner (these
    assert pipeline mechanics, not the model backend), a `gemini` provider pointed at the
    mock model server and named as the routing-off default and the cloud branch (the shipped
    default for both is `local`; tests here want the cloud route reachable), no content
    cipher, and `deberta_available=False` — a test that wants the model backend opts in
    explicitly.
    """
    settings = make_settings(
        database_url=audit_db_url,
        default_provider="gemini",
        routing={"default_cloud": "gemini"},
    )
    settings.guard_backend = "heuristic"
    settings.providers["gemini"] = Provider(
        name="gemini", base_url="http://mock", chat_completions_path="/v1/chat/completions"
    )

    store = AuditStore(settings.database_url)
    await store.init()

    handle = Gateway(gateway_app, None, store, settings)  # client assigned below

    # One outbound client for the whole test; `set_upstream` swaps the handler behind it
    # rather than rebuilding (and leaking) a client per swap.
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: handle.handler(request)),
        base_url="http://mock",
    )

    gateway_app.state.settings = settings
    gateway_app.state.http = http
    gateway_app.state.audit = store
    gateway_app.state.spend = SpendTracker(MemoryBackend(), SpendConfig())
    gateway_app.state.cipher = ContentCipher(None)
    gateway_app.state.deberta_available = False

    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=gateway_app), base_url=GATEWAY_BASE_URL
    )
    handle.client = client
    try:
        yield handle
    finally:
        await client.aclose()
        await http.aclose()
        await store.close()
