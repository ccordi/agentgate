"""Shared fixtures with a mocked upstream and temporary audit database."""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from agentgate.app import app as gateway_app
from agentgate.audit.crypto import ContentCipher
from agentgate.audit.store import AuditStore
from agentgate.config import Provider, Settings
from agentgate.limits.backend import MemoryBackend
from agentgate.limits.spend import SpendConfig, SpendTracker
from tests.support import make_settings, sse_handler


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
async def gateway(tmp_path):
    """Fresh app state on the shared FastAPI app, plus an ASGI client.

    Defaults mirror what the endpoint tests all wanted: the fast heuristic guard (these
    assert pipeline mechanics, not the model backend), a `gemini` provider pointed at the
    mock upstream, no content cipher, and `deberta_available=False` — a test that wants
    the model backend opts in explicitly.
    """
    settings = make_settings(database_url=f"sqlite+aiosqlite:///{tmp_path / 'audit.db'}")
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
        transport=httpx.ASGITransport(app=gateway_app), base_url="http://127.0.0.1"
    )
    handle.client = client
    try:
        yield handle
    finally:
        await client.aclose()
        await http.aclose()
        await store.close()
