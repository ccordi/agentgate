"""FastAPI application assembly — the gateway's inbound face.

This module owns process lifecycle and the routes that aren't traffic: startup/shutdown
(`lifespan`), health, metrics, the kill switch, and the two router includes. It holds no
request-handling logic of its own.

The two data planes it assembles:
  - `pipeline` — the model wire (`/v1/chat/completions` and friends). Scan, route,
    redact, forward, audit.
  - `egress.api` — the off-wire control plane (`/a/egress/decision`), the tier-3 PDP
    an agent harness consults before performing a tool call.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from agentgate import guards, pipeline
from agentgate.audit.crypto import ContentCipher
from agentgate.audit.store import AuditStore
from agentgate.config import get_settings, validate_runtime_settings
from agentgate.egress import api as egress_api
from agentgate.egress.policy import is_loopback_host
from agentgate.limits.backend import make_backend
from agentgate.limits.spend import SpendConfig, SpendTracker
from agentgate.observability.otel import setup_tracing

log = logging.getLogger("agentgate")


async def _sweeper_loop(audit: AuditStore, interval_s: float = 3600.0) -> None:
    """Background task: delete expired content_samples every interval_s seconds."""
    while True:
        try:
            deleted = await audit.purge_expired_content()
            if deleted:
                log.info("content sweeper: purged %d expired rows", deleted)
        except Exception:  # noqa: BLE001
            log.exception("content sweeper error")
        await asyncio.sleep(interval_s)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    # Refuse to serve with an unsafe posture (loopback bind, both gateway tokens set).
    # Here, not just the CLI entry point, so `uvicorn agentgate.app:app` is covered too —
    # uvicorn runs lifespan on every launch path. The one gap is a direct `--host` flag,
    # which never consults settings; that residual is documented in the threat model.
    validate_runtime_settings(settings)
    app.state.http = httpx.AsyncClient(timeout=settings.upstream_timeout_s)
    app.state.settings = settings
    app.state.audit = AuditStore(settings.database_url)
    await app.state.audit.init()
    app.state.spend = SpendTracker(await make_backend(settings.redis_url), SpendConfig())
    app.state.cipher = ContentCipher(settings.content_enc_key)
    if not app.state.cipher.enabled:
        log.info("content capture disabled (AGENTGATE_CONTENT_ENC_KEY not set)")
    sweeper = asyncio.create_task(_sweeper_loop(app.state.audit))
    # Warm up deberta if it's reachable from *any* configured backend — the global
    # default or a per-key override — and record whether it's actually available.
    # Per-request dispatch consults this flag, so one key's fallback cannot change the
    # backend selected for another key.
    needs_deberta = guards.requires(settings, "deberta")
    deberta_available = False
    if needs_deberta:
        try:
            log.info("loading deberta injection guard…")
            from agentgate.guards import deberta
            await asyncio.to_thread(deberta.warmup)
            deberta_available = True
        except Exception as exc:
            # Model/runtime unavailable → degrade rather than hard-fail (a dead gateway
            # blocks all agent traffic). The guard falls back per-request when this is False.
            log.warning("deberta guard unavailable (%s); per-request fallback will apply", exc)
    app.state.deberta_available = deberta_available
    log.info("agentgate up — default upstream: %s, guard: %s, deberta_available: %s",
             settings.default_provider, settings.guard_backend, deberta_available)
    try:
        yield
    finally:
        sweeper.cancel()
        try:
            await sweeper
        except asyncio.CancelledError:
            pass
        await app.state.http.aclose()
        await app.state.audit.close()


class LoopbackHostGuard:
    """Reject any request whose `Host` header is not a loopback name (400).

    Closes DNS rebinding: a page at evil.example re-points that name at 127.0.0.1 and
    fetches http://evil.example:4100/… — now same-origin, so the browser applies no
    cross-origin protections at all: every method, readable responses. The kernel can't
    help (the socket IS loopback); the tell is the Host header, which necessarily still
    says evil.example. Legitimate clients point base_url at loopback and therefore send
    a loopback Host by construction.

    This is the companion to the bearer tokens, not a substitute: tokens exclude
    browser-borne callers from the admin plane and the PDP even if this check had a
    parsing edge; this check covers the proxy routes, which are tokenless by design.
    What neither closes — blind cross-origin POSTs burning local compute — is in the
    threat model as residual. Loopback = egress.policy's `_LOOPBACK_HOSTS`, the
    project's one definition.
    """

    def __init__(self, app):  # pure ASGI middleware: no BaseHTTPMiddleware overhead
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        host = ""
        for name, value in scope["headers"]:
            if name == b"host":
                host = value.decode("latin-1")
                break
        if not is_loopback_host(host):
            response = JSONResponse(
                {"detail": f"Host {host!r} is not a loopback name; refusing "
                           "(anti-DNS-rebinding). The gateway serves loopback only."},
                status_code=400,
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


app = FastAPI(title="agentgate", version="0.0.1", lifespan=lifespan)
app.add_middleware(LoopbackHostGuard)
setup_tracing(app)
app.include_router(pipeline.router)
app.include_router(egress_api.router)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
async def prometheus_metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _check_admin_auth(request: Request, settings) -> None:
    """Mandatory bearer auth for the admin plane (`/admin/kill/*`).

    Accepts exactly the **dedicated** ``AGENTGATE_ADMIN_TOKEN`` — startup refuses to
    serve without one (`validate_runtime_settings`), so the unset branch here fails
    closed (503) rather than open, as defense in depth for app objects assembled
    without lifespan.

    Why this endpoint and not the proxy: the kill switch is the one control aimed *at* the
    agent, and the agent reaches loopback by policy. Leaving it open means the halt can be
    cleared by the thing it exists to halt. And auth on loopback is not theater: loopback
    excludes remote *sockets*, not remote *code* — browser-borne requests reach 127.0.0.1
    (blind CSRF; readable with DNS rebinding, which `LoopbackHostGuard` closes
    independently). The proxy routes stay unauthenticated, deliberately — their inbound
    auth is the loopback bind itself, enforced at startup.

    There is no fallback to ``AGENTGATE_LOCAL_API_KEY``. That value is the local
    upstream's credential, and the PEP reads it from its own environment
    (`egress/mcp_server.py`). Accepting it here would authenticate the governed agent to the
    admin plane using a credential it already holds.

    Not a tenancy boundary: `key_id` is a hash of whatever the client sends, and the path
    segment chooses *whose* kill switch. This only stops the governed agent from clearing
    its own.
    """
    expected = settings.admin_token
    if not expected:
        raise HTTPException(status_code=503, detail="admin token not configured")
    auth = request.headers.get("authorization", "")
    # Constant-time compare: this guards a privilege boundary, so the equality check must
    # not leak the expected token byte-by-byte via timing.
    # Compared as bytes: Starlette decodes headers as latin-1, and compare_digest raises
    # TypeError on a non-ASCII str — which would turn a 401 into an unhandled 500 on a
    # header a raw-socket caller controls. latin-1 round-trips back to the wire bytes.
    if not hmac.compare_digest(auth.encode("latin-1", "replace"),
                               f"Bearer {expected}".encode()):
        raise HTTPException(status_code=401, detail="unauthorized")


@app.post("/admin/kill/{key_id}")
async def admin_kill(key_id: str, request: Request) -> dict:
    _check_admin_auth(request, request.app.state.settings)
    await request.app.state.spend.kill(key_id)
    return {"key_id": key_id, "killed": True}


@app.delete("/admin/kill/{key_id}")
async def admin_clear_kill(key_id: str, request: Request) -> dict:
    _check_admin_auth(request, request.app.state.settings)
    await request.app.state.spend.clear_kill(key_id)
    return {"key_id": key_id, "killed": False}
