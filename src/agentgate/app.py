"""FastAPI application assembly — the gateway's inbound face.

This module owns process lifecycle and the routes that aren't traffic: startup/shutdown
(`lifespan`), health, metrics, the kill switch, and the two router includes. It holds no
request-handling logic of its own.

The two data planes it assembles:
  - `pipeline` — the model wire (`/v1/chat/completions` and friends). Scan, route,
    redact, forward, audit.
  - `egress.api` — the off-wire control plane (`/a/egress/decision`), the egress PDP
    an agent application consults before performing a tool call.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from agentgate import guards, keys, pipeline
from agentgate.audit.crypto import ContentCipher
from agentgate.audit.store import AuditStore
from agentgate.config import get_settings, validate_runtime_settings
from agentgate.egress import api as egress_api
from agentgate.egress import policy as egress_policy
from agentgate.limits.admission import AdmissionController
from agentgate.limits.backend import make_backend
from agentgate.limits.spend import SpendTracker
from agentgate.observability.otel import setup_tracing
from agentgate.tasks import drain_background

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
    # which never consults settings; the threat model documents that gap.
    validate_runtime_settings(settings)
    if settings.scan_executor_threads:
        # Deliberate compute budget for the shared to_thread pool (guard scans dominate
        # it) instead of asyncio's cores+4 default — see config.scan_executor_threads.
        asyncio.get_running_loop().set_default_executor(
            ThreadPoolExecutor(max_workers=settings.scan_executor_threads,
                               thread_name_prefix="agentgate-scan"))
    # Load deberta if it is reachable from *any* backend this process can resolve to — the
    # global default or a per-key override, `combined` included — and REFUSE TO START if it
    # will not load. A blocking safety control that cannot run must not be answered by
    # quietly running a weaker one: degrading `deberta`→`heuristic` and `combined`→`llm`
    # per request would give a deployment that asked for the classifier the regex
    # baseline, with every audit row reading like a scan that happened. An outage the
    # operator sees beats a downgrade nobody does.
    #
    # Placed before the client, the audit store and the sweeper are assembled, so a refusal
    # leaves nothing open — the same reasoning as the posture check above.
    needs_deberta = guards.requires(settings, "deberta")
    deberta_available = False
    if needs_deberta:
        try:
            log.info("loading deberta injection guard…")
            from agentgate.guards import deberta
            await asyncio.to_thread(deberta.warmup)
            deberta_available = True
        except Exception as exc:
            log.critical("DEBERTA GUARD UNAVAILABLE (%s) — refusing to start", exc)
            raise RuntimeError(
                f"the deberta injection guard could not load ({exc}), and a backend this "
                f"process can resolve to requires it (guard_backend={settings.guard_backend!r},"
                f" {len(settings.guard_backend_overrides)} per-key override(s)). Install the "
                "`guard` extra and point AGENTGATE_GUARD_MODEL_DIR at the classifier's model "
                "files, or configure a backend that does not need it — the gateway refuses to "
                "start rather than scan with a weaker one."
            ) from exc
    app.state.deberta_available = deberta_available
    # Explicit per-operation budget. A bare scalar expands to connect/read/write/pool of
    # the SAME value, and `read` is per-read rather than per-request (see
    # config.upstream_timeout_s). Connect is tightened to something a loopback/cloud TCP
    # handshake can actually meet; the total wall-clock budget is
    # settings.upstream_total_deadline_s, enforced separately in `pipeline`.
    app.state.http = httpx.AsyncClient(
        timeout=httpx.Timeout(
            connect=10.0,
            read=settings.upstream_timeout_s,
            write=settings.upstream_timeout_s,
            pool=settings.upstream_timeout_s,
        ),
        # Explicit pool caps (0 → unlimited); defaults equal httpx's own, so this only
        # changes behavior when a load run raises them deliberately.
        limits=httpx.Limits(
            max_connections=settings.upstream_max_connections or None,
            max_keepalive_connections=settings.upstream_max_keepalive or None,
        ),
    )
    app.state.settings = settings
    app.state.audit = AuditStore(settings.database_url)
    await app.state.audit.init()
    app.state.spend = SpendTracker(
        await make_backend(settings.redis_url, require=settings.require_shared_limits),
        settings.spend_config(),
    )
    app.state.cipher = ContentCipher(settings.content_enc_key)
    if not app.state.cipher.enabled:
        log.info("content capture disabled (AGENTGATE_CONTENT_ENC_KEY not set)")
    sweeper = asyncio.create_task(_sweeper_loop(app.state.audit))
    # Startup probe for the LLM judge: a missing AGENTGATE_JUDGE_* setting or a non-local
    # base_url must surface at boot, not on the first user request (lru_cache does not
    # memoize exceptions, so a failing construction would re-raise forever with no
    # boot-time signal). Probe is config-only (no model call): reachability is a runtime
    # property, handled by run_injection_scan's 503. Non-fatal — unlike the deberta load
    # above, a judge that is merely misconfigured still answers every `llm` request with a
    # 503 `guard_unavailable` rather than a weaker scan, so the refusal is per request.
    if guards.requires(settings, "llm"):
        try:
            from agentgate.guards import local_llm
            local_llm.warmup()
            log.info("LLM judge configured (model=%s)", local_llm.JudgeConfig().model)
        except Exception as exc:
            log.error("LLM JUDGE MISCONFIGURED (%s) — every request that uses the LLM judge "
                      "will fail with 503 guard_unavailable", exc)
    # Zero active keys with the gate on serves rather than refuses — minting happens
    # through the admin plane after boot — but it is a state worth announcing loudly.
    if settings.require_issued_keys:
        active = await keys.key_store(app).count_active()
        log.info("issued-key auth armed: %d active key(s)", active)
        if not active:
            log.warning("issued-key auth armed with ZERO active keys — every proxy "
                        "request will 401 until POST /admin/keys mints one")
    # Built here rather than on first request so an armed concurrency control is visible
    # at boot. Left unset when disabled — nothing in the request path looks for it.
    if settings.admission_enabled:
        app.state.admission = AdmissionController.from_settings(settings)
        log.info("admission control armed: per_key=%d global=%d queue_depth=%d deadline=%.1fs",
                 settings.admission_per_key, settings.admission_global,
                 settings.admission_queue_depth, settings.admission_wait_deadline_s)
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
        # Drain in-flight audit and content-sample writes before closing the store.
        await drain_background()
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
    parsing edge; this check covers the proxy routes, which need no gateway credential by
    default (issued keys are opt-in).
    What neither closes — blind cross-origin POSTs burning local compute — is
    documented in the threat model. Loopback = egress_policy._LOOPBACK_HOSTS, the project's
    one definition.
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
        if not egress_policy.is_loopback_host(host):
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
app.include_router(pipeline.router)
app.include_router(egress_api.router)
setup_tracing(app)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness only: the process is up and serving. Deliberately probes nothing.

    Kept dumb on purpose — a liveness probe that can fail on a dependency will restart a
    process that was fine. For "is this gateway actually able to do its job", see /readyz.
    """
    return {"status": "ok"}


@app.get("/readyz")
async def readyz(request: Request) -> Response:
    """Readiness: the things liveness alone lets be silently wrong.

    A one-character typo in a routing target can produce a `Settings` that constructs
    cleanly, a process that starts, and a `/healthz` returning `{"status":"ok"}` — then a
    `KeyError` → framework 500 on *every* request. Every signal in that chain reports on
    something other than the gateway's ability to serve a request.

    `_validate_routing_targets` rejects that config at construction, so this endpoint is
    the second half: it reports the resolved state a running process is serving with — the
    spend backend it settled on, the provider names it will look up, the guard model it
    loaded.

    503 when a configured guard backend is not actually available. A process started
    through `lifespan` cannot be in that state: a backend whose model will not load
    refuses startup there. The branch stays because an app object assembled without
    lifespan has no such guarantee, and a readiness probe that can only ever answer
    "ready" reports nothing. A memory spend backend is reported, not failed on — it loses
    cross-process spend state, which is an ops problem, not a safety one.
    """
    settings = request.app.state.settings
    deberta_available = getattr(request.app.state, "deberta_available", False)

    needs_deberta = guards.requires(settings, "deberta")
    guard_ready = deberta_available if needs_deberta else True

    body = {
        "status": "ready" if guard_ready else "degraded",
        "guard_backend": settings.guard_backend,
        "guard_backend_overrides": len(settings.guard_backend_overrides),
        "deberta_required": needs_deberta,
        "deberta_available": deberta_available,
        "spend_backend": request.app.state.spend.backend_kind,
        # Resolved, not configured: these are the names `provider()` will look up at
        # request time, which is where a typo would otherwise surface.
        "providers": sorted(settings.providers),
        "default_provider": settings.default_provider,
        "routing_enabled": settings.routing.enabled,
    }
    if settings.routing.enabled:
        body["routing_targets"] = {
            "default_local": settings.routing.default_local,
            "default_cloud": settings.routing.default_cloud,
        }
    return JSONResponse(status_code=200 if guard_ready else 503, content=body)


@app.get("/metrics")
async def prometheus_metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# --- Admin plane -----------------------------------------------------------------------
#
# Without auth here, the egress PDP's always-allow-loopback rule would let the agent's
# ONLY sanctioned network path, on request and by policy, DELETE the kill switch that
# exists to halt it. Bearer auth is mandatory (startup refuses without
# AGENTGATE_ADMIN_TOKEN — validate_runtime_settings): a conditional gate would leave
# unarmed deployments looking closed while open. The companion half is carving the
# gateway's own origin out of the loopback allowance — see egress.policy._is_own_origin.

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


# Issued keys (the AGENTGATE_REQUIRE_ISSUED_KEYS gate's mint/revoke plane). Same
# mandatory bearer as the kill switch. The mint response is the only place the
# plaintext secret ever exists — the table stores hashes.

@app.post("/admin/keys")
async def admin_mint_key(request: Request) -> dict:
    _check_admin_auth(request, request.app.state.settings)
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 — an empty/absent body is a label-less mint
        body = {}
    label = body.get("label") if isinstance(body, dict) else None
    if label is not None and not isinstance(label, str):
        raise HTTPException(status_code=422, detail="label must be a string")
    minted = await keys.key_store(request.app).mint(label)
    return {"key": minted.secret, "key_id": minted.key_id, "label": label}


@app.get("/admin/keys")
async def admin_list_keys(request: Request) -> list[dict]:
    _check_admin_auth(request, request.app.state.settings)
    rows = await keys.key_store(request.app).list_keys()
    # key_id + metadata only: no hashes (offline-crackable against a leaked DB dump
    # they would confirm guesses), no secrets (not stored at all).
    return [
        {"key_id": r.key_id, "label": r.label,
         "created_at": r.created_at, "revoked_at": r.revoked_at}
        for r in rows
    ]


@app.delete("/admin/keys/{key_id}")
async def admin_revoke_key(key_id: str, request: Request) -> dict:
    _check_admin_auth(request, request.app.state.settings)
    if not await keys.key_store(request.app).revoke(key_id):
        raise HTTPException(status_code=404, detail="no issued key with that key_id")
    return {"key_id": key_id, "revoked": True}


def _check_admin_auth(request: Request, settings) -> None:
    """Mandatory bearer auth for the admin plane (`/admin/kill/*`, `/admin/keys*`).

    Accepts exactly ``AGENTGATE_ADMIN_TOKEN`` — nothing else. There is deliberately no
    ``local_api_key`` fallback: it would authenticate the one caller this control exists
    to exclude, because the PEP holds that key by construction (it reads the same `.env`
    the gateway does), so the fallback would close nothing while presenting as closed.
    Startup refuses to serve without the token (validate_runtime_settings); the unset
    branch here fails closed rather than open as defense in depth for app objects
    assembled without lifespan.

    Auth on loopback is not theater: loopback excludes remote sockets, not remote
    code — browser-borne requests reach 127.0.0.1 (blind CSRF; readable with DNS
    rebinding, which LoopbackHostGuard closes independently). Not a tenancy boundary:
    by default `key_id` is a hash of whatever the client sends; this endpoint exists to
    stop the governed agent from clearing its own halt through the sanctioned egress tool.
    """
    expected = settings.admin_token
    if not expected:
        raise HTTPException(status_code=503, detail="admin token not configured")
    auth = request.headers.get("authorization", "")
    # Compared as bytes: Starlette decodes headers as latin-1, and compare_digest raises
    # TypeError on a non-ASCII str — which would turn a 401 into an unhandled 500 on a
    # header a raw-socket caller controls. latin-1 round-trips back to the wire bytes.
    if not hmac.compare_digest(auth.encode("latin-1", "replace"),
                               f"Bearer {expected}".encode()):
        raise HTTPException(status_code=401, detail="unauthorized")
