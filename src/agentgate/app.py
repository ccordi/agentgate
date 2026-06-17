"""FastAPI application — the gateway's inbound face.

Pipeline (synchronous, pre-forward):
  1. parse request, extract untrusted content (tool output + newest user turn)
  2. injection scan -> block on hard-positive (cheap; content is fully buffered)
  3. spend cap + kill-switch check -> 429 if over cap / killed
  4. forward to the chosen upstream (Gemini path rewrite + auth pass-through)
  5. stream SSE straight back; tap usage
Post-stream (off hot path): record spend, write metadata audit, update metrics.

Classify / route / redact happen between steps 3 and 4.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import random
import re
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from agentgate.audit.store import AuditStore, ContentSampleAudit, RequestAudit, utcnow
from agentgate.config import get_settings
from agentgate.limits.backend import make_backend
from agentgate.limits.spend import SpendConfig, SpendExceeded, SpendTracker, key_id_from_auth
from agentgate.observability import metrics
from agentgate.observability.otel import setup_tracing
from agentgate.pricing import estimate_cost_usd
from agentgate.proxy.forwarder import forward_stream
from agentgate.routing import providers as routing_providers
from agentgate.routing import router as routing_router
from agentgate.security import capture, classifier, egress_policy, injection, llm_guard, model_guard
from agentgate.security.classifier import Sensitivity
from agentgate.security.content_crypto import ContentCipher
from agentgate.security.injection import _coerce_content
from agentgate.security.redaction import redact as _redact_text
from agentgate.security.skill_inspector import SkillVerdict, inspect_tools

log = logging.getLogger("agentgate")

# Regex to match client-injected Gemini thinking/final wrapping instructions that
# conflict with the local model's own chat template. Uses lazy matching to tolerate
# small edits across client prompt template versions.
_SYSTEM_PROMPT_CLEAN_RE = re.compile(
    r"ALL internal reasoning MUST be inside <think>.*?</think>\..*?"
    r"Format every reply as <think>.*?</think> then <final>.*?</final>.*?"
    r"Example: <think>.*?</think>\s*<final>.*?</final>",
    re.DOTALL
)

_SYSTEM_PROMPT_REPLACEMENT = (
    "For final user-visible answers, wrap them in <final>...</final>. "
    "For tool calls, use the native tool calling schema."
)

_background: set[asyncio.Task] = set()


def _spawn(coro) -> None:
    task = asyncio.create_task(coro)
    _background.add(task)
    task.add_done_callback(_background.discard)


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
    # Per-request dispatch (_scan_request) consults this flag and falls back FOR THAT
    # REQUEST ONLY, so one key's deberta-unavailable fallback never clobbers another
    # key's resolved backend (the old global-mutation approach did, and that's the bug
    # this refactors away).
    needs_deberta = settings.guard_backend in ("deberta", "combined") or any(
        b in ("deberta", "combined") for b in settings.guard_backend_overrides.values()
    )
    deberta_available = False
    if needs_deberta:
        try:
            log.info("loading deberta injection guard…")
            await asyncio.to_thread(model_guard.warmup)
            deberta_available = True
        except Exception as exc:
            # Model/runtime unavailable → degrade rather than hard-fail (a dead gateway
            # blocks all agent traffic). _scan_request falls back per-request when this is False.
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


app = FastAPI(title="agentgate", version="0.0.1", lifespan=lifespan)
setup_tracing(app)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
async def prometheus_metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/admin/kill/{key_id}")
async def admin_kill(key_id: str, request: Request) -> dict:
    await request.app.state.spend.kill(key_id)
    return {"key_id": key_id, "killed": True}


@app.delete("/admin/kill/{key_id}")
async def admin_clear_kill(key_id: str, request: Request) -> dict:
    await request.app.state.spend.clear_kill(key_id)
    return {"key_id": key_id, "killed": False}


# --- Tier-3 egress PDP ---------------------------------------------------------------
#
# Advisory endpoint, not on the model path. The harness's PreToolUse hook (the PEP,
# implemented in pep/mcp_server.py) POSTs a tool call here before executing it; this
# returns allow/deny.

class EgressDecisionRequest(BaseModel):
    tool_name: str
    tool_kind: str | None = None
    arguments: dict = Field(default_factory=dict)
    context: dict | None = None


class EgressConditions(BaseModel):
    redactions: list[dict] = Field(default_factory=list)


class EgressDecisionResponse(BaseModel):
    decision: str  # "allow" | "deny" | "allow_with_conditions" (latter reserved, unimplemented)
    reason: str
    policy: str
    conditions: EgressConditions | None = None
    audit_id: str


def _check_egress_auth(request: Request, settings) -> None:
    """Same loopback bearer auth as the proxy: if AGENTGATE_LOCAL_API_KEY is configured,
    require a matching `Authorization: Bearer <key>` header. If unset (default loopback
    deployment, no key configured), the check is a no-op — matching the proxy's posture."""
    expected = settings.local_api_key
    if not expected:
        return
    auth = request.headers.get("authorization", "")
    # Constant-time compare: this guards a privilege boundary (the egress PDP), so the
    # equality check must not leak the expected token byte-by-byte via timing.
    if not hmac.compare_digest(auth, f"Bearer {expected}"):
        raise HTTPException(status_code=401, detail="unauthorized")


@app.post("/a/egress/decision", response_model=EgressDecisionResponse)
async def egress_decision(
    payload: EgressDecisionRequest, request: Request
) -> EgressDecisionResponse:
    settings = request.app.state.settings
    audit_store: AuditStore = request.app.state.audit

    _check_egress_auth(request, settings)

    verdict = egress_policy.evaluate(
        tool_name=payload.tool_name,
        arguments=payload.arguments,
        tool_kind=payload.tool_kind,
        allowlist=settings.egress.allowlist,
        private_repo_markers=settings.private_repo_markers,
    )

    audit_id = uuid.uuid4()
    context = payload.context or {}
    sensitivity_class = str(verdict.sensitivity) if verdict.sensitivity is not None else None
    _spawn(audit_store.write(RequestAudit(
        id=audit_id,
        ts=utcnow(),
        agent_id=context.get("agent_id"),
        key_id=key_id_from_auth(request.headers.get("authorization"), None),
        model_requested=payload.tool_name,
        route_provider="egress",
        route_is_local=False,
        upstream_model=None,
        sensitivity_class=sensitivity_class,
        tokens_prompt=0, tokens_completion=0, cost_usd=0.0,
        latency_total_ms=None, latency_upstream_ms=None,
        injection_flagged=False, injection_score=None,
        redaction_hit_count=len(verdict.hit_types),
        # Match the proxy path's column shape ([{"type","count"}, ...]) — the egress
        # verdict carries bare type names, so the per-type count is 1.
        redaction_hit_types=[{"type": t, "count": 1} for t in verdict.hit_types] or None,
        tool_call_count=1,
        finish_reason=verdict.decision,
        status=200 if verdict.decision != "deny" else 403,
    )))

    conditions = EgressConditions(**verdict.conditions) if verdict.conditions else None
    return EgressDecisionResponse(
        decision=verdict.decision,
        reason=verdict.reason,
        policy=verdict.policy,
        conditions=conditions,
        audit_id=str(audit_id),
    )


def _parse_body(body: bytes) -> tuple[str | None, list[dict]]:
    try:
        data = json.loads(body)
        return data.get("model"), data.get("messages") or []
    except (ValueError, AttributeError):
        return None, []


def _error(status: int, message: str, type_: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"message": message, "type": type_}})


async def _capture_content(
    store: AuditStore,
    cipher: ContentCipher,
    messages: list[dict],
    request_id: uuid.UUID,
    reason: str,
    retention_days: int,
) -> None:
    """Redact and encrypt the untrusted messages, then write content sample rows.

    Runs off the hot path via _spawn.  Errors are swallowed — capture must never
    break forwarding.  We re-run redact() here (messages still carry pre-redaction
    text) so what we store is always redacted-then-encrypted.
    """
    from datetime import timedelta

    now = utcnow()
    expires_at = now + timedelta(days=retention_days)

    # Capture newest user turn + last tool message (the untrusted surface).
    candidates: list[dict] = []
    last_tool = next((m for m in reversed(messages) if m.get("role") == "tool"), None)
    last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    if last_tool is not None:
        candidates.append(last_tool)
    if last_user is not None:
        candidates.append(last_user)

    for msg in candidates:
        text = _coerce_content(msg.get("content"))
        if not text:
            continue
        redacted = _redact_text(text).redacted_text
        try:
            enc = cipher.encrypt(redacted)
        except Exception:  # noqa: BLE001
            log.exception("content cipher error — skipping sample")
            continue
        await store.write_content_sample(ContentSampleAudit(
            request_id=request_id,
            ts=now,
            role=msg.get("role", "unknown"),
            redacted_content_enc=enc,
            sampled_reason=reason,
            expires_at=expires_at,
        ))


def _redact_into(text: str, acc: dict[str, int]) -> str | None:
    """Redact one text blob, folding per-type hit counts into ``acc``.

    Returns the redacted text if anything was found, else None (caller leaves the
    original in place). Aggregating into a shared dict keeps hit_types deduped by
    type across all messages instead of one entry per message.
    """
    result = _redact_text(text)
    if not result.found:
        return None
    for entry in result.hit_types:
        acc[entry["type"]] = acc.get(entry["type"], 0) + entry["count"]
    return result.redacted_text


def _resolve_guard_backend(settings, key_id: str | None, deberta_available: bool) -> str:
    """Resolve the effective backend for ``key_id`` (override, else default), applying
    the per-request deberta-availability fallback. Shared by _scan_request (dispatch)
    and _handle_chat (audit) so the audited backend always matches what actually ran."""
    backend = settings.guard_backend_overrides.get(key_id, settings.guard_backend)
    if not deberta_available:
        if backend == "combined":
            backend = "llm"
        elif backend == "deberta":
            backend = "heuristic"
    return backend


async def _scan_request(settings, messages: list[dict], key_id: str | None = None,
                         deberta_available: bool = True):
    """Inbound injection scan via the per-key-resolved backend. The model backend runs
    in a worker thread so its ~10-30ms inference never blocks the event loop; the
    heuristic is inline.

    Backend resolution:
      1. ``settings.guard_backend_overrides.get(key_id, settings.guard_backend)`` picks
         the backend for this key (default for unmapped keys).
      2. If the resolved backend needs deberta and ``deberta_available`` is False, fall
         back FOR THIS REQUEST ONLY — "combined" degrades to "llm", "deberta" degrades
         to "heuristic" — without touching settings or any other key's resolution.
    """
    backend = _resolve_guard_backend(settings, key_id, deberta_available)

    if backend == "deberta":
        return await asyncio.to_thread(model_guard.scan_request, messages)
    elif backend == "llm":
        return await asyncio.to_thread(llm_guard.scan_request, messages)
    elif backend == "combined":
        deb_verdict, llm_verdict = await asyncio.gather(
            asyncio.to_thread(model_guard.scan_request, messages),
            asyncio.to_thread(llm_guard.scan_request, messages),
        )
        return injection.Verdict(
            flagged=deb_verdict.flagged or llm_verdict.flagged,
            score=max(deb_verdict.score, llm_verdict.score),
            reasons=deb_verdict.reasons + llm_verdict.reasons,
            hard=deb_verdict.hard or llm_verdict.hard,
        )
    return injection.scan_request(messages)


async def _handle_chat(request: Request, agent_id: str | None) -> Response:
    settings = request.app.state.settings
    audit_store: AuditStore = request.app.state.audit
    spend: SpendTracker = request.app.state.spend
    client: httpx.AsyncClient = request.app.state.http
    cipher: ContentCipher = request.app.state.cipher

    # Generate request id up front so content samples can reference it before the
    # metadata row is written.
    request_id = uuid.uuid4()

    t0 = time.perf_counter()
    body = await request.body()
    model_requested, messages = _parse_body(body)
    headers = dict(request.headers)
    key_id = key_id_from_auth(headers.get("authorization"), headers.get("x-goog-api-key"))

    # --- Sensitivity classification (cheap, inline, no egress) → routing + audit policy ---
    sensitivity = classifier.classify_request(messages, settings.private_repo_markers).sensitivity
    sensitivity_class = str(sensitivity)
    if settings.routing.enabled:
        decision = routing_router.decide(
            routing_router.RouteContext(sensitivity=sensitivity_class, agent_id=agent_id),
            settings.routing,
        )
        provider = routing_providers.resolve(settings, decision)
    else:
        provider = settings.provider()

    # Single parse of the body for model-rewrite, cloud-egress redaction, and tool
    # extraction (re-dumped once, only if mutated). Redaction is cloud-only: local routes
    # handle sensitive content with zero egress, so there's nothing to scrub outbound.
    redact_ms = 0.0
    redact_hit_count = 0
    redact_hit_types: list | None = None
    raw_tools: list = []

    try:
        payload = json.loads(body)
    except (ValueError, AttributeError):
        payload = None  # malformed body — let the upstream reject it

    if isinstance(payload, dict):
        mutated = False
        if provider.model_name is not None:
            payload["model"] = provider.model_name
            mutated = True
        # Local-route request overrides (env-driven, default-off — see config.py). Lets a
        # scoped debug session tune the local upstream's model/stop/cap/reasoning without a
        # code change. Applied after the model rewrite so the override wins.
        if provider.is_local:
            if settings.local_model_override:
                payload["model"] = settings.local_model_override
                mutated = True
            if settings.local_stop:
                payload["stop"] = [s for s in settings.local_stop.split(",") if s]
                mutated = True
            if settings.local_max_tokens is not None:
                payload["max_completion_tokens"] = settings.local_max_tokens
                payload["max_tokens"] = settings.local_max_tokens
                mutated = True
            if settings.local_enable_thinking is not None:
                ctk = payload.get("chat_template_kwargs")
                ctk = ctk if isinstance(ctk, dict) else {}
                ctk["enable_thinking"] = settings.local_enable_thinking
                payload["chat_template_kwargs"] = ctk
                mutated = True
            
            # Clean system prompt of conflicting <final> wrapping instructions for local routes
            if "messages" in payload and isinstance(payload["messages"], list):
                for msg in payload["messages"]:
                    if isinstance(msg, dict) and msg.get("role") in ("system", "developer"):
                        content = msg.get("content")
                        if isinstance(content, str):
                            has_instr = (
                                "ALL internal reasoning MUST be inside" in content
                                or ("<think>" in content and "<final>" in content)
                            )
                            if has_instr:
                                new_content, count = _SYSTEM_PROMPT_CLEAN_RE.subn(
                                    _SYSTEM_PROMPT_REPLACEMENT, content
                                )
                                if count > 0:
                                    msg["content"] = new_content
                                    mutated = True
                                else:
                                    log.warning(
                                        "Detected <think>/<final> prompt instructions, "
                                        "but regex adapter failed to match."
                                    )
                        elif isinstance(content, list):
                            for part in content:
                                if (
                                    isinstance(part, dict)
                                    and part.get("type") == "text"
                                    and isinstance(part.get("text"), str)
                                ):
                                    text = part["text"]
                                    has_instr = (
                                        "ALL internal reasoning MUST be inside" in text
                                        or ("<think>" in text and "<final>" in text)
                                    )
                                    if has_instr:
                                        new_text, count = _SYSTEM_PROMPT_CLEAN_RE.subn(
                                            _SYSTEM_PROMPT_REPLACEMENT, text
                                        )
                                        if count > 0:
                                            part["text"] = new_text
                                            mutated = True
                                        else:
                                            log.warning(
                                                "Detected <think>/<final> list instructions, "
                                                "but regex adapter failed to match."
                                            )
        if not provider.is_local and settings.redaction_enabled:
            t_redact = time.perf_counter()
            acc: dict[str, int] = {}
            for msg in payload.get("messages") or []:
                raw = msg.get("content")
                if not raw:
                    continue
                if isinstance(raw, str):
                    red = _redact_into(raw, acc)
                    if red is not None:
                        msg["content"] = red
                        mutated = True
                elif isinstance(raw, list):
                    for part in raw:
                        if isinstance(part, dict) and isinstance(part.get("text"), str):
                            red = _redact_into(part["text"], acc)
                            if red is not None:
                                part["text"] = red
                                mutated = True
            redact_hit_count = sum(acc.values())
            redact_hit_types = [{"type": t, "count": c} for t, c in acc.items()] or None
            redact_ms = (time.perf_counter() - t_redact) * 1000
        raw_tools = payload.get("tools") or []
        if mutated:
            body = json.dumps(payload).encode()

    # --- Skill inspection: static analysis of tools[] definitions ---
    # Hard tier: block on description-level instruction injection (low FP).
    # Soft tier: record-only (coarse heuristics, high FP — observe first).
    skill_verdict = inspect_tools(raw_tools)
    if skill_verdict.flagged:
        log.warning("skill_inspector flagged: key=%s hard=%s reasons=%s",
                    key_id, skill_verdict.hard, skill_verdict.reasons)
    if skill_verdict.hard:
        _spawn(_audit_rejected(
            audit_store, model_requested, provider,
            verdict=_null_inject_verdict(), key_id=key_id,
            status=400, inject_ms=0.0, t0=t0, agent_id=agent_id,
            sensitivity_class=sensitivity_class,
            skill_verdict=skill_verdict,
        ))
        return _error(
            400,
            f"blocked: malicious tool definition ({', '.join(skill_verdict.reasons)})",
            "skill_blocked",
        )

    # --- 0. Passive traffic capture (off by default; only the tagged capture agent) ---
    # Fire-and-forget, before the block decision so tripped real-world injections — the
    # high-value samples — are captured even when the request is rejected inbound.
    if settings.capture_enabled and agent_id == settings.capture_agent_id:
        _spawn(capture.capture(settings.capture_path, agent_id, messages))

    # --- 1+2. Injection scan of untrusted content; block on hard-positive ---
    t_inject = time.perf_counter()
    deberta_available = getattr(request.app.state, "deberta_available", True)
    effective_guard_backend = _resolve_guard_backend(settings, key_id, deberta_available)
    verdict = await _scan_request(settings, messages, key_id=key_id,
                                   deberta_available=deberta_available)
    inject_ms = (time.perf_counter() - t_inject) * 1000
    if verdict.flagged:
        log.warning("injection flagged: key=%s score=%.2f reasons=%s hard=%s",
                    key_id, verdict.score, verdict.reasons, verdict.hard)
    if verdict.hard and not settings.guard_observe_mode:
        metrics.injection_flagged_total.labels("True").inc()
        _spawn(_audit_rejected(audit_store, model_requested, provider, verdict, key_id,
                               status=400, inject_ms=inject_ms, t0=t0, agent_id=agent_id,
                               sensitivity_class=sensitivity_class,
                               guard_backend=effective_guard_backend))
        return _error(400, f"blocked: prompt injection detected ({', '.join(verdict.reasons)})",
                      "injection_blocked")
    if verdict.hard:
        # Observe mode: would-block, but forward anyway. Logged + audited (injection_hard=True
        # in _finalize) so live FP candidates stay countable without breaking the agent loop.
        log.warning("injection OBSERVE (would-block, forwarding): key=%s score=%.2f reasons=%s",
                    key_id, verdict.score, verdict.reasons)
        metrics.injection_flagged_total.labels("True").inc()
    elif verdict.flagged:
        metrics.injection_flagged_total.labels("False").inc()

    # --- 3. Spend cap + kill switch ---
    try:
        await spend.check(key_id, provider.is_local)
    except SpendExceeded as exc:
        _spawn(_audit_rejected(audit_store, model_requested, provider, verdict, key_id,
                               status=429, inject_ms=inject_ms, t0=t0, agent_id=agent_id,
                               sensitivity_class=sensitivity_class,
                               guard_backend=effective_guard_backend))
        return _error(429, exc.reason, "spend_exceeded")

    # --- 4. Forward ---
    cm = forward_stream(client, provider, inbound_headers=headers, body=body,
                        timeout_s=settings.upstream_timeout_s)
    try:
        t_upstream = time.perf_counter()
        upstream = await cm.__aenter__()
    except httpx.HTTPError as exc:
        log.warning("upstream connection error: %s", exc)
        metrics.requests_total.labels(provider.name, str(provider.is_local), "502").inc()
        return _error(502, f"upstream error: {exc}", "upstream_error")

    async def stream_and_finalize():
        try:
            async for chunk in upstream.body:
                yield chunk
        finally:
            await cm.__aexit__(None, None, None)
            _finalize()

    def _finalize() -> None:
        now = time.perf_counter()
        latency_total_ms = (now - t0) * 1000
        latency_upstream_ms = (now - t_upstream) * 1000
        r = upstream.result
        cost = estimate_cost_usd(r.upstream_model, r.prompt_tokens, r.completion_tokens)
        finish = r.finish_reasons[-1] if r.finish_reasons else None

        metrics.requests_total.labels(
            provider.name, str(provider.is_local), str(upstream.status_code)).inc()
        metrics.request_latency_seconds.labels(provider.name).observe(latency_total_ms / 1000)
        metrics.tokens_total.labels(provider.name, "prompt").inc(r.prompt_tokens)
        metrics.tokens_total.labels(provider.name, "completion").inc(r.completion_tokens)
        if cost:
            metrics.cost_usd_total.labels(provider.name).inc(cost)

        log.info(
            "completed: key=%s provider=%s model=%s tokens=%s/%s cost=$%.5f tool_calls=%s "
            "finish=%s inject=%.2f %.1fms",
            key_id, provider.name, r.upstream_model, r.prompt_tokens, r.completion_tokens,
            cost, r.tool_call_count, finish, verdict.score, latency_total_ms,
        )

        _spawn(spend.record(key_id, provider.is_local, cost))
        _spawn(audit_store.write(RequestAudit(
            id=request_id,
            ts=utcnow(), agent_id=agent_id, key_id=key_id, model_requested=model_requested,
            route_provider=provider.name, route_is_local=provider.is_local,
            upstream_model=r.upstream_model, sensitivity_class=sensitivity_class,
            tokens_prompt=r.prompt_tokens, tokens_completion=r.completion_tokens, cost_usd=cost,
            latency_total_ms=latency_total_ms, latency_upstream_ms=latency_upstream_ms,
            latency_inject_ms=inject_ms,
            injection_flagged=verdict.flagged, injection_hard=verdict.hard,
            injection_score=verdict.score,
            redaction_hit_count=redact_hit_count, redaction_hit_types=redact_hit_types,
            latency_redact_ms=redact_ms if redact_ms else None,
            skill_flagged=skill_verdict.flagged, skill_hard=skill_verdict.hard,
            skill_reasons=skill_verdict.reasons if skill_verdict.flagged else None,
            tool_call_count=r.tool_call_count, finish_reason=finish, status=upstream.status_code,
            guard_backend=effective_guard_backend,
        )))
        # --- Content capture (off hot path) ---
        # Never for local routes (sensitive content stays on-box) or sensitive content.
        # Capture flagged requests always; benign cloud at the configured sample rate.
        is_sensitive = sensitivity_class != str(Sensitivity.NONE)
        if (
            not provider.is_local
            and not is_sensitive
            and cipher.enabled
            and settings.content_capture_enabled
        ):
            sample_reason: str | None = None
            if verdict.flagged:
                sample_reason = "flagged"
            elif random.random() < settings.content_sample_rate:
                sample_reason = "random"
            if sample_reason is not None:
                _spawn(_capture_content(
                    audit_store, cipher, messages, request_id,
                    sample_reason, settings.content_retention_days,
                ))

    return StreamingResponse(
        stream_and_finalize(), status_code=upstream.status_code,
        headers=upstream.headers, media_type="text/event-stream",
    )


# Accept both with and without the /v1 prefix: OpenAI-compatible clients may or may not
# include /v1 in their baseUrl. Tolerate both. Untagged = no agent attribution.
@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request) -> Response:
    return await _handle_chat(request, agent_id=None)


# Agent-tagged variant: an agent's baseUrl is set to ".../a/<agent_id>", so its requests
# arrive here and carry attribution (drives traffic capture + the router's agent-pin rules).
# The {agent_id} segment is the only way to identify the agent — the client sends vanilla
# OpenAI Chat Completions with no agent field in the payload.
@app.post("/a/{agent_id}/v1/chat/completions")
@app.post("/a/{agent_id}/chat/completions")
async def chat_completions_for_agent(request: Request, agent_id: str) -> Response:
    return await _handle_chat(request, agent_id=agent_id)


def _null_inject_verdict():
    """A no-op injection verdict for paths that were rejected before the injection scan."""
    return injection.Verdict.clean()


async def _audit_rejected(
    store, model_requested, provider, verdict, key_id, status,
    inject_ms, t0, agent_id=None, sensitivity_class=None,
    skill_verdict: SkillVerdict | None = None, guard_backend: str | None = None,
):
    """Write an audit row for a request rejected before forwarding."""
    sv = skill_verdict or SkillVerdict(flagged=False)
    await store.write(RequestAudit(
        ts=utcnow(), agent_id=agent_id, key_id=key_id, model_requested=model_requested,
        route_provider=provider.name, route_is_local=provider.is_local, upstream_model=None,
        sensitivity_class=sensitivity_class, tokens_prompt=0, tokens_completion=0, cost_usd=0.0,
        latency_total_ms=(time.perf_counter() - t0) * 1000, latency_upstream_ms=None,
        latency_inject_ms=inject_ms, latency_redact_ms=None,
        injection_flagged=verdict.flagged, injection_hard=verdict.hard,
        injection_score=verdict.score,
        redaction_hit_count=0, redaction_hit_types=None,
        skill_flagged=sv.flagged, skill_hard=sv.hard,
        skill_reasons=sv.reasons if sv.flagged else None,
        tool_call_count=0, finish_reason=None, status=status,
        guard_backend=guard_backend,
    ))
