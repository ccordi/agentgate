"""HTTP face of the egress PDP: `POST /a/egress/decision`.

Advisory endpoint, not on the model path. The PEP (`egress/pep.py`, offered to agents as an
MCP tool by `egress/mcp_server.py`) POSTs a tool call here before executing it; this returns
allow/deny.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import uuid

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from agentgate.audit.store import AuditStore, RequestAudit, utcnow
from agentgate.egress import policy
from agentgate.limits.spend import key_id_from_auth
from agentgate.observability import metrics
from agentgate.tasks import spawn_background

log = logging.getLogger("agentgate.egress")

router = APIRouter()


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
    """Mandatory bearer auth for the egress PDP: `Authorization: Bearer <AGENTGATE_PDP_TOKEN>`.
    Startup refuses to serve without the token (validate_runtime_settings); the unset
    branch here fails closed (503) as defense in depth for app objects assembled without
    lifespan.

    The auth map, stated plainly: the admin routes require AGENTGATE_ADMIN_TOKEN; this
    endpoint requires its own dedicated AGENTGATE_PDP_TOKEN (NOT the local upstream's
    `local_api_key` — one value must not span two trust boundaries); by default the proxy
    routes need no gateway credential (issued keys are opt-in) — a transparent proxy does
    not own the Authorization slot, so their inbound auth is the loopback bind itself,
    enforced at startup.

    What this token buys: not defense against the agent (the PEP holds it by
    construction — `egress/mcp_server.py`) but against everything else that can reach
    loopback: browser-borne requests (CSRF / DNS rebinding) probing the PDP as a policy
    oracle — "would you allow this payload to that host?" maps the allowlist and the
    classifier's edges without tripping a denial on the real path."""
    expected = settings.pdp_token
    if not expected:
        raise HTTPException(status_code=503, detail="PDP token not configured")
    auth = request.headers.get("authorization", "")
    # Constant-time compare: this guards a privilege boundary (the egress PDP), so the
    # equality check must not leak the expected token byte-by-byte via timing.
    # Compared as bytes — see the note on app._check_admin_auth: compare_digest raises
    # TypeError on a non-ASCII str, which would turn a 401 into a 500.
    if not hmac.compare_digest(auth.encode("latin-1", "replace"),
                               f"Bearer {expected}".encode()):
        raise HTTPException(status_code=401, detail="unauthorized")


@router.post("/a/egress/decision", response_model=EgressDecisionResponse)
async def egress_decision(
    payload: EgressDecisionRequest, request: Request
) -> EgressDecisionResponse:
    settings = request.app.state.settings
    audit_store: AuditStore = request.app.state.audit

    _check_egress_auth(request, settings)

    # Off the event loop. The payload is classified over a window of up to 1 MB (its cost
    # is noted at `policy._MAX_PAYLOAD_CHARS`); inline, that time would stall every
    # request this process is serving. Same reason `guards` hands its CPU-bound work to a
    # thread.
    verdict = await asyncio.to_thread(
        policy.evaluate,
        tool_name=payload.tool_name,
        arguments=payload.arguments,
        tool_kind=payload.tool_kind,
        allowlist=settings.egress.allowlist,
        private_repo_markers=settings.private_repo_markers,
        # The loopback allowance must not cover the gateway's own control plane.
        gateway_origin=(settings.host, settings.port),
    )

    audit_id = uuid.uuid4()
    if verdict.caveats:
        # One caveat kind exists on this path (the payload window); counted and logged
        # here rather than in the pure policy function, keyed by the audit id.
        metrics.scan_truncated_total.labels("egress_payload").inc()
        log.info("egress payload classification truncated at %d chars: audit_id=%s "
                 "decision=%s", policy._MAX_PAYLOAD_CHARS, audit_id, verdict.decision)
    context = payload.context or {}
    sensitivity_class = str(verdict.sensitivity) if verdict.sensitivity is not None else None
    spawn_background(audit_store.write(RequestAudit(
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
        caveats=verdict.caveats or None,
    )))

    conditions = EgressConditions(**verdict.conditions) if verdict.conditions else None
    return EgressDecisionResponse(
        decision=verdict.decision,
        reason=verdict.reason,
        policy=verdict.policy,
        conditions=conditions,
        audit_id=str(audit_id),
    )
