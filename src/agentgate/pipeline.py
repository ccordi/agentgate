"""The chat data plane — everything that happens to a request on the model wire.

Stage order (synchronous, pre-forward), and where each stage's logic lives:

  1. reject an undecodable `content-encoding` (`screen_content_encoding`) → 400. First,
     ahead of classification: there is no readable body for any later stage to read
  2. classify sensitivity (`sensitivity`) → route (`routing`)
  3. prepare the outbound body: provider model rewrite, local-route adaptation
     (`local_route`), cloud-only redaction (`redaction`)
  4. tool-definition screening (`tool_inspector`) → 400 on a hard verdict
  5. passive capture tap (`capture`) — fire-and-forget; after the tool-definition
     screen (a hard tool-def block is not captured), before the injection-scan
     and spend decisions
  6. injection scan (`guards`) → 400 on a hard verdict, unless observe mode
  7. spend cap + kill switch (`limits`) → 429
  8. forward and stream back (`proxy`), tapping usage
Post-stream (off hot path): record spend, write the metadata audit row, update metrics,
sample content into the encrypted content tier.

The body is re-serialized once, and only if a stage mutated it.
"""

from __future__ import annotations

import json
import logging
import random
import time
import uuid
import zlib
from dataclasses import dataclass, field

import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from agentgate import capture, guards, local_route, routing
from agentgate import sensitivity as sensitivity_mod
from agentgate.audit.crypto import ContentCipher
from agentgate.audit.store import AuditStore, ContentSampleAudit, RequestAudit, utcnow
from agentgate.config import Provider
from agentgate.content import coerce_content
from agentgate.limits.spend import SpendExceeded, SpendTracker, key_id_from_auth
from agentgate.observability import metrics, otel
from agentgate.pricing import estimate_cost_usd
from agentgate.proxy.forwarder import forward_stream
from agentgate.redaction import redact as _redact_text
from agentgate.sensitivity import Sensitivity
from agentgate.tasks import spawn_background as _spawn
from agentgate.tool_inspector import ToolVerdict, inspect_tools

log = logging.getLogger("agentgate")

router = APIRouter()


@dataclass
class ChatCall:
    """One in-flight chat request, threaded through the stages below."""

    request_id: uuid.UUID
    t0: float
    body: bytes
    headers: dict[str, str]
    key_id: str  # always set: key_id_from_auth hashes the credential (or "anonymous")
    agent_id: str | None
    model_requested: str | None

    # `messages` and `payload` must remain independent parses of the same bytes.
    # `messages` is what the guards scan; `payload` is what gets redacted and forwarded.
    # Keeping them disjoint is what makes redaction unable to blind the guard, on both
    # routes. Do NOT "simplify" this to `messages = payload["messages"]`: that hands the
    # scanner `[REDACTED:openai_key] …` instead of the attacker's text on every cloud
    # request carrying a secret, silently. Pinned by
    # tests/test_app_pipeline.py::test_scanner_sees_unredacted_text_on_cloud_route.
    messages: list[dict]
    payload: dict | None  # None if the body is malformed; the upstream rejects it

    # What the guard actually scans. Normally `messages`; on a body the gateway could not
    # parse it is the raw bytes wrapped as one untrusted tool-output item, which is the
    # ONE consumer that differs — the classifier, capture and redaction all still see the
    # empty `messages`.
    scan_messages: list[dict] = field(default_factory=list)

    sensitivity_class: str = ""
    provider: Provider | None = None
    redact_ms: float = 0.0
    redact_hit_count: int = 0
    redact_hit_types: list | None = None
    # Tool definitions off the FIRST parse, so the tool screen inherits the same
    # guarantee `messages` has: redaction cannot reach what the screen reads. Redaction
    # touches only messages[] today, so this changes nothing now — it is what keeps
    # extending it to tool descriptions from silently handing the screen redacted text.
    raw_tools: list = field(default_factory=list)
    mutated: bool = False

    # The `content-encoding` we were handed and could not fully decode, or None. Set in
    # `new_call` and answered at the screen point — nothing downstream may forward a body
    # this is set on, because nothing was able to read it.
    encoding_error: str | None = None

    # Filled in by the screening stages, read by the audit row.
    tool_verdict: ToolVerdict = field(default_factory=lambda: ToolVerdict(flagged=False))
    verdict: guards.Verdict = field(default_factory=guards.Verdict.clean)
    inject_ms: float = 0.0
    guard_backend: str | None = None

    # Server-span context, captured while it is current (see otel.capture_parent). Opaque;
    # None whenever tracing is off.
    otel_parent: object | None = None


# --- helpers -------------------------------------------------------------------------

def _parse_body(body: bytes) -> tuple[str | None, list[dict], list, bool]:
    """Parse the inbound body into (model, messages, tools, parsed_ok).

    ``parsed_ok`` is False when there was nothing scannable to extract — malformed JSON,
    or valid JSON with no well-formed ``messages`` array (a Responses-API-shaped body, a
    dialect the gateway doesn't model). Such bodies are scanned as one raw blob rather
    than forwarded unscanned: an upstream that parses the body differently from the
    gateway would otherwise answer content the guard never saw.

    ``tools`` comes off this parse and not the forwarded one, for the same reason
    ``messages`` does — see the separate-parse invariant on ``ChatCall``. It is read
    even when the messages array is unusable: a body the gateway cannot model can still
    carry tool definitions. It also carries the legacy ``functions[]`` array:
    OpenAI-compatible and most local servers still accept it, its entries reach the
    model's catalog the same way, and ``tool_inspector`` already reads the bare
    (unwrapped) shape — so screening one spelling and not the other only tells an
    attacker which to use.
    """
    try:
        data = json.loads(body)
    except (ValueError, AttributeError):
        return None, [], [], False
    if not isinstance(data, dict):
        return None, [], [], False
    tools = (data.get("tools") or []) + (data.get("functions") or [])
    messages = data.get("messages")
    if not isinstance(messages, list) or not messages:
        model = data.get("model") if isinstance(data.get("model"), str) else None
        return model, [], tools, False
    return data.get("model"), messages, tools, True


def _raw_blob_messages(body: bytes) -> list[dict]:
    """Wrap an unscannable body as a single untrusted tool-output item.

    Scan rather than reject: one heuristic pass over the raw bytes cannot break a
    well-formed client — a well-formed client never reaches this path — and it closes a
    bypass reachable by *shape* in a component whose entire premise is that nothing
    reaches the model unscanned.
    """
    try:
        text = body.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - decode with errors="replace" shouldn't raise
        return []
    if not text.strip():
        return []
    return [{"role": "tool", "content": text}]


# Bodies the client compressed. Decompressed for scanning *and* for forwarding: the guard,
# tool screen, classifier and redaction all read the parsed body, and a redacted rewrite
# would otherwise be plain JSON under a `content-encoding: gzip` header. This is what
# c588f35 left open when it kept that header out of _STRIP_REQUEST_HEADERS — stripping it
# without decompressing would misdescribe the bytes, so now we decompress and strip both.
#
# Bounded, because a gzip bomb runs about 1000:1. A declared encoding this cannot fully
# decode is REJECTED, not forwarded: forwarding it meant the guard, tool screen, classifier
# and redaction each read an empty request while the audit row recorded an affirmatively
# clean scan, and the sensitivity axis read `none`, so a secret that would have stayed
# local routed to cloud instead. Rejecting costs nothing measurable — no OpenAI-compatible
# client compresses a request body (checked: httpx, openai-python, continue.dev, OpenClaw).
_MAX_DECOMPRESSED_BYTES = 8 * 1024 * 1024


def _decompress_body(body: bytes, encoding: str) -> bytes | None:
    """Fully decompressed body, or None when the declared encoding cannot be honoured.

    `content-encoding` is a *list* (RFC 9110 §8.4), so it is split rather than compared
    whole: `gzip, identity` describes exactly the bytes `gzip` does, and matching the
    entire header value against one codec name sent it down the undecodable path with
    ordinary gzip bytes. `identity` is dropped; anything left that is not a single
    supported codec is undecodable, not "probably fine".
    """
    codings = [c.strip() for c in encoding.lower().split(",")]
    codings = [c for c in codings if c and c != "identity"]
    if not codings:
        return body  # identity, or a list of it: the bytes are already plain
    if len(codings) > 1:
        return None  # stacked codings — honour them in order or not at all
    if codings[0] in ("gzip", "x-gzip"):
        dec = zlib.decompressobj(16 + zlib.MAX_WBITS)
    elif codings[0] == "deflate":
        dec = zlib.decompressobj()
    else:
        return None
    try:
        out = dec.decompress(body, _MAX_DECOMPRESSED_BYTES)
    except zlib.error:
        return None
    # All three mean "this is not the whole body", and all three used to pass. `eof` is the
    # truncated stream — zlib returns the partial output *without raising*, so a cut-off
    # upload became a silently shortened request that then scanned and forwarded clean.
    # `unconsumed_tail` is the cap. `unused_data` is a second gzip member or trailing junk:
    # a real decoder concatenates members, this one would have dropped them.
    if not dec.eof or dec.unconsumed_tail or dec.unused_data:
        return None
    return out


def _error(status: int, message: str, type_: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"message": message, "type": type_}})


@dataclass
class _JSONObject:
    """Decoded JSON object that preserves member order and duplicate names."""

    pairs: list[tuple[str, object]]


def _json_object(pairs: list[tuple[str, object]]) -> _JSONObject:
    """Keep the wire object's complete member sequence instead of collapsing duplicates."""
    return _JSONObject(pairs)


def _redact_json_values(node, acc: dict[str, int]):
    """Redact every string *value* in a decoded JSON tree. Keys are left alone."""
    if isinstance(node, str):
        red = _redact_into(node, acc)
        return node if red is None else red
    if isinstance(node, _JSONObject):
        return _JSONObject([(key, _redact_json_values(value, acc))
                            for key, value in node.pairs])
    if isinstance(node, list):
        return [_redact_json_values(v, acc) for v in node]
    return node


def _dump_json_tree(node: object) -> str:
    """Encode a decoded JSON tree while preserving duplicate object members."""
    if isinstance(node, _JSONObject):
        members = (
            f"{json.dumps(key, ensure_ascii=False)}:{_dump_json_tree(value)}"
            for key, value in node.pairs
        )
        return "{" + ",".join(members) + "}"
    if isinstance(node, list):
        return "[" + ",".join(_dump_json_tree(value) for value in node) + "]"
    return json.dumps(node, ensure_ascii=False)


def _redact_json_text(text: str, acc: dict[str, int]) -> str | None:
    """Redact a JSON-encoded string — a tool call's ``arguments`` — value by value.

    Redacting the raw text strands the backslash of any escape a match starts inside.
    `"import pytest\\n@pytest.fixture"` has the two characters `\\` and `n` in the parsed
    string, the email pattern reads `n@pytest.fixture` as an address, and the result is
    `"import pytest\\[REDACTED:email]"` — no longer a valid JSON escape, so the upstream
    can no longer parse the arguments it is handed, and the decorator is destroyed on the
    way. Decoding first means the redactor sees a newline as a newline, and `json.dumps`
    re-escapes whatever survives.

    Falls back to redacting the raw text only when it will not parse. Skipping it instead
    would let a secret in malformed `arguments` through unredacted, which is a worse
    failure than a mangled one. Parseable objects take a duplicate-preserving structural
    path: collapsing duplicate members can hide a secret, while sending them through the
    malformed fallback would reintroduce the dangling-escape corruption above.

    `ensure_ascii=False` so re-encoding does not gratuitously rewrite text it did not
    redact: a tool argument containing `é` comes back as `é`, not `\\u00e9`. Numeric
    lexemes still normalize (`1e+00` -> `1.0`) — that is the round trip through Python's
    number type, and it only reaches the wire on an arguments string that carried a
    secret in the first place.
    """
    try:
        data = json.loads(text, object_pairs_hook=_json_object)
    except ValueError:
        return _redact_into(text, acc)
    before = sum(acc.values())
    redacted = _redact_json_values(data, acc)
    if sum(acc.values()) == before:
        return None
    return _dump_json_tree(redacted)


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


async def capture_content(
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
        text = coerce_content(msg.get("content"))
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


def _span_attrs(call: ChatCall, status: int) -> dict:
    """The span attributes that describe the request itself — metadata only, never
    message content.

    The span mirrors the audit row, so both paths read the same fields off the
    `ChatCall`; the completion path merges in what the upstream told us.
    """
    # `provider` is unset only on a rejection taken before routing ran — an undecodable
    # `content-encoding`, which is answered ahead of classification because there is no
    # body to classify. Null rather than a placeholder, the same convention
    # `guard_backend` and `scanned_item_count` use for "that stage never ran".
    provider = call.provider
    sv = call.tool_verdict
    return {
        "gen_ai.operation.name": "chat",
        "gen_ai.system": provider.name if provider is not None else None,
        "gen_ai.request.model": call.model_requested,
        "agentgate.request_id": str(call.request_id),
        "agentgate.agent_id": call.agent_id,
        "agentgate.key_id": call.key_id,
        "agentgate.sensitivity_class": call.sensitivity_class or None,
        "agentgate.route.is_local": provider.is_local if provider is not None else None,
        "agentgate.status": status,
        "agentgate.guard.backend": call.guard_backend,
        "agentgate.injection.flagged": call.verdict.flagged,
        "agentgate.injection.hard": call.verdict.hard,
        "agentgate.injection.score": call.verdict.score,
        "agentgate.tool_def.flagged": sv.flagged,
        "agentgate.tool_def.hard": sv.hard,
        "agentgate.redaction.hit_count": call.redact_hit_count,
        "agentgate.latency.inject_ms": call.inject_ms,
        "agentgate.latency.redact_ms": call.redact_ms or None,
    }


def _audit_row(
    call: ChatCall,
    *,
    status: int,
    latency_total_ms: float,
    row_id: uuid.UUID | None = None,
    upstream_model: str | None = None,
    tokens_prompt: int = 0,
    tokens_completion: int = 0,
    cost_usd: float = 0.0,
    latency_upstream_ms: float | None = None,
    tool_call_count: int = 0,
    finish_reason: str | None = None,
) -> RequestAudit:
    """The audit row for one call.

    Rejection and completion differ only in what the upstream told us — the keyword
    arguments — so everything describing the request is read off the `ChatCall` in one
    place. One construction site: with two hand-maintained copies, every field added
    later is free to drift between the paths silently.
    """
    # Nullable route/sensitivity fields stay null when rejection preceded routing. The
    # legacy `route_is_local` column is non-nullable, so False is its sentinel; it means
    # "cloud" only when `route_provider` is also populated. See `_span_attrs`.
    provider = call.provider
    sv = call.tool_verdict
    return RequestAudit(
        id=row_id,
        ts=utcnow(), agent_id=call.agent_id, key_id=call.key_id,
        model_requested=call.model_requested,
        route_provider=provider.name if provider is not None else None,
        route_is_local=provider.is_local if provider is not None else False,
        upstream_model=upstream_model,
        sensitivity_class=call.sensitivity_class or None,
        tokens_prompt=tokens_prompt, tokens_completion=tokens_completion, cost_usd=cost_usd,
        latency_total_ms=latency_total_ms, latency_upstream_ms=latency_upstream_ms,
        latency_inject_ms=call.inject_ms,
        latency_redact_ms=call.redact_ms if call.redact_ms else None,
        injection_flagged=call.verdict.flagged, injection_hard=call.verdict.hard,
        injection_score=call.verdict.score,
        redaction_hit_count=call.redact_hit_count, redaction_hit_types=call.redact_hit_types,
        tool_def_flagged=sv.flagged, tool_def_hard=sv.hard,
        tool_def_reasons=sv.reasons if sv.flagged else None,
        tool_call_count=tool_call_count, finish_reason=finish_reason, status=status,
        guard_backend=call.guard_backend,
    )


async def _audit_rejected(store, call: ChatCall, status: int) -> None:
    """Write an audit row for a request rejected before forwarding.

    Every field comes off the `ChatCall`: a soft tool-definition flag raised at stage 3
    is still on the call when stage 5 or 6 rejects, so it lands on the row instead of
    defaulting to clean — the property is structural, not a discipline every rejection
    site has to remember. Same for the injection verdict: `ChatCall.verdict` starts as
    `Verdict.clean()`, so a rejection that fires *before* the scan records a genuine
    no-op rather than a stale one. Pinned by
    `test_injection_block_row_keeps_a_soft_tool_flag`,
    `test_soft_tool_def_spend_rejection_keeps_tool_def_flag_in_audit` and
    `test_injection_block_row_records_redaction_hits`.
    """
    latency_total_ms = (time.perf_counter() - call.t0) * 1000
    otel.record_chat(
        call.otel_parent, model=call.model_requested, duration_ms=latency_total_ms,
        attributes=_span_attrs(call, status),
    )
    await store.write(_audit_row(call, status=status, latency_total_ms=latency_total_ms))


def _reject(store, call: ChatCall, status: int, message: str, type_: str) -> JSONResponse:
    """Audit the rejection and answer the client, from one status.

    Pairing the two by hand at every early return is how a block ends up audited with one
    status and answered with another, or answered with no row at all. One call, one status.
    """
    _spawn(_audit_rejected(store, call, status))
    return _error(status, message, type_)


# --- stages --------------------------------------------------------------------------

async def new_call(request: Request, agent_id: str | None) -> ChatCall:
    """Read the request and take the first parse of the body (the one guards scan)."""
    t0 = time.perf_counter()
    body = await request.body()
    headers = dict(request.headers)
    encoding = headers.get("content-encoding")
    encoding_error: str | None = None
    if encoding:
        decoded = _decompress_body(body, encoding)
        if decoded is None:
            # Flagged, not raised: the reject belongs at the screen point with the other
            # block decisions, so it lands one audit row with one status like they do.
            encoding_error = encoding[:64]
        else:
            body = decoded
            # Forward what we scanned. The header described the bytes we just replaced.
            headers.pop("content-encoding", None)
    model_requested, messages, tools, parsed_ok = _parse_body(body)
    if parsed_ok:
        scan_messages = messages
    else:
        # Nothing scannable was extractable — scan the raw body as one untrusted blob
        # rather than forwarding it unscanned. `messages` stays [] for every
        # other consumer (classifier, capture, redaction), so this changes exactly one
        # thing: what the guard sees on a body it could not parse.
        scan_messages = _raw_blob_messages(body)
        if scan_messages:
            log.warning("unparseable body (%d bytes); scanning raw bytes as one untrusted "
                        "blob", len(body))
    return ChatCall(
        # Generate the request id up front so content samples can reference it before
        # the metadata row is written.
        request_id=uuid.uuid4(),
        t0=t0,
        body=body,
        headers=headers,
        key_id=key_id_from_auth(headers.get("authorization"), headers.get("x-goog-api-key")),
        agent_id=agent_id,
        model_requested=model_requested,
        messages=messages,
        payload=None,
        scan_messages=scan_messages,
        raw_tools=tools,
        encoding_error=encoding_error,
        otel_parent=otel.capture_parent(),
    )


def select_provider(settings, sensitivity_class: str, agent_id: str | None) -> Provider:
    """The upstream this request goes to: the rules-table router when routing is on,
    else the configured default. Pure — no I/O, no request state."""
    if not settings.routing.enabled:
        return settings.provider()
    decision = routing.decide(
        routing.RouteContext(sensitivity=sensitivity_class, agent_id=agent_id),
        settings.routing,
    )
    return routing.resolve(settings, decision)


def classify_and_route(call: ChatCall, settings) -> None:
    """Sensitivity classification (cheap, inline, no egress) → routing + audit policy."""
    klass = sensitivity_mod.classify_request(call.messages, settings.private_repo_markers)
    call.sensitivity_class = str(klass.sensitivity)
    call.provider = select_provider(settings, call.sensitivity_class, call.agent_id)


def prepare_body(call: ChatCall, settings) -> None:
    """Build the body that actually gets forwarded.

    A second parse of the same bytes (see ChatCall's separate-parse invariant) is used
    for model rewrite and cloud-egress redaction, then serialized only if mutated.
    Redaction is cloud-only: local routes handle sensitive content with zero egress, so
    there is nothing to scrub outbound.
    """
    try:
        call.payload = json.loads(call.body)
    except (ValueError, AttributeError):
        call.payload = None  # malformed body — let the upstream reject it

    payload = call.payload
    if not isinstance(payload, dict):
        return

    provider = call.provider
    assert provider is not None  # set by classify_and_route, which runs first
    if provider.model_name is not None:
        payload["model"] = provider.model_name
        call.mutated = True
    # Local-route request overrides + prompt cleaning (env-driven, default-off — see
    # local_route.py). Applied after the model rewrite so the override wins.
    if provider.is_local and local_route.adapt(payload, settings):
        call.mutated = True

    if not provider.is_local and settings.redaction_enabled:
        t_redact = time.perf_counter()
        acc: dict[str, int] = {}
        for msg in payload.get("messages") or []:
            raw = msg.get("content")
            if isinstance(raw, str) and raw:
                red = _redact_into(raw, acc)
                if red is not None:
                    msg["content"] = red
                    call.mutated = True
            elif isinstance(raw, list):
                for part in raw:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        red = _redact_into(part["text"], acc)
                        if red is not None:
                            part["text"] = red
                            call.mutated = True
            # Not an `elif`, and deliberately outside the `content` branches: on an
            # assistant tool-call message `content` is None, so keying off it skipped
            # the arguments entirely. Redacted through the JSON rather than over it —
            # `[REDACTED:...]` carries no backslash of its own, but replacing a span that
            # *starts inside* an escape leaves the escape's backslash dangling, and the
            # arguments stop being parseable. See `_redact_json_text`.
            for tc in msg.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function")
                if not isinstance(fn, dict) or not isinstance(fn.get("arguments"), str):
                    continue
                red = _redact_json_text(fn["arguments"], acc)
                if red is not None:
                    fn["arguments"] = red
                    call.mutated = True
        call.redact_hit_count = sum(acc.values())
        call.redact_hit_types = [{"type": t, "count": c} for t, c in acc.items()] or None
        call.redact_ms = (time.perf_counter() - t_redact) * 1000

    if call.mutated:
        call.body = json.dumps(payload).encode()


def inspect_tool_definitions(call: ChatCall, audit_store) -> Response | None:
    """Static analysis of tools[] definitions.

    Hard tier: block on description-level instruction injection (low FP).
    Soft tier: record-only (coarse heuristics, high FP — observe first).
    """
    call.tool_verdict = inspect_tools(call.raw_tools)
    if call.tool_verdict.flagged:
        log.warning("tool_inspector flagged: key=%s hard=%s reasons=%s",
                    call.key_id, call.tool_verdict.hard, call.tool_verdict.reasons)
    if not call.tool_verdict.hard:
        return None
    return _reject(
        audit_store, call, 400,
        f"blocked: malicious tool definition ({', '.join(call.tool_verdict.reasons)})",
        "tool_def_blocked",
    )


def tap_capture(call: ChatCall, settings) -> None:
    """Passive traffic capture (off by default; only the tagged capture agent).

    Fire-and-forget, before the injection-scan and spend decisions, so message-borne
    injections that trip the guard — the high-value samples — are captured even when
    the request is rejected. Requests rejected one stage earlier by the
    tool-definition screen are not captured: this tap sits below that block.
    """
    if settings.capture_enabled and call.agent_id == settings.capture_agent_id:
        assert call.agent_id is not None  # eligibility requires the configured capture agent
        _spawn(capture.capture(settings.capture_path, call.agent_id, call.messages))


def screen_content_encoding(call: ChatCall, audit_store) -> Response | None:
    """Reject a body whose declared `content-encoding` we could not fully decode.

    Forwarding it was the whole bypass: nothing could parse the compressed bytes, so the
    guard, tool screen, classifier and redaction all read an empty request, the row
    recorded an affirmatively clean scan, and `sensitivity=none` sent a secret-bearing
    body to cloud. This is the one stage that must run before the body is used for
    anything, because on this path there is no body to use.
    """
    if call.encoding_error is None:
        return None
    return _reject(
        audit_store, call, 400,
        f"unsupported or undecodable content-encoding: {call.encoding_error}",
        "unsupported_encoding",
    )


async def run_injection_scan(
    call: ChatCall, settings, deberta_available: bool, audit_store
) -> Response | None:
    """Injection scan of the untrusted content; block on hard-positive.

    Resolving the backend separately from dispatching it is deliberate: the audited
    backend is the one that actually ran, including the per-request deberta fallback.
    """
    t_inject = time.perf_counter()
    call.guard_backend = guards.resolve_backend(settings, call.key_id, deberta_available)
    call.verdict = await guards.scan(call.guard_backend, call.scan_messages)
    call.inject_ms = (time.perf_counter() - t_inject) * 1000

    verdict = call.verdict
    if verdict.flagged:
        log.warning("injection flagged: key=%s score=%.2f reasons=%s hard=%s",
                    call.key_id, verdict.score, verdict.reasons, verdict.hard)
    if verdict.hard and not settings.guard_observe_mode:
        metrics.injection_flagged_total.labels("True").inc()
        return _reject(audit_store, call, 400,
                       f"blocked: prompt injection detected ({', '.join(verdict.reasons)})",
                       "injection_blocked")
    if verdict.hard:
        # Observe mode: would-block, but forward anyway. Logged + audited (injection_hard=True
        # in _finalize) so live FP candidates stay countable without breaking the agent loop.
        log.warning("injection OBSERVE (would-block, forwarding): key=%s score=%.2f reasons=%s",
                    call.key_id, verdict.score, verdict.reasons)
        metrics.injection_flagged_total.labels("True").inc()
    elif verdict.flagged:
        metrics.injection_flagged_total.labels("False").inc()
    return None


async def check_spend(call: ChatCall, spend: SpendTracker, audit_store) -> Response | None:
    """Spend cap + kill switch."""
    assert call.provider is not None  # set by classify_and_route, which runs first
    try:
        await spend.check(call.key_id, call.provider.is_local)
    except SpendExceeded as exc:
        return _reject(audit_store, call, 429, exc.reason, "spend_exceeded")
    return None


async def forward_and_stream(
    call: ChatCall,
    client: httpx.AsyncClient,
    settings,
    audit_store: AuditStore,
    spend: SpendTracker,
    cipher: ContentCipher,
) -> Response:
    """Forward to the upstream and stream the SSE straight back, tapping usage."""
    provider = call.provider
    assert provider is not None  # set by classify_and_route, which runs first
    verdict = call.verdict
    cm = forward_stream(client, provider, inbound_headers=call.headers, body=call.body,
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
            # Accounting first. On a client disconnect the generator is cancelled, so an
            # `await` in here resumes with the cancellation pending and re-raises at the
            # first point it suspends — anything behind it would be skipped.
            #
            # Ordering, not repair: on this stack the row landed either way. httpcore
            # shields both halves of the upstream release, so it never actually suspends
            # here, and a real-socket disconnect against the pre-fix code audited every
            # time (uvicorn, FIN and RST, h11 and httptools). This ordering is what keeps
            # that true if the release ever stops being shielded; it is not a fix for an
            # observed loss. What the disconnect DOES cost is the numbers: usage arrives
            # in the stream's final event, so a request cut short records tok=0/0 and
            # cost 0.0 — with this ordering and without it. See ROADMAP #22; the per-key
            # cap is cooperative accounting, not a boundary that survives a hangup.
            #
            # The nested `finally` keeps the release from being skipped if `_finalize`
            # itself raises.
            try:
                _finalize()
            finally:
                await cm.__aexit__(None, None, None)

    def _finalize() -> None:
        now = time.perf_counter()
        latency_total_ms = (now - call.t0) * 1000
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
            call.key_id, provider.name, r.upstream_model, r.prompt_tokens, r.completion_tokens,
            cost, r.tool_call_count, finish, verdict.score, latency_total_ms,
        )

        # The span mirrors the audit row below — metadata only, never message content.
        otel.record_chat(
            call.otel_parent, model=call.model_requested, duration_ms=latency_total_ms,
            attributes=_span_attrs(call, upstream.status_code) | {
                "gen_ai.response.model": r.upstream_model,
                "gen_ai.usage.input_tokens": r.prompt_tokens,
                "gen_ai.usage.output_tokens": r.completion_tokens,
                "gen_ai.usage.cost": cost or None,
                "gen_ai.response.finish_reasons": [finish] if finish else None,
                "agentgate.tool_call_count": r.tool_call_count,
                "agentgate.latency.upstream_ms": latency_upstream_ms,
            },
        )
        _spawn(spend.record(call.key_id, provider.is_local, cost))
        _spawn(audit_store.write(_audit_row(
            call, status=upstream.status_code, latency_total_ms=latency_total_ms,
            row_id=call.request_id, upstream_model=r.upstream_model,
            tokens_prompt=r.prompt_tokens, tokens_completion=r.completion_tokens,
            cost_usd=cost, latency_upstream_ms=latency_upstream_ms,
            tool_call_count=r.tool_call_count, finish_reason=finish,
        )))
        # --- Content capture (off hot path) ---
        # Never for local routes (sensitive content stays on-box) or sensitive content.
        # Capture flagged requests always; benign cloud at the configured sample rate.
        is_sensitive = call.sensitivity_class != str(Sensitivity.NONE)
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
                _spawn(capture_content(
                    audit_store, cipher, call.messages, call.request_id,
                    sample_reason, settings.content_retention_days,
                ))

    return StreamingResponse(
        stream_and_finalize(), status_code=upstream.status_code,
        headers=upstream.headers, media_type="text/event-stream",
    )


async def handle_chat(request: Request, agent_id: str | None) -> Response:
    """Sequence the stages. Every early return is a block decision."""
    settings = request.app.state.settings
    audit_store: AuditStore = request.app.state.audit
    spend: SpendTracker = request.app.state.spend
    client: httpx.AsyncClient = request.app.state.http
    cipher: ContentCipher = request.app.state.cipher

    call = await new_call(request, agent_id)

    # Ahead of every other stage that reads the body, classification included: on this
    # path there is no readable body, so anything reading it reads nothing — and then
    # records that nothing as a finding. Classification used to run first and put an
    # affirmative `sensitivity=none` on the row for a body it had not been able to read.
    blocked = screen_content_encoding(call, audit_store)
    if blocked is not None:
        return blocked

    classify_and_route(call, settings)

    prepare_body(call, settings)

    blocked = inspect_tool_definitions(call, audit_store)
    if blocked is not None:
        return blocked

    tap_capture(call, settings)

    deberta_available = getattr(request.app.state, "deberta_available", True)
    blocked = await run_injection_scan(call, settings, deberta_available, audit_store)
    if blocked is not None:
        return blocked

    blocked = await check_spend(call, spend, audit_store)
    if blocked is not None:
        return blocked

    return await forward_and_stream(call, client, settings, audit_store, spend, cipher)


# Accept both with and without the /v1 prefix: OpenAI-compatible clients may or may not
# include /v1 in their baseUrl. Tolerate both. Untagged = no agent attribution.
@router.post("/v1/chat/completions")
@router.post("/chat/completions")
async def chat_completions(request: Request) -> Response:
    return await handle_chat(request, agent_id=None)


# Agent-tagged variant: an agent's baseUrl is set to ".../a/<agent_id>", so its requests
# arrive here and carry attribution (drives traffic capture + the router's agent-pin rules).
# The {agent_id} segment is the only way to identify the agent — the client sends vanilla
# OpenAI Chat Completions with no agent field in the payload.
@router.post("/a/{agent_id}/v1/chat/completions")
@router.post("/a/{agent_id}/chat/completions")
async def chat_completions_for_agent(request: Request, agent_id: str) -> Response:
    return await handle_chat(request, agent_id=agent_id)
