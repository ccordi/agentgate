"""The chat data plane — everything that happens to a request on the model wire.

Stage order (synchronous, pre-forward), and where each stage's logic lives:

  0. issued-key auth (`keys`) → 401, then admission control (`limits.admission`) →
     429/503. Both off by default and both ahead of the body read, so identity comes from
     the headers alone; the admission slot is held until the response has drained
  1. reject an undecodable `content-encoding` (`screen_content_encoding`) → 400. First,
     ahead of classification: there is no readable body for any later stage to read
  2. classify sensitivity (`sensitivity`) → route (`routing`)
  3. prepare the outbound body: provider model rewrite, local-route adaptation
     (`local_route`), cloud-only redaction (`redaction`)
  4. upstream-credential screen — with issued keys required, a non-local provider with
     no `api_key` to inject would be sent the minted gateway key → 503 (a minted key
     never leaves the gateway)
  5. tool-definition screening (`tool_inspector`) → 400 on a hard verdict
  6. injection scan (`guards`) → 400 on a hard verdict, unless observe mode;
     503 when the resolved guard could not run at all (a blocking control fails closed)
  7. spend cap + kill switch (`limits`) → 429 over cap, 503 if the backend is down
  8. forward and stream back (`proxy`), tapping usage
Post-stream (off hot path): record spend, write the metadata audit row, update metrics,
sample content into the encrypted content tier.

The body is re-serialized once, and only if a stage mutated it.
"""

from __future__ import annotations

import asyncio
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

from agentgate import guards, keys, local_route, routing
from agentgate import sensitivity as sensitivity_mod
from agentgate.audit.crypto import ContentCipher
from agentgate.audit.store import AuditStore, ContentSampleAudit, RequestAudit, utcnow
from agentgate.config import Provider
from agentgate.content import (
    JSONObject,
    coerce_content,
    json_object,
    map_json_lexemes,
    tool_call_functions,
    trailing_tool_outputs,
)
from agentgate.limits import admission
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

    # --- `messages` and `payload` are two INDEPENDENT parses of the same bytes. ---
    # `messages` is what the guards scan; `payload` is what gets redacted and forwarded.
    # Keeping them disjoint is what makes redaction unable to blind the guard, on both
    # routes. Do NOT "simplify" this to `messages = payload["messages"]`: that hands the
    # scanner `[REDACTED:openai_key] …` instead of the attacker's text on every cloud
    # request carrying a secret, silently, while looking like a cleanup.
    # Pinned by tests/test_separate_parse.py.
    messages: list[dict]
    payload: dict | None            # None if the body is malformed — the upstream rejects it

    # What the guard actually scans. Normally `messages`; on a body the gateway could not
    # parse it is the raw bytes wrapped as one untrusted tool-output item, which is the
    # ONE consumer that differs — the classifier and redaction still see the empty
    # `messages`.
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

    # Caveats on what the verdicts above mean, appended in stage order and written to
    # `RequestRecord.caveats` as-is (None when empty). Closed vocabulary — see the column.
    caveats: list[str] = field(default_factory=list)

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
    ``messages`` does — see the note on ``ChatCall.messages``. It is read even when the
    messages array is unusable: a body the gateway cannot model can still carry tool
    definitions.
    It also carries the legacy ``functions[]`` array: OpenAI-compatible and most local
    servers still accept it, its entries reach the model's catalog the same way, and
    ``tool_inspector`` already reads the bare (unwrapped) shape — so screening one
    spelling and not the other only tells an attacker which to use.
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
# would otherwise be plain JSON under a `content-encoding: gzip` header. That is also why
# _STRIP_REQUEST_HEADERS does not list the header: stripping it without decompressing
# would misdescribe the bytes, so we decompress and strip both.
#
# Bounded, because a gzip bomb runs about 1000:1. A declared encoding this cannot fully
# decode is REJECTED, not forwarded: forwarding it would mean the guard, tool screen,
# classifier and redaction each read an empty request while the audit row records an
# affirmatively clean scan, and the sensitivity axis reads `none`, so a secret that
# should stay local routes to cloud instead. Rejecting costs little: common
# OpenAI-compatible clients (httpx, openai-python) do not compress request bodies.
_MAX_DECOMPRESSED_BYTES = 8 * 1024 * 1024


def _decompress_body(body: bytes, encoding: str) -> bytes | None:
    """Fully decompressed body, or None when the declared encoding cannot be honoured.

    `content-encoding` is a *list* (RFC 9110 §8.4), so it is split rather than compared
    whole: `gzip, identity` describes exactly the bytes `gzip` does, and matching the
    entire header value against one codec name would send it down the undecodable path
    with ordinary gzip bytes. `identity` is dropped; anything left that is not a single
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
    # All three mean "this is not the whole body". `eof` is the truncated stream — zlib
    # returns the partial output *without raising*, so a cut-off upload would become a
    # silently shortened request that then scans and forwards clean. `unconsumed_tail` is
    # the cap. `unused_data` is a second gzip member or trailing junk: a real decoder
    # concatenates members, this one would drop them.
    if not dec.eof or dec.unconsumed_tail or dec.unused_data:
        return None
    return out


# The wire `error.type` strings a pre-forward rejection can carry. `_reject` records the
# same string on the audit row (`rejected:<type>`), so the closed set here is what keeps
# the caveat column free of anything but these seven. `upstream_error` (502) is not one:
# that path writes no audit row today, and the admission and auth refusals never read
# the request at all.
REJECTION_TYPES = frozenset({
    "unsupported_encoding",
    "upstream_credentials_missing",
    "tool_def_blocked",
    "injection_blocked",
    "guard_unavailable",
    "spend_exceeded",
    "limits_unavailable",
})


def _error(status: int, message: str, type_: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"message": message, "type": type_}})


def _redact_json_values(node, acc: dict[str, int]):
    """Redact every lexeme in a decoded JSON tree that the detectors can match.

    String values, object keys and numbers of a scannable length — everything a raw-text
    pass would see. Walking string values only would miss two leaks: an email or key
    used as an *object key* (a tool result keyed by address is ordinary API output) and a
    card or phone number sent as a *JSON number*, both forwarded with zero hits recorded.
    The enumeration itself is `content.map_json_lexemes`, shared with the sensitivity
    classifier so the two readers of a document can never disagree about what it
    contains; what it visits, and why short numbers are skipped, is
    documented there. Here every lexeme goes through `_redact_into`: a key is redacted
    exactly like a value, two keys that redact to the same placeholder are both kept, in
    order, the way the encoder already keeps duplicate names, and a redacted number
    becomes the placeholder *string* — the document stays valid JSON at the price of a
    changed type. On no hit every node comes back as it was, so a tree with nothing to
    redact is never re-encoded (see `_redact_json_tree`).
    """
    return map_json_lexemes(node, lambda lexeme: _redact_into(lexeme, acc))


def _dump_json_tree(node: object) -> str:
    """Encode a decoded JSON tree while preserving duplicate object members."""
    if isinstance(node, JSONObject):
        members = (
            f"{json.dumps(key, ensure_ascii=False)}:{_dump_json_tree(value)}"
            for key, value in node.pairs
        )
        return "{" + ",".join(members) + "}"
    if isinstance(node, list):
        return "[" + ",".join(_dump_json_tree(value) for value in node) + "]"
    return json.dumps(node, ensure_ascii=False)


def _redact_json_tree(data: object, acc: dict[str, int]) -> str | None:
    """Redact a decoded JSON tree; the re-encoded text when anything was found, else None."""
    before = sum(acc.values())
    redacted = _redact_json_values(data, acc)
    if sum(acc.values()) == before:
        return None
    return _dump_json_tree(redacted)


def _redact_json_text(text: str, acc: dict[str, int]) -> str | None:
    """Redact a JSON-encoded string — a tool call's ``arguments`` — value by value.

    Redacting the raw text strands the backslash of any escape a match starts inside.
    `"import pytest\\n@pytest.fixture"` has the two characters `\\` and `n` in the parsed
    string, the email pattern reads `n@pytest.fixture` as an address, and the result is
    `"import pytest\\[REDACTED:email]"` — no longer a valid JSON escape, so the upstream
    can no longer parse the arguments it is handed, and the decorator is destroyed on the
    way. Decoding first means the redactor sees a newline as a newline, and `json.dumps`
    re-escapes whatever survives.

    Falls back to redacting the raw text when it will not parse, and when it parses but
    nests too deep to walk. Skipping it instead would let a secret in malformed
    `arguments` through unredacted, which is a worse failure than a mangled one.
    Parseable objects take a duplicate-preserving structural path: collapsing duplicate
    members can hide a secret, while sending them through the malformed fallback would
    reintroduce the dangling-escape corruption above.

    The deep-document window is `_redact_content_text`'s, on the other attacker-influenced
    channel — a tool call is built from whatever a tool handed the agent: `json.loads`
    parses thousands of nesting levels (its guard is the C stack) while the value walker
    and the encoder are Python recursion and give out near the recursion limit, and a
    document in that window must fall back to raw text rather than turn redaction into a
    500. The tally is rolled back first, so an abandoned walk's hits are not counted again
    by the raw pass.

    `ensure_ascii=False` so re-encoding does not gratuitously rewrite text it did not
    redact: a tool argument containing `é` comes back as `é`, not `\\u00e9`. Numeric
    lexemes still normalize (`1e+00` -> `1.0`) — that is the round trip through Python's
    number type, and it only reaches the wire on an arguments string that carried a
    secret in the first place.
    """
    snapshot = dict(acc)
    try:
        data = json.loads(text, object_pairs_hook=json_object)
        return _redact_json_tree(data, acc)
    except (ValueError, RecursionError):
        acc.clear()
        acc.update(snapshot)
    return _redact_into(text, acc)


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


def _redact_content_text(text: str, acc: dict[str, int]) -> str | None:
    """Redact a message's ``content`` (or one text part) — structurally when it is JSON.

    A user or tool message whose text is itself a JSON document — an API response the
    agent fetched, a config file, a log it pasted — carries the same escapes ``arguments``
    does, and raw-text redaction *leaks* on them: the ``\\b``-anchored detectors read the
    raw characters, so `"\\nAKIA…"` presents `nAKIA…` (no word boundary, no match),
    `"user\\u0040example.com"` has no `@` at all, and `password = \\"…\\"` has no quote
    where the assignment pattern needs one — each would be forwarded unredacted with zero
    hits recorded. Where a match does land inside an escape it strands the backslash
    instead: `"a\\nuser@example.com"` becomes `"a\\[REDACTED:email]"`, no longer JSON.
    Decoding first hands the detectors the text the model will read; `_redact_json_text`
    has the mechanics and the properties kept, `_redact_json_values` what is scanned.

    Structural, not a lookbehind: `(?<!\\\\)` on the email pattern would make
    `\\nuser@example.com` match with local part `nuser` — suppressing a real address
    instead of shifting the boundary, a false negative bought with a false positive.

    Non-JSON content takes the raw path, unchanged. So does text that decodes to a bare
    number, boolean or null: it carries no escape to strand, so raw redaction is exact
    and keeps a redacted bare number unquoted. And because content is
    the attacker-influenced channel, the structural attempt is bounded: `json.loads`
    parses thousands of nesting levels (its guard is the C stack), while the value walker
    and the encoder are Python recursion and give out near the recursion limit — a
    document in that window falls back to raw text, with the hit tally rolled back first,
    rather than turning redaction into a 500.
    """
    snapshot = dict(acc)
    try:
        data = json.loads(text, object_pairs_hook=json_object)
        if isinstance(data, (JSONObject, list, str)):
            return _redact_json_tree(data, acc)
    except (ValueError, RecursionError):
        acc.clear()
        acc.update(snapshot)
    return _redact_into(text, acc)


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

    # The untrusted surface, as the guards actually scanned it — `extract_untrusted` in
    # message form: this turn's whole unanswered tool-output batch (`tool` and the legacy
    # `function` role, parallel calls included, wherever the agent application put it in
    # the array) plus the newest user turn. The message
    # dicts are kept rather than reusing `extract_untrusted`'s (source, text) pairs
    # because the row records the real `role`, and redaction re-runs on the text below.
    #
    # Deriving the candidates any other way here would let a flagged sample store
    # content that is not what flagged — one of N batch results, or a tool message turns
    # old re-sampled as if it had arrived now — and omit what did. No per-request row cap:
    # a cap would re-open exactly that divergence. Volume is held by the caller's
    # flagged-or-sampled gate and the retention sweeper.
    candidates: list[dict] = list(trailing_tool_outputs(messages))
    last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
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
        "agentgate.caveats": call.caveats or None,
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
        # None, not 0, when the row predates any scan — guard_backend is None on exactly
        # those rows, so the two columns stay consistent about "no scan ran".
        scanned_item_count=(
            call.verdict.scanned_items if call.guard_backend is not None else None
        ),
        # None, never []: null is "nothing to report", and it is also what every row
        # written before the column existed reads — one meaning, one query.
        caveats=call.caveats or None,
    )


async def _audit_rejected(store, call: ChatCall, status: int) -> None:
    """Write an audit row for a request rejected before forwarding.

    Every field comes off the `ChatCall`: a soft tool-definition flag raised at the tool
    screen is still on the call when a later stage rejects, so it lands on the row instead of
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
    # row_id must be the request's own id, as on the completion path: without it the row
    # gets a fresh uuid, and exactly the enforcement-action rows (blocks, rejections)
    # become unjoinable to their span and content samples.
    await store.write(_audit_row(
        call, status=status, latency_total_ms=latency_total_ms, row_id=call.request_id,
    ))


def _reject(store, call: ChatCall, status: int, message: str, type_: str) -> JSONResponse:
    """Audit the rejection and answer the client, from one status.

    Pairing the two by hand at every early return is how a block ends up audited with one
    status and answered with another, or answered with no row at all. One call, one status.

    One reason, too: the wire `type_` is what the row records, so the audit and the
    client's answer can never disagree about why — and every rejection site gets it
    structurally, the way `_audit_rejected` already threads the verdicts.
    """
    assert type_ in REJECTION_TYPES, type_  # the caveat column carries no free text
    call.caveats.append(f"rejected:{type_}")
    metrics.rejects_total.labels(type_).inc()
    log.info("rejected: key=%s status=%d type=%s", call.key_id, status, type_)
    _spawn(_audit_rejected(store, call, status))
    return _error(status, message, type_)


def _record_gap_abandoned(call: ChatCall, tags: list[str]) -> None:
    """Thread a screen's gap-abandon observations onto the row.

    Padding a known phrase past a known bound is attacker-shaped, so each one is
    counted and warned — but it is never a verdict: the tags ride on `caveats`, and the
    flagged/hard fields the screens set are untouched by construction (the probes never
    write them). `tags` are the closed `gap_abandoned:<pattern>:<distance>` strings.
    """
    for tag in tags:
        call.caveats.append(tag)
        metrics.gap_abandoned_total.labels(tag.split(":")[1]).inc()
        log.warning("bounded-gap pattern abandoned past its bound: key=%s %s",
                    call.key_id, tag)


# --- stages --------------------------------------------------------------------------

async def new_call(request: Request, agent_id: str | None, key_id: str) -> ChatCall:
    """Read the request and take the first parse of the body (the one guards scan).

    `key_id` is derived from the headers by the caller, before the body is read, and
    passed in so there is one derivation per request rather than one per consumer.
    """
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
        # other consumer (classifier, redaction), so this changes exactly one
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
        key_id=key_id,
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
    if klass.truncated_at is not None:
        # The verdict describes the first N characters only. Say so on the row, so
        # "classified clean" and "classified the head and stopped" stay distinguishable.
        call.caveats.append(f"truncated:classify:{klass.truncated_at}")
        metrics.scan_truncated_total.labels("classify").inc()
        log.info("classify window exhausted at %d chars: key=%s messages=%d sensitivity=%s",
                 klass.truncated_at, call.key_id, len(call.messages), call.sensitivity_class)
    call.provider = select_provider(settings, call.sensitivity_class, call.agent_id)


def prepare_body(call: ChatCall, settings) -> None:
    """Build the body that actually gets forwarded.

    A SECOND parse of the same bytes (see the note on ChatCall.messages) for model
    rewrite and cloud-egress redaction — re-dumped once, only if mutated.
    Redaction is cloud-only: local routes handle sensitive content with zero egress, so
    there is nothing to scrub outbound.
    """
    try:
        call.payload = json.loads(call.body)
    except (ValueError, AttributeError):
        call.payload = None  # malformed body — scanned as a raw blob, forwarded as-is

    payload = call.payload
    if not isinstance(payload, dict):
        return

    provider = call.provider
    assert provider is not None  # set by classify_and_route, which runs first
    if provider.model_name is not None:
        payload["model"] = provider.model_name
        call.mutated = True
    # Local-route request overrides (env-driven, off until set) and prompt cleaning (always
    # on) — see local_route.py. Applied after the model rewrite so the override wins.
    if provider.is_local and local_route.adapt(payload, settings):
        call.mutated = True

    if not provider.is_local and settings.redaction_enabled:
        t_redact = time.perf_counter()
        acc: dict[str, int] = {}
        for msg in payload.get("messages") or []:
            # Content that is itself a JSON document is redacted through the JSON, the
            # way `arguments` is below — see `_redact_content_text`. Prose is unchanged.
            raw = msg.get("content")
            if isinstance(raw, str) and raw:
                red = _redact_content_text(raw, acc)
                if red is not None:
                    msg["content"] = red
                    call.mutated = True
            elif isinstance(raw, list):
                for part in raw:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        red = _redact_content_text(part["text"], acc)
                        if red is not None:
                            part["text"] = red
                            call.mutated = True
            # Not an `elif`, and deliberately outside the `content` branches: on an
            # assistant tool-call message `content` is None, so keying off it would skip
            # the arguments entirely. Redacted through the JSON rather than over it —
            # `[REDACTED:...]` carries no backslash of its own, but replacing a span that
            # *starts inside* an escape leaves the escape's backslash dangling, and the
            # arguments stop being parseable. See `_redact_json_text`.
            #
            # Both tool-call spellings, through the same enumeration the classifier reads
            # (`content.tool_call_functions`): a set of arguments one of them walks and
            # the other does not is exactly a secret that classifies `none`, routes to
            # cloud, and then egresses untouched.
            for fn in tool_call_functions(msg):
                red = _redact_json_text(fn["arguments"], acc)
                if red is not None:
                    fn["arguments"] = red
                    call.mutated = True
        call.redact_hit_count = sum(acc.values())
        call.redact_hit_types = [{"type": t, "count": c} for t, c in acc.items()] or None
        call.redact_ms = (time.perf_counter() - t_redact) * 1000

    if call.mutated:
        call.body = json.dumps(payload).encode()


def screen_upstream_credentials(call: ChatCall, settings, audit_store) -> Response | None:
    """Refuse to forward a gateway-minted key to a non-local upstream.

    With issued keys required, the inbound credential IS a minted gateway key. The
    forwarder passes inbound auth through unless the provider carries its own
    ``api_key`` to inject — so a non-local provider without one would be sent the
    minted key: a guaranteed upstream 401, with a live gateway credential landing in a
    third party's request logs on the way. A minted key never leaves the gateway: fail
    closed here, at the forward decision, before any headers are built. Scoped to the
    provider this request actually resolved to, so unused keyless cloud entries in the
    registry cost nothing (forced-local deployments are valid; boot warns rather than
    refuses — see ``validate_runtime_settings``).
    """
    provider = call.provider
    assert provider is not None  # set by classify_and_route, which runs first
    if not settings.require_issued_keys or provider.is_local or provider.api_key is not None:
        return None
    log.error("provider %r resolved with no api_key while issued keys are required; "
              "refusing to forward the minted gateway key", provider.name)
    metrics.upstream_credentials_missing_total.labels(provider.name).inc()
    return _reject(
        audit_store, call, 503,
        f"no upstream credential configured for provider '{provider.name}'; request "
        f"not forwarded because the gateway-minted inbound key must not be sent upstream",
        "upstream_credentials_missing",
    )


def inspect_tool_definitions(call: ChatCall, audit_store) -> Response | None:
    """Static analysis of tools[] definitions.

    Hard tier: block on description-level instruction injection (few false positives).
    Soft tier: record-only (coarse heuristics, many false positives).
    """
    call.tool_verdict = inspect_tools(call.raw_tools)
    _record_gap_abandoned(call, call.tool_verdict.gap_abandoned)
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


def screen_content_encoding(call: ChatCall, audit_store) -> Response | None:
    """Reject a body whose declared `content-encoding` we could not fully decode.

    Forwarding it would be a bypass: nothing can parse the compressed bytes, so the
    guard, tool screen, classifier and redaction would all read an empty request, the row
    would record an affirmatively clean scan, and `sensitivity=none` would send a
    secret-bearing body to cloud. This is the one stage that must run before the body is
    used for anything, because on this path there is no body to use.
    """
    if call.encoding_error is None:
        return None
    return _reject(
        audit_store, call, 400,
        f"unsupported or undecodable content-encoding: {call.encoding_error}",
        "unsupported_encoding",
    )


async def run_injection_scan(call: ChatCall, settings, audit_store) -> Response | None:
    """Injection scan of the untrusted content; block on hard-positive.

    Resolving the backend separately from dispatching it is deliberate: the audited
    backend is the one that actually ran. No availability fallback: a backend whose
    model will not load refuses startup (`app.lifespan`).
    """
    t_inject = time.perf_counter()
    call.guard_backend = guards.resolve_backend(settings, call.key_id)
    try:
        call.verdict = await guards.scan(call.guard_backend, call.scan_messages)
    except guards.GuardUnavailable as exc:
        # A blocking safety control fails closed with a distinguishable
        # status. 503 + a named error type, not a 500 — the agent (and the operator
        # reading the audit row) can tell "the guard could not run" from "the gateway
        # broke". Audited so an outage is countable rather than invisible.
        call.inject_ms = (time.perf_counter() - t_inject) * 1000
        metrics.guard_unavailable_total.labels(exc.backend).inc()
        return _reject(audit_store, call, 503,
                       f"injection guard unavailable ({exc.backend}); request not "
                       f"forwarded because it could not be scanned",
                       "guard_unavailable")
    call.inject_ms = (time.perf_counter() - t_inject) * 1000
    # Before the block decision, so a row that is then rejected still carries it.
    _record_gap_abandoned(call, call.verdict.gap_abandoned)

    verdict = call.verdict
    if verdict.flagged:
        log.warning("injection flagged: key=%s score=%.4f reasons=%s hard=%s",
                    call.key_id, verdict.score, verdict.reasons, verdict.hard)
    if verdict.hard and not settings.guard_observe_mode:
        metrics.injection_flagged_total.labels("True").inc()
        return _reject(audit_store, call, 400,
                       f"blocked: prompt injection detected ({', '.join(verdict.reasons)})",
                       "injection_blocked")
    if verdict.hard:
        # Observe mode: would-block, but forward anyway. Logged + audited (injection_hard=True
        # in _finalize) so possible false positives on live traffic stay countable without
        # breaking the agent loop.
        log.warning("injection OBSERVE (would-block, forwarding): key=%s score=%.4f reasons=%s",
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
    except Exception as exc:
        # The startup path degrades to MemoryBackend deliberately, but a limits backend
        # dying mid-run must not surface as a framework 500. Same shape as the guard:
        # fail closed, distinguishable, counted.
        log.error("spend backend unavailable: %r", exc)
        metrics.limits_unavailable_total.labels("spend").inc()
        return _reject(audit_store, call, 503,
                       "spend/kill-switch backend unavailable; request not forwarded",
                       "limits_unavailable")
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
        # Total wall-clock deadline for the whole request: the per-read timeout alone
        # never trips on a slow drip (see config.upstream_timeout_s). Headers are already
        # sent by the time we stream, so a trip truncates the stream rather than changing
        # the status — a backstop against holding capacity forever, not an error path.
        remaining = settings.upstream_total_deadline_s - (time.perf_counter() - call.t0)
        try:
            async with asyncio.timeout(max(remaining, 0.0)):
                async for chunk in upstream.body:
                    yield chunk
        except TimeoutError:
            log.warning("request exceeded total deadline %.0fs (key=%s provider=%s); "
                        "stream truncated", settings.upstream_total_deadline_s, call.key_id,
                        provider.name)
        finally:
            # Accounting first. On a client disconnect the generator is cancelled, so an
            # `await` in here resumes with the cancellation pending and re-raises at the
            # first point it suspends — anything behind it would be skipped.
            #
            # A request cut short records zero tokens and zero cost whatever the order: usage
            # arrives in the stream's final event, so the per-key cap is cooperative
            # accounting, not a boundary that survives a hangup.
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
            "finish=%s inject=%.4f %.1fms",
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


# --- issued-key auth -------------------------------------------------------------------

def _bearer_token(authorization: str | None) -> str | None:
    """The token from a Bearer Authorization header, or None. Scheme is
    case-insensitive per RFC 7235 — the raw header is NOT identity here (see
    `_verify_issued_key`), so lenient parsing costs nothing."""
    if authorization is None:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


def _auth_error() -> JSONResponse:
    """One 401 for missing, unknown and revoked keys — no oracle telling a caller
    which; metrics and logs keep the distinction internally."""
    return JSONResponse(
        status_code=401,
        content={"error": {
            "message": "missing or invalid API key: this gateway requires a "
                       "gateway-issued key (AGENTGATE_REQUIRE_ISSUED_KEYS is on)",
            "type": "invalid_api_key",
        }},
        headers={"WWW-Authenticate": "Bearer"},
    )


async def _verify_issued_key(app, authorization: str | None, goog_key: str | None) -> str | None:
    """Canonical key_id for a valid minted credential, else None (counted by reason).

    The returned id comes from the mint record, not from hashing the raw header:
    raw-hash identity would let a valid key holder vary header casing or channel to
    obtain fresh per-key counters and dodge caps.
    """
    secret = _bearer_token(authorization) or goog_key
    if not secret:
        reason = "missing"
    else:
        row = await keys.key_store(app).verify(secret)
        if row is None:
            reason = "unknown"
        elif row.revoked_at is not None:
            reason = "revoked"
        else:
            return row.key_id
    metrics.auth_rejected_total.labels(reason).inc()
    log.warning("issued-key auth rejected: reason=%s", reason)
    return None


# --- admission -------------------------------------------------------------------------

# A per-key shed is that key asking for more than its share, so it is the caller's
# problem and retryable: 429. A global shed is the gateway out of capacity for everyone:
# 503. Named types so a client can tell the two apart from each other and from the
# spend cap's 429.
_ADMISSION_ERRORS: dict[str, tuple[int, str, str]] = {
    admission.QUEUE_FULL: (
        429, "admission_queue_full",
        "too many concurrent requests for this key; the wait queue is full",
    ),
    admission.DEADLINE: (
        429, "admission_timeout",
        "too many concurrent requests for this key; waited for a slot and did not get one",
    ),
    admission.GLOBAL: (
        503, "admission_capacity",
        "gateway is at its global concurrency ceiling; request not admitted",
    ),
}


def _admission_controller(app, settings) -> admission.AdmissionController:
    """The process's controller: built at startup, or here on first use for an app
    assembled without lifespan."""
    controller = getattr(app.state, "admission", None)
    if controller is None:
        controller = admission.AdmissionController.from_settings(settings)
        app.state.admission = controller
    return controller


def _hold_slot_until_drained(response: Response, lease: admission.Lease) -> Response:
    """Keep the admission slot until the response body is finished.

    Releasing when the handler returns would free the slot at the moment a streamed
    answer *starts*, which is the point of the whole control: one turn can stream for
    tens of seconds. A non-streaming response is already complete, so it releases here.

    The wrapper's `finally` is the same hook the upstream release and the audit write
    already ride on — it runs on normal exhaustion, and on a client disconnect when the
    abandoned generator is finalized. `Lease.release` is idempotent, so overlapping
    paths cost nothing.
    """
    if not isinstance(response, StreamingResponse):
        lease.release()
        return response
    body = response.body_iterator

    async def _held():
        try:
            async for chunk in body:
                yield chunk
        finally:
            lease.release()

    response.body_iterator = _held()
    return response


async def handle_chat(request: Request, agent_id: str | None) -> Response:
    """Auth, admission, then the stages.

    Identity comes off the headers here — before `new_call` reads the body — because
    admission has to decide whether to do any work at all before any work is done.
    Issued-key verification comes first of all: an unauthenticated caller gets no
    queue slot and no body read.
    """
    settings = request.app.state.settings
    authorization = request.headers.get("authorization")
    goog_key = request.headers.get("x-goog-api-key")
    if settings.require_issued_keys:
        verified = await _verify_issued_key(request.app, authorization, goog_key)
        if verified is None:
            return _auth_error()
        key_id = verified
    else:
        key_id = key_id_from_auth(authorization, goog_key)

    if not settings.admission_enabled:
        return await _handle_admitted(request, agent_id, key_id)

    try:
        lease = await _admission_controller(request.app, settings).acquire(key_id)
    except admission.Shed as shed:
        status, type_, message = _ADMISSION_ERRORS[shed.reason]
        metrics.admission_shed_total.labels(shed.reason).inc()
        log.warning("admission shed: key=%s reason=%s status=%d", key_id, shed.reason, status)
        return _error(status, message, type_)

    try:
        response = await _handle_admitted(request, agent_id, key_id)
    except BaseException:
        lease.release()
        raise
    return _hold_slot_until_drained(response, lease)


async def _handle_admitted(request: Request, agent_id: str | None, key_id: str) -> Response:
    """Sequence the stages. Every early return is a block decision."""
    settings = request.app.state.settings
    audit_store: AuditStore = request.app.state.audit
    spend: SpendTracker = request.app.state.spend
    client: httpx.AsyncClient = request.app.state.http
    cipher: ContentCipher = request.app.state.cipher

    call = await new_call(request, agent_id, key_id)

    # Ahead of every other stage that reads the body, classification included: on this
    # path there is no readable body, so anything reading it reads nothing — and then
    # records that nothing as a finding.
    blocked = screen_content_encoding(call, audit_store)
    if blocked is not None:
        return blocked

    classify_and_route(call, settings)

    prepare_body(call, settings)

    # A minted key never leaves the gateway — fail closed before the forward path can
    # build headers for a keyless non-local provider.
    blocked = screen_upstream_credentials(call, settings, audit_store)
    if blocked is not None:
        return blocked

    blocked = inspect_tool_definitions(call, audit_store)
    if blocked is not None:
        return blocked

    blocked = await run_injection_scan(call, settings, audit_store)
    if blocked is not None:
        return blocked

    blocked = await check_spend(call, spend, audit_store)
    if blocked is not None:
        return blocked

    return await forward_and_stream(call, client, settings, audit_store, spend, cipher)


# Accept both with and without the /v1 prefix: OpenAI-compat clients append
# "/chat/completions" to the configured baseUrl, so whether the version segment appears
# depends on whether baseUrl includes /v1. Tolerate both. Untagged = no agent attribution.
@router.post("/v1/chat/completions")
@router.post("/chat/completions")
async def chat_completions(request: Request) -> Response:
    return await handle_chat(request, agent_id=None)


# Agent-tagged variant: an agent's baseUrl is configured as ".../a/<agent_id>", so its
# requests arrive here and carry attribution (the audit row's agent_id and any per-agent
# routing rule). The {agent_id} segment is the only way to identify the agent —
# clients send vanilla OpenAI Chat Completions with no agent field.
@router.post("/a/{agent_id}/v1/chat/completions")
@router.post("/a/{agent_id}/chat/completions")
async def chat_completions_for_agent(request: Request, agent_id: str) -> Response:
    return await handle_chat(request, agent_id=agent_id)
