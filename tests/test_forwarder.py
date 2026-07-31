"""End-to-end forwarder tests: gateway -> mock upstream, both over ASGI (no network).

Verifies two core invariants: SSE streams through byte-for-byte,
and the tap extracts usage / finish_reason / tool-call signals from the stream.
"""

from __future__ import annotations

import gzip
import json

import httpx
import pytest
from fastapi import FastAPI, Response

from agentgate.app import app as gateway_app
from agentgate.config import Provider
from agentgate.proxy.forwarder import build_upstream_url, forward_stream, prepare_headers
from agentgate.proxy.streaming import StreamTap
from bench.mock_upstream import app as mock_app
from bench.mock_upstream import canned_sse


@pytest.fixture
async def mock_client() -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=mock_app)
    client = httpx.AsyncClient(transport=transport, base_url="http://mock")
    try:
        yield client
    finally:
        await client.aclose()


async def test_forward_streams_and_taps_usage(mock_client: httpx.AsyncClient):
    provider = Provider(name="mock", base_url="http://mock")
    chunks: list[bytes] = []
    async with forward_stream(
        mock_client, provider, inbound_headers={"authorization": "Bearer x"}, body=b"{}", timeout_s=10
    ) as upstream:
        assert upstream.status_code == 200
        async for chunk in upstream.body:
            chunks.append(chunk)

    body = b"".join(chunks)
    # Streamed through intact, including the SSE terminator.
    assert b"data: [DONE]" in body
    # Reconstruct assistant text from the streamed deltas (tokens arrive separately).
    text = ""
    for line in body.decode().split("\n\n"):
        line = line.strip()
        if not line.startswith("data:") or "[DONE]" in line:
            continue
        ev = json.loads(line.removeprefix("data:").strip())
        for ch in ev.get("choices", []):
            text += (ch.get("delta") or {}).get("content") or ""
    assert text == "Hello from the mock upstream."

    # Tap extracted accounting signals from the usage + finish chunks.
    r = upstream.result
    assert r.prompt_tokens == 11
    assert r.completion_tokens == len(["Hello", " from", " the", " mock", " upstream", "."])
    assert r.finish_reasons == ["stop"]
    assert r.upstream_model == "mock-model"
    assert not r.had_tool_calls


async def test_gemini_path_rewrite():
    provider = Provider(
        name="gemini",
        base_url="https://generativelanguage.googleapis.com",
        chat_completions_path="/v1beta/openai/chat/completions",
    )
    assert (
        build_upstream_url(provider)
        == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    )


def test_auth_header_passthrough():
    cloud = Provider(name="gemini", base_url="https://example.com")
    out = prepare_headers(
        {"Authorization": "Bearer secret", "host": "gw", "content-length": "10", "x-goog-api-key": "AIzaX"},
        cloud,
    )
    assert out["Authorization"] == "Bearer secret"
    assert out["x-goog-api-key"] == "AIzaX"
    assert "host" not in out
    assert "content-length" not in out


def test_auth_header_local_override():
    local = Provider(name="local", base_url="http://127.0.0.1:8000", is_local=True, api_key="local")
    out = prepare_headers(
        {"x-goog-api-key": "AIzaX", "content-type": "application/json"},
        local,
    )
    assert out["authorization"] == "Bearer local"
    assert "x-goog-api-key" not in out


def test_hop_by_hop_headers_are_not_forwarded():
    """RFC 7230 §6.1 headers describe the inbound connection, not the upstream one.

    `transfer-encoding` beside the `content-length` httpx computes is the CL+TE
    desync pair, and one AsyncClient is shared process-wide, so a poisoned pooled
    connection would reach other callers. `proxy-authorization` is the client's own
    proxy credential and must not reach a model provider.
    """
    cloud = Provider(name="gemini", base_url="https://example.invalid", is_local=False)
    out = prepare_headers(
        {
            "content-type": "application/json",
            "transfer-encoding": "chunked",
            "te": "trailers",
            "trailer": "X-Foo",
            "upgrade": "websocket",
            "keep-alive": "timeout=5",
            "proxy-authorization": "Basic abc",
            "proxy-authenticate": "Basic",
            "connection": "keep-alive",
            "host": "127.0.0.1:4100",
            "content-length": "13",
        },
        cloud,
    )
    for stripped in ("transfer-encoding", "te", "trailer", "upgrade", "keep-alive",
                     "proxy-authorization", "proxy-authenticate", "connection",
                     "host", "content-length"):
        assert stripped not in out, f"{stripped} must not reach the upstream"
    assert out["content-type"] == "application/json", "ordinary headers still pass through"


def test_content_encoding_is_still_forwarded():
    """The body goes upstream as received, so the header describing it must too.

    Stripping it here would misdescribe the bytes. Whether an encoded body should be
    accepted at all is a separate question from header hygiene.
    """
    cloud = Provider(name="gemini", base_url="https://example.invalid", is_local=False)
    out = prepare_headers({"content-encoding": "gzip"}, cloud)
    assert out["content-encoding"] == "gzip"


def test_tap_counts_tool_calls():
    tap = StreamTap()
    tap.feed(b'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"exec","arguments":""}}]}}]}\n\n')
    tap.feed(b'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]},"finish_reason":"tool_calls"}]}\n\n')
    tap.feed(b"data: [DONE]\n\n")
    assert tap.result.tool_call_count == 1
    assert tap.result.had_tool_calls


async def test_forward_decodes_gzipped_upstream():
    """Regression: a gzip-compressed upstream response must reach the client as
    *decoded* SSE (we strip content-encoding), and the tap must still parse it.

    Caught in production: aiter_raw() forwarded compressed bytes with the
    content-encoding header stripped -> client got undecodable data
    ('incomplete_result') and the tap saw model=None/tokens=0. aiter_bytes() fixes both.
    """
    sse = b""
    async for c in canned_sse():
        sse += c

    gz_app = FastAPI()

    @gz_app.post("/v1/chat/completions")
    async def _gz() -> Response:
        return Response(
            content=gzip.compress(sse),
            media_type="text/event-stream",
            headers={"content-encoding": "gzip"},
        )

    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=gz_app), base_url="http://gz")
    provider = Provider(name="gz", base_url="http://gz")
    chunks: list[bytes] = []
    async with forward_stream(
        client, provider, inbound_headers={}, body=b"{}", timeout_s=10
    ) as upstream:
        # content-encoding must not be forwarded (body is decoded downstream).
        assert "content-encoding" not in {k.lower() for k in upstream.headers}
        async for chunk in upstream.body:
            chunks.append(chunk)

    body = b"".join(chunks)
    assert b"data: [DONE]" in body          # client receives decoded SSE, not gzip
    assert upstream.result.completion_tokens == 6   # tap parsed the decoded stream


def test_chat_completions_route_accepts_both_paths():
    """Regression: some clients send /chat/completions (no /v1) when baseUrl lacks the
    version segment. Both path forms must be registered."""
    paths = {r.path for r in gateway_app.routes}
    assert "/v1/chat/completions" in paths
    assert "/chat/completions" in paths


async def test_canned_sse_shape():
    """Guards the mock's chunk shape against drift from the real OpenAI SSE format."""
    out = b""
    async for c in canned_sse():
        out += c
    events = [line for line in out.decode().split("\n\n") if line.strip()]
    assert events[-1] == "data: [DONE]"
    first = json.loads(events[0].removeprefix("data: "))
    assert first["choices"][0]["delta"]["role"] == "assistant"


def test_tap_reads_usage_from_a_non_streamed_completion():
    """A client that omits `stream` gets one JSON object, not `data:`-framed events.

    The SSE line parser finds no `data:` prefix anywhere in it, so usage never landed:
    the request recorded tok=0/0 cost=0.0 and the per-key USD cap did not apply to
    anyone who left `stream` at its OpenAI default of false.
    """
    tap = StreamTap(sse=False)
    tap.feed(json.dumps({
        "model": "gpt-4",
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant",
            "tool_calls": [{"id": "c1", "function": {"name": "exec", "arguments": "{}"}}],
        }}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 6, "total_tokens": 17},
    }).encode())
    tap.close()

    assert tap.result.prompt_tokens == 11
    assert tap.result.completion_tokens == 6
    assert tap.result.upstream_model == "gpt-4"
    assert tap.result.tool_call_count == 1  # from `message`, not `delta`
    assert tap.result.had_tool_calls


def test_tap_in_sse_mode_does_not_parse_a_bare_json_body():
    """Why the mode flag exists: in SSE mode an unframed body yields nothing."""
    tap = StreamTap()  # sse=True, the default
    tap.feed(b'{"usage":{"prompt_tokens":11,"completion_tokens":6}}')
    tap.close()
    assert tap.result.prompt_tokens == 0


def test_tap_meters_sse_whatever_the_upstream_calls_it():
    """Accounting must not depend on the upstream labelling its stream correctly.

    Mode was taken from `content-type` alone, so an SSE upstream that omits the header —
    or calls it `application/json` — parsed as neither shape: tok=0/0, model=None, and
    the request accrued no spend while the client got a perfectly good stream. Omitting
    it is the regression that matters, because before the tap had a mode at all it always
    ran the SSE parser and got this right.
    """
    from agentgate.proxy.streaming import StreamTap

    sse = (b'data: {"model":"m","usage":{"prompt_tokens":11,"completion_tokens":5}}\n\n'
           b'data: [DONE]\n\n')
    completion = json.dumps({
        "model": "m",
        "usage": {"prompt_tokens": 11, "completion_tokens": 5},
        "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
    }).encode()

    for label, mode, payload in [
        ("sse, declared", True, sse),
        ("sse, no content-type", None, sse),
        ("sse, mislabelled json", None, sse),
        ("completion, sniffed", None, completion),
        ("completion, declared", False, completion),
    ]:
        tap = StreamTap(sse=mode)
        for i in range(0, len(payload), 7):  # chunked, so the sniff sees a partial head
            tap.feed(payload[i:i + 7])
        tap.close()
        assert tap.result.prompt_tokens == 11, f"{label}: prompt tokens lost"
        assert tap.result.completion_tokens == 5, f"{label}: completion tokens lost"
        assert tap.result.upstream_model == "m", f"{label}: model lost"


def test_buffered_tap_never_exceeds_its_cap():
    """One oversized chunk must not carry the buffer past the bound the constant names."""
    from agentgate.proxy.streaming import _MAX_BUFFERED_RESPONSE_BYTES, StreamTap

    tap = StreamTap(sse=False)
    for _ in range(12):
        tap.feed(b"{" + b"x" * (1024 * 1024))
    assert len(tap._buf) <= _MAX_BUFFERED_RESPONSE_BYTES

    # And while the mode is still UNDECIDED, which is where the cap was not enforced at
    # all: whitespace gives the sniff no first byte to judge on, and the early return
    # that waits for one skipped the trim, so an upstream sending nothing but whitespace
    # grew the buffer without bound — past the very cap this test exists to pin.
    undecided = StreamTap(sse=None)
    for _ in range(12):
        undecided.feed(b" " * (1024 * 1024))
    assert undecided._sse is None, "still undecided — that is the case under test"
    assert len(undecided._buf) <= _MAX_BUFFERED_RESPONSE_BYTES


def test_bom_prefixed_completion_is_not_mistaken_for_sse():
    """A byte-order mark must not cost a request its accounting.

    The shape sniff reads the first non-blank byte, and a BOM is not `{`, so an ordinary
    JSON completion was parsed as SSE: no `data:` line ever arrived and the request
    recorded zero tokens. A sender should not emit one (RFC 8259 §8.1) but a parser may
    ignore it — `json.loads` already does, so the BOM broke nothing except the sniff.
    """
    completion = json.dumps({
        "model": "m",
        "usage": {"prompt_tokens": 11, "completion_tokens": 6},
        "choices": [{"finish_reason": "stop"}],
    }).encode()

    cases = {
        "one chunk": [b"\xef\xbb\xbf" + completion],
        "BOM split": [b"\xef", b"\xbb", b"\xbf", completion],
        "whitespace after BOM": [b"\xef\xbb\xbf", b" \r\n\t", completion],
        "every byte split": [bytes([b]) for b in b"\xef\xbb\xbf \n" + completion],
    }
    for label, chunks in cases.items():
        tap = StreamTap(sse=None)
        for chunk in chunks:
            tap.feed(chunk)
        tap.close()

        assert tap._sse is False, f"{label}: a BOM-prefixed completion is JSON, not SSE"
        assert (tap.result.prompt_tokens, tap.result.completion_tokens) == (11, 6), label
        assert tap.result.upstream_model == "m", label
