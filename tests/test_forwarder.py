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
from agentgate.proxy.forwarder import forward_stream
from agentgate.proxy.streaming import StreamTap
from bench.mock_upstream import app as mock_app
from bench.mock_upstream import canned_sse


@pytest.fixture
def mock_client() -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=mock_app)
    return httpx.AsyncClient(transport=transport, base_url="http://mock")


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
    from agentgate.proxy.forwarder import build_upstream_url

    assert (
        build_upstream_url(provider)
        == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    )


def test_auth_header_passthrough():
    from agentgate.config import Provider
    from agentgate.proxy.forwarder import prepare_headers

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
    from agentgate.config import Provider
    from agentgate.proxy.forwarder import prepare_headers

    local = Provider(name="local", base_url="http://127.0.0.1:8000", is_local=True, api_key="local")
    out = prepare_headers(
        {"x-goog-api-key": "AIzaX", "content-type": "application/json"},
        local,
    )
    assert out["authorization"] == "Bearer local"
    assert "x-goog-api-key" not in out


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
