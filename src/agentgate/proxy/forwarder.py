"""httpx streaming forwarder.

Forwards an OpenAI Chat Completions request to the chosen upstream, rewriting the
path for Gemini and passing the auth header through unchanged. Streams the SSE
response back to the client while teeing it into a StreamTap for accounting.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import httpx

from agentgate.config import Provider
from agentgate.proxy.streaming import StreamResult, StreamTap

# Hop-by-hop and length/host headers we must not forward verbatim; httpx recomputes
# the ones it needs for the new connection.
_STRIP_REQUEST_HEADERS = {"host", "content-length", "connection", "accept-encoding"}
_STRIP_RESPONSE_HEADERS = {"content-length", "content-encoding", "transfer-encoding", "connection"}


@dataclass
class UpstreamStream:
    """A live upstream response: status + headers known, body not yet consumed."""

    status_code: int
    headers: dict[str, str]
    body: AsyncIterator[bytes]
    result: StreamResult = field(default_factory=StreamResult)


def build_upstream_url(provider: Provider) -> str:
    return provider.base_url.rstrip("/") + provider.chat_completions_path


def prepare_headers(inbound: dict[str, str], provider: Provider) -> dict[str, str]:
    """Pass auth through unchanged; drop hop-by-hop headers.

    Auth is a plain header (`x-goog-api-key` or `Authorization: Bearer`) with no
    signing, so a straight copy is correct. When the provider specifies an api_key
    override (local servers that require a Bearer token), inject it and strip the
    inbound Gemini key.
    """
    out = {k: v for k, v in inbound.items() if k.lower() not in _STRIP_REQUEST_HEADERS}
    if provider.api_key is not None:
        out.pop("x-goog-api-key", None)
        out["authorization"] = f"Bearer {provider.api_key}"
    return out


@asynccontextmanager
async def forward_stream(
    client: httpx.AsyncClient,
    provider: Provider,
    inbound_headers: dict[str, str],
    body: bytes,
    timeout_s: float,
) -> AsyncIterator[UpstreamStream]:
    """Open the upstream stream as a context manager.

    On enter, the upstream status and headers are known. The yielded object's
    ``body`` iterator streams SSE bytes to the client and feeds the tap as a side
    effect; ``result`` is fully populated once ``body`` is exhausted.
    """
    url = build_upstream_url(provider)
    headers = prepare_headers(inbound_headers, provider)
    tap = StreamTap()

    async with client.stream(
        "POST", url, headers=headers, content=body, timeout=timeout_s
    ) as resp:
        out_headers = {
            k: v for k, v in resp.headers.items() if k.lower() not in _STRIP_RESPONSE_HEADERS
        }

        async def body_iter() -> AsyncIterator[bytes]:
            # aiter_bytes() yields the *decoded* body (httpx undoes any gzip/br it
            # negotiated and de-chunks), still streamed incrementally. We must use
            # this — not aiter_raw() — because we strip content-encoding/length/
            # transfer-encoding from the response headers, so the client expects
            # plain decoded SSE. aiter_raw() would forward compressed bytes with no
            # content-encoding header -> client can't decode -> "incomplete_result".
            async for chunk in resp.aiter_bytes():
                tap.feed(chunk)
                yield chunk
            # Parse any final event not terminated by a trailing newline (some local
            # SSE servers omit it on the last frame) so usage/finish_reason aren't lost.
            tap.close()

        yield UpstreamStream(
            status_code=resp.status_code,
            headers=out_headers,
            body=body_iter(),
            result=tap.result,
        )
