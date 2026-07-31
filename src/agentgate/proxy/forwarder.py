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
#
# The hop-by-hop set is RFC 7230 §6.1: these describe the *client-to-gateway* connection
# and are meaningless — or actively harmful — on the gateway-to-upstream one. Forwarding
# `transfer-encoding: chunked` alongside the `content-length` httpx computes is the
# classic CL+TE desync pair, and the process shares one AsyncClient, so a desynced pooled
# connection would contaminate other callers' requests rather than just the sender's.
# `proxy-authorization` is the client's own proxy credential and has no business reaching
# a model provider.
#
# `content-encoding` is not here because it is already gone by this point: `new_call`
# decompresses the body and pops the header off the request it hands us, and rejects the
# request outright when it cannot decode what the header declares. So there is never a
# compressed body to describe here — a static entry would be stripping a header that the
# only bodies reaching this function do not carry.
_STRIP_REQUEST_HEADERS = {
    "host", "content-length", "accept-encoding",
    # RFC 7230 §6.1 hop-by-hop
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade",
}
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

    async with client.stream(
        "POST", url, headers=headers, content=body, timeout=timeout_s
    ) as resp:
        # Everything is piped through the tap, streamed or not. A client that omits
        # `stream` — which is OpenAI's default — gets a single JSON completion back, and
        # the SSE line parser finds no `data:` prefix in it, so usage never lands and the
        # request accrues zero spend.
        content_type = (resp.headers.get("content-type") or "").lower()
        # A declared SSE stream is taken at its word; anything else — including no header
        # at all — is decided from the first bytes. Trusting the header alone made the
        # accounting depend on it: an upstream that streams SSE without saying so, or
        # labels it `application/json`, metered nothing and accrued no spend.
        tap = StreamTap(sse=True if "text/event-stream" in content_type else None)
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
