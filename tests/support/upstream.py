"""Mock upstream handlers for the endpoint tests.

One factory covers every variant the tests need: record the forwarded
requests, log the forwarded bodies (egress proof — the gateway redacts *before*
forwarding, so what lands here is what a cloud upstream would see), or just answer
with the canonical stream.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx

from bench.mock_upstream import canned_sse_bytes

SSE_HEADERS = {"content-type": "text/event-stream"}


def sse_handler(
    *,
    record: list[httpx.Request] | None = None,
    body_log: list[bytes] | None = None,
    status: int = 200,
) -> Callable[[httpx.Request], httpx.Response]:
    """Build an `httpx.MockTransport` handler answering with the canonical SSE stream.

    `record` collects the forwarded requests; `body_log` collects their raw bodies.
    The stream is `bench.mock_upstream`'s canned fixture, so the usage chunk (11
    prompt / 6 completion tokens) is the same one a load test against the mock sees.
    """
    def handler(request: httpx.Request) -> httpx.Response:
        if record is not None:
            record.append(request)
        if body_log is not None:
            body_log.append(request.content)
        return httpx.Response(status, content=canned_sse_bytes(), headers=SSE_HEADERS)

    return handler
