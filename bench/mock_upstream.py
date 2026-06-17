"""Canned-SSE mock upstream.

Emits a deterministic OpenAI Chat Completions SSE stream — no token cost, no
network — so the load benchmark measures pure gateway overhead (p99) and the
forwarder/tap can be tested end-to-end. Mirrors the chunk shape of a real
OpenAI-compatible streaming SSE response.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

DEFAULT_TOKENS = ["Hello", " from", " the", " mock", " upstream", "."]


def _chunk(**choice_fields) -> bytes:
    event = {
        "id": "chatcmpl-mock",
        "object": "chat.completion.chunk",
        "model": "mock-model",
        "choices": [{"index": 0, **choice_fields}],
    }
    return f"data: {json.dumps(event)}\n\n".encode()


def _usage_chunk(prompt: int, completion: int) -> bytes:
    event = {
        "id": "chatcmpl-mock",
        "object": "chat.completion.chunk",
        "model": "mock-model",
        "choices": [],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
    }
    return f"data: {json.dumps(event)}\n\n".encode()


async def canned_sse(tokens: list[str] | None = None) -> AsyncIterator[bytes]:
    tokens = tokens or DEFAULT_TOKENS
    yield _chunk(delta={"role": "assistant", "content": ""}, finish_reason=None)
    for tok in tokens:
        yield _chunk(delta={"content": tok}, finish_reason=None)
    yield _chunk(delta={}, finish_reason="stop")
    yield _usage_chunk(prompt=11, completion=len(tokens))
    yield b"data: [DONE]\n\n"


app = FastAPI(title="mock-upstream")


@app.post("/v1/chat/completions")
@app.post("/v1beta/openai/chat/completions")
async def chat_completions(request: Request) -> StreamingResponse:
    # Body is accepted but ignored for response purposes — output is deterministic by
    # design. When MOCK_LOG_BODIES is set (a file path), append the raw body received
    # (one JSON line) — this serves as egress proof (the gateway redacts
    # before forwarding, so what lands here is what the cloud upstream would see).
    body = await request.body()
    log_path = os.environ.get("MOCK_LOG_BODIES")
    if log_path:
        with open(log_path, "a") as f:
            f.write(body.decode("utf-8", errors="replace") + "\n")
    return StreamingResponse(canned_sse(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=4200)
