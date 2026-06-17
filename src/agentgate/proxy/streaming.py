"""SSE passthrough + async response tap.

The gateway streams the upstream SSE response straight back to the client (no
buffering — keeps the p99 benchmark clean), while *teeing* the same bytes into a
tap that extracts usage and finish_reason for accounting.

Outbound assistant-text scanning is detection-only and async; it never blocks the
stream. (Security-critical blocking happens on the inbound request side.)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class StreamResult:
    """What the tap learns from an SSE response, available once the stream ends."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reasons: list[str] = field(default_factory=list)
    tool_call_count: int = 0
    upstream_model: str | None = None

    @property
    def had_tool_calls(self) -> bool:
        return "tool_calls" in self.finish_reasons


class StreamTap:
    """Incrementally parses OpenAI SSE chunks to extract accounting signals.

    Feed every raw byte chunk to ``feed``; read ``result`` after the stream closes.
    Parsing is best-effort and never raises into the forwarding path — a parse miss
    just means slightly less metadata, never a broken stream.
    """

    def __init__(self) -> None:
        self.result = StreamResult()
        self._buf = b""

    def feed(self, chunk: bytes) -> None:
        self._buf += chunk
        # SSE events are newline-delimited; process complete lines, keep remainder.
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            self._parse_line(line.strip())

    def close(self) -> None:
        """Flush any trailing line not terminated by a newline. OpenAI terminates the
        final event with a newline, but some local servers (llama.cpp, oMLX) omit it on
        the last frame before closing — without this, the final `data: {...usage}` event
        would sit unparsed in the buffer and usage/finish_reason would be lost."""
        if self._buf:
            self._parse_line(self._buf.strip())
            self._buf = b""

    def _parse_line(self, line: bytes) -> None:
        if not line.startswith(b"data:"):
            return
        payload = line[len(b"data:") :].strip()
        if not payload or payload == b"[DONE]":
            return
        try:
            event = json.loads(payload)
        except (ValueError, UnicodeDecodeError):
            return
        self._ingest(event)

    def _ingest(self, event: dict) -> None:
        if isinstance(event.get("model"), str):
            self.result.upstream_model = event["model"]

        # Usage arrives in the final chunk when stream_options.include_usage is set.
        usage = event.get("usage")
        if isinstance(usage, dict):
            self.result.prompt_tokens = usage.get("prompt_tokens", 0) or 0
            self.result.completion_tokens = usage.get("completion_tokens", 0) or 0
            self.result.total_tokens = usage.get("total_tokens", 0) or 0

        for choice in event.get("choices") or []:
            fr = choice.get("finish_reason")
            if fr:
                self.result.finish_reasons.append(fr)
            delta = choice.get("delta") or {}
            tcs = delta.get("tool_calls")
            if isinstance(tcs, list):
                # Each tool call streams across multiple deltas; count only the ones
                # that announce a new call (carry an index with a function name).
                for tc in tcs:
                    if (tc.get("function") or {}).get("name"):
                        self.result.tool_call_count += 1
