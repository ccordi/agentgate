"""SSE passthrough + async response tap.

The gateway streams the upstream SSE response straight back to the client (no
buffering — keeps the p99 benchmark clean), while *teeing* the same bytes into a
tap that extracts usage and finish_reason for accounting.

Response content is NOT scanned: the tap reads only model, usage, finish_reason and
tool-call count. All screening — and all blocking — happens on the inbound request side.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

# Cap on a buffered (non-streaming) response held for accounting. Streaming responses
# drain line by line and never accumulate.
_MAX_BUFFERED_RESPONSE_BYTES = 8 * 1024 * 1024

# A sender must not emit one (RFC 8259 §8.1) but a parser may ignore it, and Python's
# `json.loads` does for bytes input. Skipped before the shape sniff below so a BOM cannot
# make a JSON completion look like SSE.
_BOM = b"\xef\xbb\xbf"

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


def _non_negative(value: object) -> int:
    """Upstream-reported token count, floored at zero (absent/garbage -> 0)."""
    return max(0, value) if isinstance(value, int) and not isinstance(value, bool) else 0


class StreamTap:
    """Incrementally parses OpenAI SSE chunks to extract accounting signals.

    Feed every raw byte chunk to ``feed``; read ``result`` after the stream closes.
    Parsing is best-effort and never raises into the forwarding path — a parse miss
    just means slightly less metadata, never a broken stream.
    """

    def __init__(self, *, sse: bool | None = True) -> None:
        self.result = StreamResult()
        # A non-streaming response is one JSON object, not `data:`-framed events, so it
        # is accumulated whole and parsed at close instead of line by line.
        #
        # `None` means "decide from the bytes". Keying the mode off the upstream's
        # `content-type` alone made a header the accounting depended on: an SSE upstream
        # that omits it — or labels it `application/json` — parsed as neither, and the
        # request accrued no spend at all while the client got its stream. A completion is
        # always a JSON object, so the first non-blank byte separates the two shapes
        # without trusting anyone.
        self._sse = sse
        self._buf = b""

    def _trim(self) -> None:
        """Hold no more than the cap.

        This is the one path that holds the body rather than draining it, and the size is
        the upstream's choice. Trimmed to the cap rather than stopping short of it, so one
        oversized chunk cannot carry the buffer past the bound it names. Past the cap
        accounting is lost, not the response — the bytes still stream through untouched.
        """
        if len(self._buf) > _MAX_BUFFERED_RESPONSE_BYTES:
            self._buf = self._buf[:_MAX_BUFFERED_RESPONSE_BYTES]

    def feed(self, chunk: bytes) -> None:
        self._buf += chunk
        if self._sse is None:
            head = self._buf.lstrip()
            # The transport may split the three-byte BOM anywhere. While all bytes seen
            # so far are a BOM prefix, there is still no shape byte to judge; deciding SSE
            # on the first `\xef` loses accounting solely because of chunk boundaries.
            if head and len(head) < len(_BOM) and _BOM.startswith(head):
                self._trim()
                return
            if head.startswith(_BOM):
                # JSON whitespace is legal after the ignored BOM. Strip it again: the
                # first lstrip handled only whitespace that preceded the BOM.
                head = head[len(_BOM):].lstrip()
            if not head:
                # Leading whitespace (or a lone BOM) — no first byte to judge on yet. The
                # buffer is being HELD, not drained, so it is capped here too: an upstream
                # that sends nothing else must not grow it past the bound while the mode
                # is still undecided. Returning before the trim was a live cap bypass.
                self._trim()
                return
            self._sse = not head.startswith((b"{", b"["))
        if not self._sse:
            self._trim()
            return
        # SSE events are newline-delimited; process complete lines, keep remainder.
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            self._parse_line(line.strip())

    def close(self) -> None:
        """Parse whatever is left in the buffer.

        Streaming: the trailing line, when it has no newline after it. OpenAI terminates
        the final event with one, but some local servers (llama.cpp, oMLX) omit it on the
        last frame before closing — without this the final `data: {...usage}` event would
        sit unparsed and usage/finish_reason would be lost.

        Non-streaming: the entire response body, which is where its usage lives.
        """
        if not self._buf:
            return
        if self._sse:
            self._parse_line(self._buf.strip())
        else:
            self._ingest_completion(self._buf)
        self._buf = b""

    def _ingest_completion(self, raw: bytes) -> None:
        """A whole non-streamed Chat Completion: same accounting fields, different shape."""
        try:
            event = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            return
        if not isinstance(event, dict):
            return
        # model, usage and finish_reason sit in the same places; _ingest reads
        # `delta.tool_calls`, which a completion does not have, so it counts nothing here.
        self._ingest(event)
        for choice in event.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if not isinstance(message, dict):
                continue
            tcs = message.get("tool_calls")
            if not isinstance(tcs, list):
                continue
            # Complete in one object rather than assembled across deltas, so each entry
            # carrying a function name is exactly one call.
            for tc in tcs:
                if isinstance(tc, dict) and (tc.get("function") or {}).get("name"):
                    self.result.tool_call_count += 1

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
            # Clamped at zero: token counts come off the upstream's wire, and a negative
            # value flows through estimate_cost_usd into spend.record, *decrementing*
            # accrued spend and pushing the key back under its cap.
            self.result.prompt_tokens = _non_negative(usage.get("prompt_tokens"))
            self.result.completion_tokens = _non_negative(usage.get("completion_tokens"))
            self.result.total_tokens = _non_negative(usage.get("total_tokens"))

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
