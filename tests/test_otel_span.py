"""The per-request OTel ``chat`` span: audit-row metadata on it, message content never.

The span-asserting tests need the `tracing` extra and skip without it:
`uv run --extra tracing pytest tests/test_otel_span.py`. The no-op test runs anywhere.
"""

from __future__ import annotations

import pytest

from agentgate.observability import otel
from agentgate.tasks import drain_background
from tests.support import HARD_INJECTION
from tests.support.audit import wait_for_audit_row

STREAMING = {"stream": True, "stream_options": {"include_usage": True}}
AUTH = {"authorization": "Bearer t"}
# A distinctive user message: the privacy assertion greps every exported span for it.
MARKER = "the moss on the north face of the seawall"


def test_tracing_calls_do_nothing_when_tracing_is_inactive(monkeypatch):
    monkeypatch.setattr(otel, "_ACTIVE", False)
    assert otel.capture_parent() is None
    otel.record_chat(None, model="m", duration_ms=12.0, attributes={"agentgate.status": 200})


@pytest.fixture
def span_exporter():
    """An in-memory exporter attached to the process-global tracer provider."""
    trace_sdk = pytest.importorskip("opentelemetry.sdk.trace")
    from opentelemetry import trace
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    provider = trace.get_tracer_provider()
    if not isinstance(provider, trace_sdk.TracerProvider):
        provider = trace_sdk.TracerProvider()
        trace.set_tracer_provider(provider)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield exporter
    exporter.clear()


def test_record_chat_backdates_and_parents_the_span(span_exporter, monkeypatch):
    from opentelemetry import trace

    monkeypatch.setattr(otel, "_ACTIVE", True)
    with trace.get_tracer("test").start_as_current_span("server") as server:
        parent = otel.capture_parent()
    otel.record_chat(
        parent, model="m", duration_ms=250.0,
        attributes={"gen_ai.usage.input_tokens": 11, "agentgate.dropped": None},
    )

    span = {s.name: s for s in span_exporter.get_finished_spans()}["chat m"]
    server_ctx = server.get_span_context()
    assert span.context.trace_id == server_ctx.trace_id
    assert span.parent is not None and span.parent.span_id == server_ctx.span_id
    assert span.attributes["gen_ai.usage.input_tokens"] == 11
    assert "agentgate.dropped" not in span.attributes  # None values are dropped
    assert span.end_time - span.start_time == 250_000_000  # backdated by duration_ms


async def test_forwarded_request_span_carries_usage_and_never_content(gateway, span_exporter):
    r = await gateway.client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": MARKER}], **STREAMING},
        headers=AUTH,
    )
    assert r.status_code == 200

    chats = [s for s in span_exporter.get_finished_spans() if s.name == "chat m"]
    assert len(chats) == 1
    attrs = chats[0].attributes
    assert attrs["gen_ai.usage.input_tokens"] == 11  # the canned mock-upstream usage chunk
    assert attrs["gen_ai.usage.output_tokens"] == 6
    assert attrs["agentgate.status"] == 200
    assert attrs["agentgate.injection.flagged"] is False
    assert chats[0].parent is not None  # joined the server span's trace

    # The privacy pin: message text appears on no span, under no key, on no code path.
    for s in span_exporter.get_finished_spans():
        for key, value in (s.attributes or {}).items():
            assert MARKER not in str(value), (s.name, key)


async def test_blocked_request_emits_span_with_the_verdict(gateway, span_exporter):
    r = await gateway.client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": HARD_INJECTION}], **STREAMING},
        headers=AUTH,
    )
    assert r.status_code == 400
    await drain_background()  # the rejected-path span is emitted from a spawned task

    chats = [s for s in span_exporter.get_finished_spans() if s.name == "chat m"]
    assert len(chats) == 1
    attrs = chats[0].attributes
    assert attrs["agentgate.status"] == 400
    assert attrs["agentgate.injection.hard"] is True
    assert "gen_ai.usage.input_tokens" not in attrs  # nothing was forwarded
    for s in span_exporter.get_finished_spans():
        for key, value in (s.attributes or {}).items():
            assert HARD_INJECTION not in str(value), (s.name, key)


async def test_rejected_row_id_is_the_request_id(gateway, span_exporter):
    """A rejected request's audit row keys by the request's own id, like the completion path.

    If the rejected path did not pass ``row_id``, blocked and rejected requests would get
    a fresh id, and their audit rows could not be joined to their spans or content samples.
    """
    r = await gateway.client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": HARD_INJECTION}], **STREAMING},
        headers=AUTH,
    )
    assert r.status_code == 400
    await drain_background()

    chats = [s for s in span_exporter.get_finished_spans() if s.name == "chat m"]
    row = await wait_for_audit_row(gateway.store)
    assert row is not None and row.status == 400
    assert str(row.id) == chats[0].attributes["agentgate.request_id"]
