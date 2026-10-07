"""OpenTelemetry tracing — off by default; live only with the ``tracing`` extra installed.

Imports are lazy, so without the extra the gateway runs identically: ``setup_tracing``,
``capture_parent`` and ``record_chat`` are no-ops. With it, each request gets the FastAPI
server span plus one ``chat`` span carrying the audit row's metadata fields as
GenAI-semconv attributes — never message content. The extra carries the OTLP exporter, so
``AGENTGATE_OTLP_ENDPOINT`` (environment or `.env`) exports spans to any OTLP/HTTP sink.

**The audit DB stays the system of record**: span attributes mirror the row's fields, and
the published per-stage latency figures come from the DB's `latency_*_ms`
columns, not from spans.

    uv sync --extra tracing     # to actually get spans
"""

from __future__ import annotations

import logging
import time
from typing import Any

from agentgate.config import env_setting

log = logging.getLogger("agentgate.otel")

# Three states in one flag: None = setup has not run, False = it ran but the extra is
# absent (no tracer provider), True = an SDK tracer provider is installed. `capture_parent`
# and `record_chat` only care about True, so they test it directly.
_ACTIVE: bool | None = None


def otlp_endpoint() -> str | None:
    """``AGENTGATE_OTLP_ENDPOINT`` from the environment or `.env`; unset or empty means
    spans are not exported."""
    return env_setting("AGENTGATE_OTLP_ENDPOINT")


def setup_tracing(app) -> None:
    global _ACTIVE
    if _ACTIVE is not None:  # idempotent
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        log.debug("opentelemetry not installed (extra 'tracing'); tracing disabled")
        _ACTIVE = False
        return

    provider = TracerProvider(resource=Resource.create({"service.name": "agentgate"}))

    endpoint = otlp_endpoint()
    if endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
            log.info("OTLP span export -> %s", endpoint)
        except ImportError:  # only on a partial hand-install; the extra ships the exporter
            log.warning("OTLP exporter not installed; spans dropped")

    trace.set_tracer_provider(provider)
    _ACTIVE = True

    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
    except Exception:  # noqa: BLE001
        log.exception("FastAPI OTel instrumentation failed")


def capture_parent() -> object | None:
    """The current request's server-span context, taken while it is still current.

    Opaque to callers. Captured eagerly at request entry because the pipeline finishes
    inside the response generator, where that span is no longer reliably current.
    """
    if not _ACTIVE:
        return None
    from opentelemetry import trace

    ctx = trace.get_current_span().get_span_context()
    return ctx if ctx.is_valid else None


def record_chat(
    parent: object | None,
    *,
    model: str | None,
    duration_ms: float,
    attributes: dict[str, Any],
) -> None:
    """Emit the per-request ``chat`` span, backdated to cover the whole request.

    ``attributes`` are the audit row's metadata fields — never message content; ``None``
    values are dropped. Parented under ``parent`` when given, so the span joins the
    server span's trace.
    """
    if not _ACTIVE:
        return
    # Tracing must never break the caller. Both call sites sit in the request's accounting
    # path, ahead of the spend and audit writes, so an exporter that raises would other-
    # wise cost the audit row — losing the system of record to keep a span.
    try:
        from opentelemetry import trace

        end_ns = time.time_ns()
        start_ns = end_ns - int(duration_ms * 1_000_000)
        context = None
        if isinstance(parent, trace.SpanContext):  # capture_parent returns one, or None
            context = trace.set_span_in_context(trace.NonRecordingSpan(parent))
        span = trace.get_tracer("agentgate").start_span(
            f"chat {model}" if model else "chat",
            context=context,
            start_time=start_ns,
            attributes={k: v for k, v in attributes.items() if v is not None},
        )
        span.end(end_time=end_ns)
    except Exception:  # noqa: BLE001 - degrade to no span
        log.debug("span emit failed", exc_info=True)
