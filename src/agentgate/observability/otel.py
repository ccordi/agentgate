"""OpenTelemetry tracing setup.

Establishes the tracer + FastAPI auto-instrumentation. Pipeline stages
(inject / classify / route / redact / upstream) can be wrapped in spans; that span
tree backs the per-stage latency decomposition in benchmark reports. Spans export
via OTLP only when an endpoint is configured; otherwise they're recorded and
dropped, so there's no collector dependency for local deployments.
"""

from __future__ import annotations

import logging
import os

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

log = logging.getLogger("agentgate.otel")

_INSTRUMENTED = False


def setup_tracing(app) -> None:
    global _INSTRUMENTED
    if _INSTRUMENTED:
        return

    provider = TracerProvider(resource=Resource.create({"service.name": "agentgate"}))

    endpoint = os.environ.get("AGENTGATE_OTLP_ENDPOINT")
    if endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
            log.info("OTLP span export -> %s", endpoint)
        except ImportError:
            log.warning("AGENTGATE_OTLP_ENDPOINT set but OTLP exporter missing; spans dropped")

    trace.set_tracer_provider(provider)

    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
    except Exception:  # noqa: BLE001
        log.exception("FastAPI OTel instrumentation failed")

    _INSTRUMENTED = True


def tracer():
    return trace.get_tracer("agentgate")
