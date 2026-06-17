"""Prometheus metrics.

Exposed at /metrics. Covers request/latency/token/cost series for benchmarking and
cost attribution, plus security-telemetry series (injection flag rate).
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

# Latency buckets tuned for LLM proxying: sub-ms gateway overhead up to long turns.
_LATENCY_BUCKETS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120)

requests_total = Counter(
    "agentgate_requests_total",
    "Chat-completion requests handled.",
    ["provider", "is_local", "status"],
)

request_latency_seconds = Histogram(
    "agentgate_request_latency_seconds",
    "End-to-end gateway latency (request received -> stream finalized).",
    ["provider"],
    buckets=_LATENCY_BUCKETS,
)

tokens_total = Counter(
    "agentgate_tokens_total",
    "Tokens accounted from upstream usage chunks.",
    ["provider", "direction"],  # direction: prompt|completion
)

cost_usd_total = Counter(
    "agentgate_cost_usd_total",
    "Estimated USD cost attributed to cloud upstreams.",
    ["provider"],
)

injection_flagged_total = Counter(
    "agentgate_injection_flagged_total",
    "Requests flagged by the inbound injection scanner.",
    ["blocked"],
)
