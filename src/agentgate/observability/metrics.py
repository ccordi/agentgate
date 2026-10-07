"""Prometheus metrics.

Exposed at /metrics: request, latency, token and cost series, plus the injection-flag
and failure-visibility counters below.
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

# --- Failure-visibility counters --------------------------------------------------------
#
# The rule these serve: where the system swallows an error or fails a control, it should
# count it — otherwise it cannot tell you it is broken.

guard_unavailable_total = Counter(
    "agentgate_guard_unavailable_total",
    "Requests rejected 503 because a guard scan could not run.",
    ["backend"],  # deberta|llm|combined
)

limits_unavailable_total = Counter(
    "agentgate_limits_unavailable_total",
    "Requests rejected 503 because the spend/kill-switch limits backend could not be read.",
    ["backend"],  # spend
)

auth_rejected_total = Counter(
    "agentgate_auth_rejected_total",
    "Requests refused 401 by the issued-keys gate (no audit row: rejection happens "
    "at the header stage, before anything is read).",
    ["reason"],  # missing|unknown|revoked — the caller sees one opaque 401
)

upstream_credentials_missing_total = Counter(
    "agentgate_upstream_credentials_missing_total",
    "Requests rejected 503 because issued keys are required and the resolved non-local "
    "provider has no api_key to inject (a minted key never leaves the gateway).",
    ["provider"],
)

admission_shed_total = Counter(
    "agentgate_admission_shed_total",
    "Requests refused at the front door by admission control (no audit row exists for "
    "these: nothing was read, so there is nothing to record about the request).",
    ["reason"],  # queue_full|deadline|global
)

audit_write_failures_total = Counter(
    "agentgate_audit_write_failures_total",
    "Audit writes that raised and were swallowed by design (auditing must never "
    "break forwarding).",
    ["kind"],  # request|content_sample
)

scan_truncated_total = Counter(
    "agentgate_scan_truncated_total",
    "Classification windows exhausted: the scan covered the first N characters and "
    "stopped. The audit row carries truncated:<site>:<N> in `caveats`.",
    ["site"],  # classify|egress_payload
)

rejects_total = Counter(
    "agentgate_rejects_total",
    "Requests rejected before forwarding, by wire error type — the same string the "
    "client was answered with and the audit row carries as rejected:<type>. Auth "
    "refusals and admission sheds are counted separately: those paths write no row.",
    ["type"],
)

gap_abandoned_total = Counter(
    "agentgate_gap_abandoned_total",
    "A bounded-gap detection pattern saw its prefix with the suffix past the bound: "
    "text shaped like a match, padded out of reach. Recorded on the audit row as "
    "gap_abandoned:<pattern>:<distance>; never flags or blocks.",
    ["pattern"],
)

price_unknown_model_total = Counter(
    "agentgate_price_unknown_model_total",
    "Cost lookups that found no price for the model and recorded $0 — spend the "
    "USD cap cannot see. No model label (unbounded cardinality); the model name "
    "is in the pricing module's warning log line.",
)
