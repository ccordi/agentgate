"""Pure latency-stats helpers for the benchmark (no numpy).

Used to turn the per-request latency columns the gateway already records in the audit DB
(`latency_total_ms`, `latency_upstream_ms`, `latency_inject_ms`) into the p50/p90/p95/p99
decomposition the benchmark report renders. Kept dependency-free and unit-tested in isolation.
"""

from __future__ import annotations

from collections.abc import Sequence


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile (``p`` in 0–100), matching numpy's default method."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    rank = (p / 100.0) * (len(s) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    frac = rank - lo
    return float(s[lo] + (s[hi] - s[lo]) * frac)


def summarize(values: Sequence[float | None]) -> dict:
    """count + min/mean/p50/p90/p95/p99/max over non-null values (ms), rounded."""
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"count": 0}
    return {
        "count": len(vals),
        "min": round(min(vals), 3),
        "mean": round(sum(vals) / len(vals), 3),
        "p50": round(percentile(vals, 50), 3),
        "p90": round(percentile(vals, 90), 3),
        "p95": round(percentile(vals, 95), 3),
        "p99": round(percentile(vals, 99), 3),
        "max": round(max(vals), 3),
    }


def gateway_overhead_ms(total: Sequence[float | None], upstream: Sequence[float | None]) -> list[float]:
    """Per-request gateway-internal overhead = total − upstream (ms).

    Both columns come from the same audit row, so they're paired positionally. Negative
    values (clock jitter on sub-ms diffs) are clamped to 0.
    """
    out: list[float] = []
    for t, u in zip(total, upstream, strict=True):
        if t is None or u is None:
            continue
        out.append(max(0.0, float(t) - float(u)))
    return out
