"""Token -> USD cost estimation.

A small static price table, used by both audit (cost attribution) and the spend cap
(provider-aware USD ceiling). Local routes are free. Prices are USD per 1M tokens
and are approximate — good enough for cost *attribution* and cap enforcement, not
billing. Keyed by a loose model-name prefix match so minor version suffixes resolve.
"""

from __future__ import annotations

# (prompt_per_1m, completion_per_1m) USD. Extend as upstreams are added.
PRICE_TABLE: dict[str, tuple[float, float]] = {
    "gemini-3-flash": (0.30, 2.50),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
}


def estimate_cost_usd(model: str | None, prompt_tokens: int, completion_tokens: int) -> float:
    """Best-effort USD estimate. Unknown models (incl. local) cost 0.0."""
    if not model:
        return 0.0
    # Longest matching prefix wins so "gemini-2.5-flash-lite" doesn't resolve to
    # the more expensive "gemini-2.5-flash" entry.
    prices = None
    best_len = -1
    for prefix, p in PRICE_TABLE.items():
        if model.startswith(prefix) and len(prefix) > best_len:
            prices = p
            best_len = len(prefix)
    if prices is None:
        return 0.0
    prompt_rate, completion_rate = prices
    return (prompt_tokens / 1_000_000) * prompt_rate + (
        completion_tokens / 1_000_000
    ) * completion_rate
