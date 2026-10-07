"""Token -> USD cost estimation.

Prices come from a vendored copy of LiteLLM's model price map
(``data/model_prices.json`` — provenance and refresh instructions in the
adjacent ``model_prices.PROVENANCE.md``; refresh with
``scripts/refresh_price_map.py``). Used by both audit (cost attribution) and
the spend cap (provider-aware USD ceiling): estimates are good enough for cost
*attribution* and cap enforcement, not billing. Local routes are free. Models
with no price in the map cost 0.0 — loudly (warning log + counter), because a
$0 estimate on a cloud route is spend the USD cap cannot see.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

from agentgate.observability import metrics

log = logging.getLogger("agentgate")

_PRICE_MAP_PATH = Path(__file__).parent / "data" / "model_prices.json"

# Meta keys in the LiteLLM map that are not model entries.
_RESERVED_KEYS = frozenset({"sample_spec", "fallback_generalizations"})

# Model names already warned about — one log line per name per process keeps a
# busy route from flooding the log; the counter still counts every $0 lookup.
_warned_unknown: set[str] = set()


def _cost(entry: dict, field: str) -> float | None:
    """A numeric USD-per-token cost, or None (absent, or not a number — JSON
    booleans also count as int, so exclude them explicitly)."""
    value = entry.get(field)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


@lru_cache(maxsize=1)
def _price_table() -> dict[str, tuple[float, float, str | None]]:
    """model id -> (input USD/token, output USD/token, source map key when the
    id was indexed from a provider-prefixed entry — None for a bare key).

    Entries missing either price field are skipped, not defaulted: an
    unpriceable entry must fall through to the loud unknown-model path rather
    than price at $0 silently. Keys come bare (``gpt-4o``) and
    provider-prefixed (``gemini/gemini-2.5-flash``); prefixed keys are indexed
    under their suffix only where no bare key claims it — bare key wins. Among
    several prefixed entries sharing a suffix a priced one beats a $0-rated one,
    and otherwise the first in file order wins: the map carries free-tier and
    paid rows for the same alias (``codestral/codestral-latest`` at 0/0 ahead of
    ``mistral/codestral-latest``), and taking the $0 row would make every request on
    that alias invisible to the USD spend cap. Over-pricing an alias only makes
    the cap arrive early; under-pricing lets spend escape it.
    """
    raw = json.loads(_PRICE_MAP_PATH.read_text())
    table: dict[str, tuple[float, float, str | None]] = {}
    suffixed: dict[str, tuple[float, float, str | None]] = {}
    for key, entry in raw.items():
        if key in _RESERVED_KEYS or not isinstance(entry, dict):
            continue
        input_cost = _cost(entry, "input_cost_per_token")
        output_cost = _cost(entry, "output_cost_per_token")
        if input_cost is None or output_cost is None:
            continue
        if "/" in key:
            suffix = key.split("/", 1)[1]
            held = suffixed.get(suffix)
            if held is None or (held[0] == 0.0 and held[1] == 0.0
                                and (input_cost or output_cost)):
                suffixed[suffix] = (input_cost, output_cost, key)
        else:
            table[key] = (input_cost, output_cost, None)
    for suffix, priced in suffixed.items():
        table.setdefault(suffix, priced)
    return table


def estimate_cost_usd(model: str | None, prompt_tokens: int, completion_tokens: int) -> float:
    """Best-effort USD estimate. No model, or a model with no price in the map
    (including local models — they are never in it) -> 0.0; the no-price case
    warns and counts, since on a cloud route that $0 escapes the spend cap."""
    if not model:
        return 0.0
    table = _price_table()
    priced = table.get(model)
    if priced is None:
        # Longest matching bare-key prefix tolerates version suffixes the map
        # doesn't list ("gemini-2.5-flash-lite-zzz" resolves to
        # "gemini-2.5-flash-lite", not the shorter, pricier "gemini-2.5-flash").
        # Bare keys only: suffix-indexed keys include stubs like "llama3" (from
        # "ollama/llama3") that would falsely prefix-match local model names.
        best_len = -1
        for key, candidate in table.items():
            if candidate[2] is None and len(key) > best_len and model.startswith(key):
                priced, best_len = candidate, len(key)
    if priced is None:
        metrics.price_unknown_model_total.inc()
        if model not in _warned_unknown:
            _warned_unknown.add(model)
            log.warning(
                "no price for model %r — cost recorded as $0.0 (the USD spend cap "
                "does not cover this model)", model,
            )
        return 0.0
    input_rate, output_rate, via_key = priced
    if via_key is not None:
        log.debug("price for %r taken from provider-prefixed entry %r", model, via_key)
    return prompt_tokens * input_rate + completion_tokens * output_rate
