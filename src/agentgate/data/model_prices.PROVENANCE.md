# model_prices.json — provenance

Vendored copy of LiteLLM's model price and context window map. The gateway's
pricing code reads this file from disk to estimate costs without importing
LiteLLM.

- **Source**: `model_prices_and_context_window.json` from
  [BerriAI/litellm](https://github.com/BerriAI/litellm) at tag `v1.95.0`.
- **License**: MIT (litellm 1.95.0 dist metadata, `License-Expression: MIT`).
  Attribution: BerriAI/litellm contributors.
- **Schema notes**: prices are USD **per token** (`input_cost_per_token` /
  `output_cost_per_token`). `sample_spec` and `fallback_generalizations` are
  reserved metadata keys, not models. The loader in `src/agentgate/pricing.py`
  skips entries unless both price fields are numeric. Zero is a valid rate.
- **Refresh**: the command below reads that release's map over the network.
  Use `--dry-run` to preview changes:

```bash
uv run python scripts/refresh_price_map.py \
  --from-url https://raw.githubusercontent.com/BerriAI/litellm/v1.95.0/model_prices_and_context_window.json \
  --dry-run
```

Remove `--dry-run` to write the map. When the script writes a map, it updates
this file's "Last refreshed" line.

Last refreshed: 2026-08-21 from BerriAI/litellm v1.95.0 model_prices_and_context_window.json
