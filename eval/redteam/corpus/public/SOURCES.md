# Vendored public corpus — provenance

Pinned snapshots of public datasets, normalized into the harness JSONL schema
(`{id, source, text, label, label_origin, category, meta}`). Committed so harness runs are
offline and reproducible. Regenerate with the adjacent `_vendor_*.py` script.

## deepset/prompt-injections

- **File:** `deepset_prompt_injections.jsonl`
- **Source:** https://huggingface.co/datasets/deepset/prompt-injections
- **License:** Apache-2.0 (permissive — redistribution OK with attribution)
- **Schema:** `text` (string), `label` (0 = benign, 1 = injection)
- **Snapshot:** 662 items — 263 positives / 399 negatives (train+test splits merged;
  original split retained in `meta.split`)
- **`label_origin`:** `known` (dataset-provided labels)
- **`category`:** `public` (no fine-grained attack taxonomy in the source)
- **sha256 (of normalized JSONL):** `3febc62f6b7bdc5b7f0b44844e7f6a2085f3abed94e057f6d602c2740ea331b6`
- **Fetched:** 2026-06-05 via HF datasets-server REST API (`_vendor_deepset.py`)
