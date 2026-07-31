# Vendored public corpus — provenance

Pinned snapshots of public datasets and generated probe sets, normalized into the harness
JSONL schema (`{id, source, text, label, label_origin, category, meta}`). Committed so
harness runs are offline and reproducible.

Two kinds of file live here, and they regenerate differently:

- **Vendored** (`deepset_prompt_injections.jsonl`) — re-fetchable byte-for-byte; the
  adjacent `_vendor_*.py` script checks the result against the sha256 recorded below.
- **Generated** (`garak.jsonl`, `seed_mutate.jsonl`) — produced by running tools/models
  outside this repo. The committed file is the canonical snapshot; regeneration
  reproduces the *method*, not necessarily the bytes.

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

## NVIDIA garak — curated probe sample

- **File:** `garak.jsonl`
- **Source:** https://github.com/NVIDIA/garak — v0.15.1 (the version installed when this
  corpus was built), probe families `latentinjection.*`, `promptinject.*`, `encoding.*`
- **License:** Apache-2.0
- **Snapshot:** 240 items — all positives (20 probes × 12 prompts). Each probe is its own
  `garak:<probe>` category so per-category recall is the reported signal.
- **`label_origin`:** `known` (garak's probes are attacks by construction)
- **Curation:** a capped, evenly-strided sample per probe rather than garak's full 256
  prompts each — and the `encoding.*` families are tagged `meta.expected_miss`. Both
  choices, and why the split falls where it does, are argued in `gen/probe_map.py`.
- **Regenerate:** `bash eval/redteam/gen/run-garak.sh` (needs garak in its own Python 3.12
  venv — it is deliberately not a project dependency)
- **sha256:** `73937cfab9a5ee95a2de0bc71bd75cddfb56193b9726c3182b6db224510af97c`

## seed-and-mutate — locally generated attacks

- **File:** `seed_mutate.jsonl`
- **Source:** generated in-repo by `gen/seed_mutate.py` from the agent-specific intents in
  `gen/seeds.py`, expanded by a **local attacker model** (`gen/attacker.py`, configured via
  `AGENTGATE_ATTACKER_*`); the obfuscation control items come from the deterministic
  converters in `gen/converters.py`.
- **Snapshot:** 68 items — all positives. 28 primary (`doc_embedded` / `tool_result`,
  untagged: a miss counts as an unanticipated false negative) + 40 obfuscation controls (`enc_*`, tagged
  `meta.expected_miss`).
- **`label_origin`:** `known`
- **⚠️ Regeneration is nondeterministic** — the attacker model is sampled, so a re-run
  produces different text. **The committed file is the canonical snapshot**; every published
  seed_mutate number was measured against these exact bytes. Reproduce the *method* with
  `uv run python -m eval.redteam gen seed-mutate`, not the bytes.
- **Circularity note:** the attacker and the LLM guard can be the same model family, so
  the write-up treats the independent garak result as the stronger recall evidence
  (`sections/local_llm_guard.md`).
- **sha256:** `c4ad02e4780b4fd37737ab4b886c883f5365f68e7a51e9fc9392e4f4f8d25f47`
