#!/usr/bin/env bash
# Regenerate corpus/public/garak.jsonl from garak's probes.
#
# garak lives in its OWN isolated venv (Python 3.12 — torch has no cp314 wheels, and we keep
# torch/transformers out of agentgate's lock). It is NOT a project dependency; install once:
#     uv tool install --python 3.12 garak     # pinned at v0.15.1 when this corpus was built
#
# Two stages, two venvs, handed off via a JSON file:
#   1. dump_prompts.py  — garak's venv (uvx)  — extract probe prompts → raw JSON
#   2. export.py        — agentgate's venv    — raw JSON → corpus/public/garak.jsonl
#
# After this, score/report run unchanged:
#     uv run python -m eval.redteam score --detector deberta
#     uv run python -m eval.redteam report
set -euo pipefail
cd "$(dirname "$0")/../../.."   # repo root

RAW="$(mktemp -t garak_prompts.XXXXXX.json)"
trap 'rm -f "$RAW"' EXIT

echo "[1/2] dumping garak probe prompts (garak venv)…"
uvx --from garak python eval/redteam/gen/dump_prompts.py "$RAW"

echo "[2/2] exporting → corpus/public/garak.jsonl (agentgate venv)…"
uv run python -m eval.redteam.gen.export "$RAW"
