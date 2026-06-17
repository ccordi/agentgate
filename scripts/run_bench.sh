#!/usr/bin/env bash
# Manual latency benchmark for the gateway overhead figure used in the write-up.
#
# Spins up an ISOLATED bench gateway (:4300, mock provider, throwaway data/bench.db)
# in front of a deterministic mock upstream (:4200), runs k6 load phases, and renders
# the report. It NEVER touches a running/production gateway instance.
#
#   Output artifact : docs/benchmarks.md   (full report — this is what I read)
#   Quick summary   : bench/runs/SUMMARY.txt (key overhead lines, easy to paste)
#
# Usage:
#   bash scripts/run_bench.sh           # full run  (12s phases, ~2-3 min)
#   bash scripts/run_bench.sh --quick   # smoke run (6s phases,  ~1-2 min)
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v k6 >/dev/null 2>&1; then
  echo "ERROR: k6 not found on PATH. Install it:  brew install k6" >&2
  exit 1
fi

# Pin the injection guard to the heuristic (regex) scanner so this measures the
# gateway's PROXY-FLOOR overhead deterministically — no ML model loaded, regardless
# of whether the `guard` extra / DeBERTa ONNX model happen to be present locally.
# The DeBERTa (~40ms) and local-LLM (~1.4s) guard costs are separate tiers, measured
# elsewhere; this run is the proxy pass-through figure only.
export AGENTGATE_GUARD_BACKEND=heuristic

echo "==> Running gateway latency benchmark (isolated; no live instance touched)…"
echo "    Guard backend pinned to: heuristic (no ML model loaded — pure proxy floor)."
echo "    This takes ~2-3 min (--quick: ~1-2 min). Phases print as they run."
echo

uv run python -m bench.run "$@"

ART="docs/benchmarks.md"
OUT="bench/runs/SUMMARY.txt"
mkdir -p bench/runs

{
  echo "agentgate bench summary"
  echo "======================="
  echo
  echo "--- Gateway overhead (the headline figure) ---"
  grep -A2 -iE "## Gateway overhead" "$ART" || echo "(overhead section not found in $ART)"
  echo
  echo "--- Streaming overhead ---"
  grep -A2 -iE "## Streaming overhead" "$ART" || true
  echo
  echo "Full report: $ART"
} | tee "$OUT"

echo
echo "==> Done."
echo "    Full report : $ART"
echo "    Quick paste : $OUT"
echo "    Tell me when this is finished and I'll read $ART and wire the verified number into the post."
