"""Two targeted probes the main corpus doesn't cover, run against both guard backends.

  - obfuscated tool-channel payloads — bare base64 / ROT13 / zero-width, no cleartext
      "decode this and follow it" lure. Measures recall on disguised content.
  - security meta-content — benign tool results that merely *quote* attack strings.
      Measures the false-positive cost of pattern-matching on security discussion.

Needs a local model server for the LLM guard; the DeBERTa column degrades to N/A when
the `guard` extra or the model directory is absent.

    uv run python -m eval.redteam probes-eval
"""

from __future__ import annotations

import json
import time

from agentgate.guards import deberta
from agentgate.guards.local_llm import JudgeConfig, LLMGuard

from .common import percentile
from .judge import CACHE_PATH
from .loader import CORPUS_DIR, RUNS_DIR, load_jsonl

# Model is read from AGENTGATE_JUDGE_MODEL (env / .env). Set it before running this script.

OBFUSCATED_PROBE = CORPUS_DIR / "obfuscated_tool_probe.jsonl"
SECURITY_META_PROBE = CORPUS_DIR / "security_meta_probe.jsonl"


def _score_probe(items, llm_guard_obj, deberta_scan, latencies: list[float]) -> list[dict]:
    """Score one probe set with both backends; append every LLM call's latency.

    Both probe sets are scored identically — the only difference is which meta fields
    the caller pulls off the items afterwards.
    """
    results = []
    for idx, item in enumerate(items):
        t0 = time.perf_counter()
        llm_verdict = llm_guard_obj.scan_text(item.text)
        llm_lat = time.perf_counter() - t0
        latencies.append(llm_lat)

        deb_verdict = deberta_scan(item.text) if deberta_scan else None
        deb_flagged = deb_verdict.flagged if deb_verdict else False
        combined_flagged = llm_verdict.flagged or deb_flagged

        results.append({
            "text": item.text,
            "obfuscation": item.meta.get("obfuscation"),
            "intent": item.meta.get("intent"),
            "llm_flagged": llm_verdict.flagged,
            "llm_latency": llm_lat,
            "deberta_flagged": deb_flagged,
            "combined_flagged": combined_flagged,
        })
        tag = f"[{item.meta['obfuscation']}] " if item.meta.get("obfuscation") else ""
        print(f"Item {idx + 1}/{len(items)}: {tag}LLM={llm_verdict.flagged} "
              f"({llm_lat:.2f}s) | DeBERTa={deb_verdict.flagged if deb_verdict else 'N/A'} "
              f"| Combined={combined_flagged}")
    return results


def _rate(results: list[dict], key: str) -> float:
    return sum(1 for r in results if r[key]) / len(results) if results else 0.0


def run_extra_eval():
    print("Initializing guard configurations...")
    # Measurement runs share the harness's verdict cache, not the gateway's runtime one.
    cfg = JudgeConfig(cache_path=str(CACHE_PATH))
    llm_guard_obj = LLMGuard(cfg)

    print("Warming up DeBERTa...")
    try:
        deberta.scan_text("warm-up")
        has_deberta = True
    except Exception as e:
        print(f"Warning: could not load DeBERTa ({e})")
        has_deberta = False

    print("Loading probes...")
    obfuscated_items = list(load_jsonl(OBFUSCATED_PROBE))
    meta_items = list(load_jsonl(SECURITY_META_PROBE))
    
    print(f"Loaded {len(obfuscated_items)} obfuscated items and {len(meta_items)} security meta-content items.")

    deberta_scan = deberta.scan_text if has_deberta else None
    llm_latencies: list[float] = []

    print("\n--- Evaluating Obfuscated Tool Probe ---")
    obfuscated_results = _score_probe(obfuscated_items, llm_guard_obj, deberta_scan, llm_latencies)

    print("\n--- Evaluating Security Meta Probe ---")
    meta_results = _score_probe(meta_items, llm_guard_obj, deberta_scan, llm_latencies)

    obf_llm_recall = _rate(obfuscated_results, "llm_flagged")
    obf_deb_recall = _rate(obfuscated_results, "deberta_flagged") if has_deberta else 0.0
    obf_comb_recall = _rate(obfuscated_results, "combined_flagged")

    meta_llm_fp = _rate(meta_results, "llm_flagged")
    meta_deb_fp = _rate(meta_results, "deberta_flagged") if has_deberta else 0.0
    meta_comb_fp = _rate(meta_results, "combined_flagged")

    # Latency Stats (all LLM calls in this script). The first call is reported
    # separately from the rest, on the assumption that the server already has the model
    # resident — i.e. these are warm-path numbers, not a cold load. That assumption is
    # NOT verified here; a cold first call would show up as an outlier first_call_s.
    first_call_lat = llm_latencies[0]
    subsequent_lats = llm_latencies[1:]

    p50_lat = percentile(subsequent_lats, 50) if subsequent_lats else first_call_lat
    p95_lat = percentile(subsequent_lats, 95) if subsequent_lats else first_call_lat

    results = {
        "model": cfg.model,
        "obfuscated_probe": {
            "llm_recall": obf_llm_recall,
            "deberta_recall": obf_deb_recall,
            "combined_recall": obf_comb_recall,
            "details": obfuscated_results
        },
        "security_meta_probe": {
            "llm_fp_rate": meta_llm_fp,
            "deberta_fp_rate": meta_deb_fp,
            "combined_fp_rate": meta_comb_fp,
            "details": meta_results
        },
        "latency": {
            "first_call_s": first_call_lat,
            "p50_s": p50_lat,
            "p95_s": p95_lat
        }
    }

    out_path = RUNS_DIR / "extra_probes_results.json"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved results to: {out_path}")

    print("\n" + "=" * 80)
    print("EXTRA PROBES EVALUATION SUMMARY")
    print(f"Model: {cfg.model}")
    print("=" * 80)
    row_fmt = "{:<35} | {:<12} | {:<12} | {:<12}"
    print(row_fmt.format("Metric", "LLM Guard", "DeBERTa", "Combined"))
    print("-" * 80)
    print(row_fmt.format(
        "Obfuscation Recall",
        f"{obf_llm_recall:.1%}", 
        f"{obf_deb_recall:.1%}" if has_deberta else "N/A", 
        f"{obf_comb_recall:.1%}"
    ))
    print(row_fmt.format(
        "Security Meta FP Rate",
        f"{meta_llm_fp:.1%}", 
        f"{meta_deb_fp:.1%}" if has_deberta else "N/A", 
        f"{meta_comb_fp:.1%}"
    ))
    print("-" * 80)
    print(f"Latency (Subsequent Warm Calls): p50 = {p50_lat:.3f}s, p95 = {p95_lat:.3f}s")
    print(f"Latency (First call in this run): {first_call_lat:.3f}s")
    print("=" * 80)


def register(sub) -> None:
    """Register the `probes-eval` subcommand on `python -m eval.redteam`."""
    sub.add_parser(
        "probes-eval", help="obfuscation + security-meta probes (needs a local model server)"
    ).set_defaults(fn=main)


def main(_args=None) -> None:
    run_extra_eval()


if __name__ == "__main__":
    # No-arg entrypoint that launches a model-pinning eval — guard with an empty
    # argparse so `--help`/unknown flags exit cleanly instead of running the eval.
    import argparse
    argparse.ArgumentParser(
        description="Run the obfuscated/security-meta extra probes eval. Takes no arguments."
    ).parse_args()
    main()
