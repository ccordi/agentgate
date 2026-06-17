import json
import time
from pathlib import Path

import numpy as np

from agentgate.security import model_guard
from agentgate.security.llm_guard import JudgeConfig, LLMGuard
from eval.redteam.loader import RUNS_DIR, load_jsonl

# Model is read from AGENTGATE_JUDGE_MODEL (env / .env). Set it before running this script.

CORPUS_DIR = Path("eval/redteam/corpus")
OBFUSCATED_PROBE = CORPUS_DIR / "obfuscated_tool_probe.jsonl"
SECURITY_META_PROBE = CORPUS_DIR / "security_meta_probe.jsonl"

def run_extra_eval():
    print("Initializing guard configurations...")
    cfg = JudgeConfig()
    llm_guard_obj = LLMGuard(cfg)
    
    # Warm up DeBERTa
    print("Warming up DeBERTa...")
    try:
        model_guard.scan_text("warm-up")
        has_deberta = True
    except Exception as e:
        print(f"Warning: could not load DeBERTa ({e})")
        has_deberta = False

    # Load datasets
    print("Loading probes...")
    obfuscated_items = list(load_jsonl(OBFUSCATED_PROBE))
    meta_items = list(load_jsonl(SECURITY_META_PROBE))
    
    print(f"Loaded {len(obfuscated_items)} obfuscated items and {len(meta_items)} security meta-content items.")

    # 1. Evaluate Obfuscated Tool Probe
    print("\n--- Evaluating Obfuscated Tool Probe (c) ---")
    obfuscated_results = []
    llm_latencies = []
    
    for idx, item in enumerate(obfuscated_items):
        t0 = time.perf_counter()
        llm_verdict = llm_guard_obj.scan_text(item.text)
        t1 = time.perf_counter()
        llm_lat = t1 - t0
        llm_latencies.append(llm_lat)
        
        deb_verdict = model_guard.scan_text(item.text) if has_deberta else None
        
        combined_flagged = llm_verdict.flagged or (deb_verdict.flagged if deb_verdict else False)
        
        obfuscated_results.append({
            "text": item.text,
            "obfuscation": item.meta.get("obfuscation"),
            "intent": item.meta.get("intent"),
            "llm_flagged": llm_verdict.flagged,
            "llm_latency": llm_lat,
            "deberta_flagged": deb_verdict.flagged if deb_verdict else False,
            "combined_flagged": combined_flagged
        })
        print(f"Item {idx + 1}/{len(obfuscated_items)}: [{item.meta.get('obfuscation')}] LLM={llm_verdict.flagged} ({llm_lat:.2f}s) | DeBERTa={deb_verdict.flagged if deb_verdict else 'N/A'} | Combined={combined_flagged}")

    # 2. Evaluate Security Meta Probe
    print("\n--- Evaluating Security Meta Probe (d) ---")
    meta_results = []
    
    for idx, item in enumerate(meta_items):
        t0 = time.perf_counter()
        llm_verdict = llm_guard_obj.scan_text(item.text)
        t1 = time.perf_counter()
        llm_lat = t1 - t0
        llm_latencies.append(llm_lat)
        
        deb_verdict = model_guard.scan_text(item.text) if has_deberta else None
        combined_flagged = llm_verdict.flagged or (deb_verdict.flagged if deb_verdict else False)
        
        meta_results.append({
            "text": item.text,
            "llm_flagged": llm_verdict.flagged,
            "llm_latency": llm_lat,
            "deberta_flagged": deb_verdict.flagged if deb_verdict else False,
            "combined_flagged": combined_flagged
        })
        print(f"Item {idx + 1}/{len(meta_items)}: LLM={llm_verdict.flagged} ({llm_lat:.2f}s) | DeBERTa={deb_verdict.flagged if deb_verdict else 'N/A'} | Combined={combined_flagged}")

    # Calculations
    obf_llm_recall = sum(1 for r in obfuscated_results if r["llm_flagged"]) / len(obfuscated_items)
    obf_deb_recall = sum(1 for r in obfuscated_results if r["deberta_flagged"]) / len(obfuscated_items) if has_deberta else 0.0
    obf_comb_recall = sum(1 for r in obfuscated_results if r["combined_flagged"]) / len(obfuscated_items)
    
    meta_llm_fp = sum(1 for r in meta_results if r["llm_flagged"]) / len(meta_items)
    meta_deb_fp = sum(1 for r in meta_results if r["deberta_flagged"]) / len(meta_items) if has_deberta else 0.0
    meta_comb_fp = sum(1 for r in meta_results if r["combined_flagged"]) / len(meta_items)

    # Latency Stats (all calls in this script)
    # The very first call of this script run represents a warm call because the server already loaded the model.
    # We can separate first call from the rest of the calls.
    first_call_lat = llm_latencies[0]
    subsequent_lats = llm_latencies[1:]
    
    p50_lat = np.percentile(subsequent_lats, 50) if subsequent_lats else first_call_lat
    p95_lat = np.percentile(subsequent_lats, 95) if subsequent_lats else first_call_lat

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

    # Save results
    out_path = RUNS_DIR / "extra_probes_results.json"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved results to: {out_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("EXTRA PROBES EVALUATION SUMMARY")
    print(f"Model: {cfg.model}")
    print("=" * 80)
    row_fmt = "{:<35} | {:<12} | {:<12} | {:<12}"
    print(row_fmt.format("Metric", "LLM Guard", "DeBERTa", "Combined"))
    print("-" * 80)
    print(row_fmt.format(
        "Obfuscation Recall (c)", 
        f"{obf_llm_recall:.1%}", 
        f"{obf_deb_recall:.1%}" if has_deberta else "N/A", 
        f"{obf_comb_recall:.1%}"
    ))
    print(row_fmt.format(
        "Security Meta FP Rate (d)", 
        f"{meta_llm_fp:.1%}", 
        f"{meta_deb_fp:.1%}" if has_deberta else "N/A", 
        f"{meta_comb_fp:.1%}"
    ))
    print("-" * 80)
    print(f"Latency (Subsequent Warm Calls): p50 = {p50_lat:.3f}s, p95 = {p95_lat:.3f}s")
    print(f"Latency (First call in this run): {first_call_lat:.3f}s")
    print("=" * 80)

if __name__ == "__main__":
    # No-arg entrypoint that launches a model-pinning eval — guard with an empty
    # argparse so `--help`/unknown flags exit cleanly instead of running the eval.
    import argparse
    argparse.ArgumentParser(
        description="Run the obfuscated/security-meta extra probes eval. Takes no arguments."
    ).parse_args()
    run_extra_eval()
