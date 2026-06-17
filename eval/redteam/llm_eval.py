"""Reproduce and output the local-LLM guard vs DeBERTa baseline evaluation.

Loads the corpus and runs the detectors over the specific sub-corpora:
  - Independent Garak latentinjection positives (72)
  - seed_mutate headline positives (28)
  - Organic benign FP captures (257) split by user vs tool_output channels.

Outputs a formatted comparative table and saves the results to a JSON file.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from eval.redteam.judge import LLMGuard
from eval.redteam.loader import FP_CAPTURE, RUNS_DIR, load_corpus, load_jsonl

# Define paths
_REPO_ROOT = Path(__file__).resolve().parents[2]
_JSON_OUT = RUNS_DIR / "llm_eval_results.json"


def run_eval() -> None:
    # 1. Initialize detectors
    print("Initializing detectors...")
    llm_guard = LLMGuard()
    
    # Try to load deberta model. The model + numpy/onnxruntime are imported lazily on first
    # scan, so warm it up here to surface a missing `guard` extra or model dir now (and let
    # the eval degrade to LLM-only) rather than mid-run with an uncaught error.
    try:
        from agentgate.security import model_guard
        deberta_scan = model_guard.scan_text
        deberta_scan("warm-up")  # force the lazy model/numpy import
        has_deberta = True
    except Exception as e:
        print(f"Warning: could not load DeBERTa guard ({type(e).__name__}: {e}); "
              "reporting LLM-guard column only.")
        has_deberta = False
        deberta_scan = None

    # 2. Load and filter corpora
    print("Loading evaluation corpus...")
    items = load_corpus(include_fp_capture=True)

    # Independent Garak latentinjection (positive targets)
    garak_items = [
        i for i in items
        if i.source == "garak"
        and i.category
        and i.category.startswith("garak:latentinjection.")
        and i.label == 1
    ]

    # seed_mutate headline (positive targets)
    seed_mutate_items = [
        i for i in items
        if i.source == "seed_mutate"
        and i.category in ("seedmutate:doc_embedded", "seedmutate:tool_result")
        and i.label == 1
    ]

    # Benign FP captures (all treated as benign / negatives for FP calculation)
    # Load all items from the FP-capture file to get correct vector metadata
    fp_items = list(load_jsonl(FP_CAPTURE))
    user_fp = [i for i in fp_items if i.meta.get("vector") == "user"]
    tool_fp = [i for i in fp_items if i.meta.get("vector") == "tool_output"]

    print("Evaluation dataset loaded:")
    print(f"  - Independent Garak latentinjection: {len(garak_items)} items")
    print(f"  - seed_mutate headline: {len(seed_mutate_items)} items")
    print(f"  - Benign capture User (FP): {len(user_fp)} items")
    print(f"  - Benign capture Tool (FP): {len(tool_fp)} items")
    print(f"  - Benign capture Combined (FP): {len(fp_items)} items")

    # 3. Score items
    print("\nRunning evaluation (using cache where available)...")
    
    def evaluate(detector_name: str, scan_fn, dataset):
        # Count the detector's own binary verdict (v.flagged). For the LLM guard this is the
        # label-based binary decision; for DeBERTa .flagged is score >= FLAG_THRESHOLD.
        flagged_count = 0
        total = len(dataset)
        for item in dataset:
            if scan_fn(item.text).flagged:
                flagged_count += 1

        rate = flagged_count / total if total > 0 else 0.0
        return {
            "flagged": flagged_count,
            "total": total,
            "rate": rate
        }

    # Evaluate LLM Guard
    llm_garak = evaluate("llm-guard", llm_guard.scan_text, garak_items)
    llm_seed = evaluate("llm-guard", llm_guard.scan_text, seed_mutate_items)
    llm_fp_tool = evaluate("llm-guard", llm_guard.scan_text, tool_fp)
    llm_fp_user = evaluate("llm-guard", llm_guard.scan_text, user_fp)
    llm_fp_comb = evaluate("llm-guard", llm_guard.scan_text, fp_items)

    # Evaluate DeBERTa if available
    if has_deberta:
        deb_garak = evaluate("deberta", deberta_scan, garak_items)
        deb_seed = evaluate("deberta", deberta_scan, seed_mutate_items)
        deb_fp_tool = evaluate("deberta", deberta_scan, tool_fp)
        deb_fp_user = evaluate("deberta", deberta_scan, user_fp)
        deb_fp_comb = evaluate("deberta", deberta_scan, fp_items)
    else:
        deb_garak = deb_seed = deb_fp_tool = deb_fp_user = deb_fp_comb = {
            "flagged": 0, "total": 0, "rate": 0.0
        }

    # 4. Save structured results
    results = {
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "model": llm_guard.cfg.model,
        "garak_recall": llm_garak,
        "seed_mutate_recall": llm_seed,
        "fp_tool_output": llm_fp_tool,
        "fp_user": llm_fp_user,
        "fp_combined": llm_fp_comb,
        "deberta_baseline": {
            "garak_recall": deb_garak,
            "seed_mutate_recall": deb_seed,
            "fp_tool_output": deb_fp_tool,
            "fp_user": deb_fp_user,
            "fp_combined": deb_fp_comb,
        } if has_deberta else None
    }
    
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    _JSON_OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved structured results to: {_JSON_OUT.relative_to(_REPO_ROOT)}")

    # 5. Print results comparison table
    print("\n" + "=" * 80)
    print("LOCAL-LLM GUARD EVALUATION SUMMARY")
    print(f"Model: {llm_guard.cfg.model} | Decision: binary label (0/1)")
    print("=" * 80)
    
    def fmt_cell(val, total, is_fp: bool = False):
        pct = val / total if total > 0 else 0.0
        return f"{pct:.1%} ({val}/{total})"

    headers = [
        "Metric / Sub-Corpus",
        f"Local-LLM Guard ({llm_guard.cfg.model[:10]}...)",
        "DeBERTa Baseline",
        "Type / Context"
    ]
    
    row_fmt = "{:<32} | {:<22} | {:<22} | {:<32}"
    print(row_fmt.format(*headers))
    print("-" * 105)
    
    print(row_fmt.format(
        "Recall - Independent Garak",
        fmt_cell(llm_garak["flagged"], llm_garak["total"]),
        fmt_cell(deb_garak["flagged"], deb_garak["total"]) if has_deberta else "N/A",
        "garak:latentinjection (indep.)"
    ))
    print(row_fmt.format(
        "Recall - seed_mutate",
        fmt_cell(llm_seed["flagged"], llm_seed["total"]),
        fmt_cell(deb_seed["flagged"], deb_seed["total"]) if has_deberta else "N/A",
        "seedmutate headline (circular)"
    ))
    print(row_fmt.format(
        "FP Rate - Untrusted Channel",
        fmt_cell(llm_fp_tool["flagged"], llm_fp_tool["total"], is_fp=True),
        fmt_cell(deb_fp_tool["flagged"], deb_fp_tool["total"], is_fp=True) if has_deberta else "N/A",
        "benign capture:tool_output (FP)"
    ))
    print(row_fmt.format(
        "FP Rate - All-Channel (User)",
        fmt_cell(llm_fp_user["flagged"], llm_fp_user["total"], is_fp=True),
        fmt_cell(deb_fp_user["flagged"], deb_fp_user["total"], is_fp=True) if has_deberta else "N/A",
        "benign capture:user (FP)"
    ))
    print(row_fmt.format(
        "FP Rate - Combined",
        fmt_cell(llm_fp_comb["flagged"], llm_fp_comb["total"], is_fp=True),
        fmt_cell(deb_fp_comb["flagged"], deb_fp_comb["total"], is_fp=True) if has_deberta else "N/A",
        "benign capture:combined (FP)"
    ))
    print("=" * 105)


if __name__ == "__main__":
    # No-arg entrypoint: this launches a full, model-pinning eval. Guard with an
    # empty argparse so `--help`/unknown flags exit cleanly instead of silently
    # running the eval (a stray `... --help` would otherwise trigger the eval and
    # pin the oMLX model, blocking other loads until it finishes).
    import argparse
    argparse.ArgumentParser(
        description="Run the local-LLM guard vs DeBERTa eval. Takes no arguments."
    ).parse_args()
    run_eval()
