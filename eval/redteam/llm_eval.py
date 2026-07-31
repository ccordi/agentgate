"""Reproduce and output the local-LLM guard vs DeBERTa baseline evaluation.

Loads the corpus and runs the detectors over the specific sub-corpora:
  - Independent Garak latentinjection positives (72)
  - seed_mutate primary positives (28)
  - Benign FP captures (153) split by user vs tool_output channels.

Outputs a formatted comparative table and saves the results to a JSON file.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from agentgate.guards.local_llm import JudgeConfig, LLMGuard

from .common import REPO_ROOT, fmt_rate_ci
from .judge import CACHE_PATH
from .loader import FP_CAPTURE, RUNS_DIR, load_corpus, load_jsonl

_JSON_OUT = RUNS_DIR / "llm_eval_results.json"


def run_eval() -> None:
    print("Initializing detectors...")
    # Measurement runs share the harness's verdict cache, not the gateway's runtime one.
    llm_guard = LLMGuard(JudgeConfig(cache_path=str(CACHE_PATH)))

    # Try to load deberta model. The model + numpy/onnxruntime are imported lazily on first
    # scan, so warm it up here to surface a missing `guard` extra or model dir now (and let
    # the eval degrade to LLM-only) rather than mid-run with an uncaught error.
    try:
        from agentgate.guards import deberta
        deberta_scan = deberta.scan_text
        deberta_scan("warm-up")  # force the lazy model/numpy import
        has_deberta = True
    except Exception as e:
        print(f"Warning: could not load DeBERTa guard ({type(e).__name__}: {e}); "
              "reporting LLM-guard column only.")
        has_deberta = False
        deberta_scan = None

    print("Loading evaluation corpus...")
    items = load_corpus(include_fp_capture=True)

    garak_items = [
        i for i in items
        if i.source == "garak"
        and i.category
        and i.category.startswith("garak:latentinjection.")
        and i.label == 1
    ]

    seed_mutate_items = [
        i for i in items
        if i.source == "seed_mutate"
        and i.category in ("seedmutate:doc_embedded", "seedmutate:tool_result")
        and i.label == 1
    ]

    # Benign FP captures (all treated as benign / negatives for FP calculation)
    # Load all items from the FP-capture file to get correct vector metadata
    if not FP_CAPTURE.exists():
        raise SystemExit(
            f"no FP capture corpus at {FP_CAPTURE}; point AGENTGATE_FP_CAPTURE_PATH at a "
            "capture file, or unset it to use the committed frozen snapshot."
        )
    fp_items = list(load_jsonl(FP_CAPTURE))
    user_fp = [i for i in fp_items if i.meta.get("vector") == "user"]
    tool_fp = [i for i in fp_items if i.meta.get("vector") == "tool_output"]

    print("Evaluation dataset loaded:")
    print(f"  - Independent Garak latentinjection: {len(garak_items)} items")
    print(f"  - seed_mutate primary set: {len(seed_mutate_items)} items")
    print(f"  - Benign capture User (FP): {len(user_fp)} items")
    print(f"  - Benign capture Tool (FP): {len(tool_fp)} items")
    print(f"  - Benign capture Combined (FP): {len(fp_items)} items")

    print("\nRunning evaluation (using cache where available)...")
    
    def evaluate(scan_fn, dataset):
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

    llm_garak = evaluate(llm_guard.scan_text, garak_items)
    llm_seed = evaluate(llm_guard.scan_text, seed_mutate_items)
    llm_fp_tool = evaluate(llm_guard.scan_text, tool_fp)
    llm_fp_user = evaluate(llm_guard.scan_text, user_fp)
    llm_fp_comb = evaluate(llm_guard.scan_text, fp_items)

    if has_deberta:
        deb_garak = evaluate(deberta_scan, garak_items)
        deb_seed = evaluate(deberta_scan, seed_mutate_items)
        deb_fp_tool = evaluate(deberta_scan, tool_fp)
        deb_fp_user = evaluate(deberta_scan, user_fp)
        deb_fp_comb = evaluate(deberta_scan, fp_items)
    else:
        deb_garak = deb_seed = deb_fp_tool = deb_fp_user = deb_fp_comb = {
            "flagged": 0, "total": 0, "rate": 0.0
        }

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
    print(f"\nSaved structured results to: {_JSON_OUT.relative_to(REPO_ROOT)}")

    # Rates carry a 95% Wilson interval because these
    # sub-corpora are small (n=28 to n=153) and the point estimate alone reads far
    # more precise than the sample supports.
    print("\n" + "=" * 80)
    print("LOCAL-LLM GUARD EVALUATION SUMMARY")
    print(f"Model: {llm_guard.cfg.model} | Decision: binary label (0/1)")
    print("=" * 80)

    rows = [
        ("Recall - Independent Garak", llm_garak, deb_garak,
         "garak:latentinjection (indep.)"),
        ("Recall - seed_mutate", llm_seed, deb_seed,
         "seedmutate primary set (circular)"),
        ("FP Rate - Untrusted Channel", llm_fp_tool, deb_fp_tool,
         "benign capture:tool_output (FP)"),
        ("FP Rate - All-Channel (User)", llm_fp_user, deb_fp_user,
         "benign capture:user (FP)"),
        ("FP Rate - Combined", llm_fp_comb, deb_fp_comb,
         "benign capture:combined (FP)"),
    ]

    headers = [
        "Metric / Sub-Corpus",
        f"Local-LLM Guard ({llm_guard.cfg.model[:10]}...)",
        "DeBERTa Baseline",
        "Type / Context",
    ]
    row_fmt = "{:<32} | {:<34} | {:<34} | {:<32}"
    print(row_fmt.format(*headers))
    print("-" * 140)
    for label, llm, deb, note in rows:
        print(row_fmt.format(
            label,
            fmt_rate_ci(llm["flagged"], llm["total"]),
            fmt_rate_ci(deb["flagged"], deb["total"]) if has_deberta else "N/A",
            note,
        ))
    print("=" * 140)


def register(sub) -> None:
    """Register the `llm-eval` subcommand on `python -m eval.redteam`."""
    sub.add_parser(
        "llm-eval", help="local-LLM guard vs DeBERTa comparison (needs a local model server)"
    ).set_defaults(fn=main)


def main(_args=None) -> None:
    run_eval()


if __name__ == "__main__":
    # No-arg entrypoint: this launches a full, model-pinning eval. Guard with an
    # empty argparse so `--help`/unknown flags exit cleanly instead of silently
    # running the eval (a stray `... --help` would otherwise trigger the eval and
    # pin the oMLX model, blocking other loads until it finishes).
    import argparse
    argparse.ArgumentParser(
        description="Run the local-LLM guard vs DeBERTa eval. Takes no arguments."
    ).parse_args()
    main()
