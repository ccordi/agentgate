"""CLI: ``python -m eval.redteam {score|judge|sample|report}``.

  score   offline metrics over the labeled corpus (no API key needed)
  judge   run the independent LLM judge to label items (needs AGENTGATE_JUDGE_API_KEY)
  sample  emit a stratified gold-set template for a human to label
  report  generate docs/redteam-results.md from scores + judge cache + gold set
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from . import harness
from . import judge as judge_mod
from . import report as report_mod
from .agreement import cohen_kappa
from .loader import (
    CORPUS_DIR,
    GOLD_SET,
    PKG_DIR,
    RUNS_DIR,
    load_corpus,
    load_gold_set,
    write_jsonl,
)
from .sampling import stratified_sample, to_gold_template
from .schema import LabelOrigin

REPO_ROOT = PKG_DIR.parents[1]
ARTIFACT = REPO_ROOT / "docs" / "redteam-results.md"
GOLD_TEMPLATE = CORPUS_DIR / "gold_set_unlabeled.jsonl"


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _detector(name: str):
    """Map a detector name → (scan_fn, label). Imports the model backend lazily."""
    if name == "deberta":
        from agentgate.security import model_guard
        return model_guard.scan_text, "deberta-v3-prompt-injection-v2"
    elif name == "llm-guard":
        from .judge import JudgeConfig
        from .judge import scan_text as llm_scan_text
        cfg = JudgeConfig()
        return llm_scan_text, f"llm-guard:{cfg.model}"
    from agentgate.security import injection
    return injection.scan_text, "heuristic"


def cmd_score(args) -> None:
    scan_fn, dname = _detector(args.detector)
    items = load_corpus()
    result = harness.run(items, scan_fn, dname).as_dict()
    print(f"detector: {dname}")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = RUNS_DIR / f"score-{_timestamp()}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    op = result["operating_point"]
    c = result["corpus"]
    print(f"corpus: {c['total_labeled']} labeled ({c['positives']}+/{c['negatives']}-) "
          f"from {', '.join(c['sources'])}")
    print(f"operating point (t={op['threshold']}): recall={op['recall']:.0%} "
          f"precision={op['precision']:.0%} fp_rate={op['fp_rate']:.0%} "
          f"(TP={op['tp']} FN={op['fn']} FP={op['fp']} TN={op['tn']})")
    em = result["expected_misses"]
    print(f"expected-miss evasions: {em['caught_anyway']}/{em['total']} caught")
    print(f"surprise false negatives: {len(result['missed_surprises'])}")
    print(f"→ {out.relative_to(REPO_ROOT)}")


def cmd_judge(_args) -> None:
    items = load_corpus(include_fp_capture=True)  # also label any captured FP text
    cfg = judge_mod.JudgeConfig()
    if not cfg.configured:
        raise SystemExit("set AGENTGATE_JUDGE_API_KEY to run the judge "
                         "(model/base_url via AGENTGATE_JUDGE_MODEL/_BASE_URL).")
    print(f"judging {len(items)} items with {cfg.model} (cached results are reused)…")
    labels = judge_mod.run_judge(items, cfg=cfg)

    # Sanity: judge vs. known-source labels.
    known = {i.id: i.label for i in items
             if i.label is not None and i.label_origin == LabelOrigin.KNOWN}
    judged = {i: labels[i].label for i in labels}
    shared = set(known) & set(judged)
    print(f"labeled {len(labels)} items.")
    if shared:
        agr = cohen_kappa({i: judged[i] for i in shared}, {i: known[i] for i in shared})
        print(f"vs known-source labels: agreement {agr.observed:.0%}, κ={agr.kappa:.2f} "
              f"(n={agr.n})")


def cmd_sample(args) -> None:
    items = load_corpus(include_fp_capture=True, labeled_only=False)
    existing = set(load_gold_set())
    picked = stratified_sample(items, args.n, seed=args.seed, exclude_ids=existing)
    templates = [to_gold_template(p) for p in picked]
    n = write_jsonl(GOLD_TEMPLATE, templates)
    print(f"wrote {n} unlabeled items → {GOLD_TEMPLATE.relative_to(REPO_ROOT)}")
    print("Hand-label each (set \"label\" to 0 or 1), then append the lines to "
          f"{GOLD_SET.relative_to(REPO_ROOT)} and re-run `report`.")


def _comparison_row(run: dict) -> dict:
    op, em = run["operating_point"], run["expected_misses"]
    return {
        "detector": run["corpus"]["detector"],
        "recall": op["recall"], "precision": op["precision"], "fp_rate": op["fp_rate"],
        "evasions_caught": em["caught_anyway"], "evasions_total": em["total"],
        "surprise_fn": len(run["missed_surprises"]),
    }


def cmd_report(_args) -> None:
    items = load_corpus()
    heuristic_run = harness.run(items).as_dict()

    # The deployed guard is the model-backed detector (gateway runs with `--extra guard`), so
    # when it's available it provides the detailed body and the heuristic is the baseline row.
    comparison = [_comparison_row(heuristic_run)]
    run = heuristic_run
    try:
        from agentgate.security import model_guard
        deberta_run = harness.run(
            items, model_guard.scan_text, "deberta-v3-prompt-injection-v2"
        ).as_dict()
        comparison.append(_comparison_row(deberta_run))
        run = deberta_run  # detailed body describes the deployed detector
    except (ImportError, FileNotFoundError):
        comparison = None  # model/extra not present → omit the section, body stays heuristic

    cfg = judge_mod.JudgeConfig()
    cached = judge_mod.cached_labels(cfg.model)
    judge_section = None
    if cached:
        judged = {i: lab.label for i, lab in cached.items()}
        known = {i.id: i.label for i in items
                 if i.label is not None and i.label_origin == LabelOrigin.KNOWN}
        gold = load_gold_set()
        judge_section = {"model": cfg.model, "n_labeled": len(cached)}
        shared_known = set(judged) & set(known)
        if shared_known:
            judge_section["vs_known"] = cohen_kappa(
                {i: judged[i] for i in shared_known}, {i: known[i] for i in shared_known}
            ).as_dict()
        shared_gold = set(judged) & set(gold)
        if shared_gold:
            judge_section["vs_gold"] = cohen_kappa(
                {i: judged[i] for i in shared_gold}, {i: gold[i] for i in shared_gold}
            ).as_dict()

    dest = Path(_args.out) if getattr(_args, "out", None) else ARTIFACT
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(report_mod.render(run, judge_section, comparison))
    try:
        shown = dest.relative_to(REPO_ROOT)
    except ValueError:
        shown = dest
    print(f"wrote {shown}"
          + ("" if judge_section else "  (judge not run yet → preliminary)"))


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m eval.redteam")
    sub = p.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("score", help="offline metrics (no key)")
    sc.add_argument("--detector", default="heuristic", choices=["heuristic", "deberta", "llm-guard"],
                    help="detector to evaluate (deberta needs the `guard` extra + model)")
    sc.set_defaults(fn=cmd_score)
    sub.add_parser("judge", help="run the independent LLM judge (needs key)") \
        .set_defaults(fn=cmd_judge)
    sp = sub.add_parser("sample", help="emit a stratified gold-set template")
    sp.add_argument("-n", type=int, default=150, help="sample size (default 150)")
    sp.add_argument("--seed", type=int, default=0, help="reproducible sample seed")
    sp.set_defaults(fn=cmd_sample)
    rp = sub.add_parser("report", help="render docs/redteam-results.md")
    rp.add_argument("--out", default=None,
                    help="write to PATH instead of docs/redteam-results.md (e.g. a temp file "
                         "to diff before overwriting the deliverable)")
    rp.set_defaults(fn=cmd_report)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
