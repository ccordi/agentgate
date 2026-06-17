"""Render the injection-detection red-team results — ``docs/redteam-results.md``.

Pure formatting: takes the structured harness/judge results and returns markdown. The
methodology narrative is intentionally prominent — the *labeling + evaluation methodology*
(avoiding circular eval, measuring false negatives, validating the judge against humans)
is the real security-engineering signal, more than the scanner.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from .agreement import interpret_kappa

_MAX_FN_ROWS = 15  # cap the surprise-FN table; full list lives in the run JSON

_METHODOLOGY = """\
## Methodology

The number that matters for a safety scanner is the **honest catch-rate** — including the
attacks it *misses*. Two traps make naive evaluations lie, and this harness is built to
avoid both:

1. **Circularity.** The scanner's own verdict is never treated as ground truth. Labels come
   from an independent chain: *known-source* labels (authored payloads + a labeled public
   dataset) → an **independent LLM judge** (a different model family than the scanner, so
   errors aren't correlated) → a **versioned human gold set** that validates the judge.
2. **The false-negative trap.** If you only inspect what the scanner flagged, you can never
   measure what slipped through. Recall is measured over *labeled positives*, and the gold
   set is sampled across *all* content (flagged **and** unflagged), stratified by scanner
   prediction and source.

The harness drives the same `scan_text` entry point the live gateway uses, so these numbers
describe the deployed detector — and re-running after the scanner moves to a local model
measures that detector with zero changes here.
"""

def _is_model(detector: str) -> bool:
    """True for the model-backed guard (anything but the regex heuristic)."""
    return detector != "heuristic"


def _table(headers: list[str], rows: list[list]) -> str:
    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows)
    return "\n".join([line, sep, body])


# Map oMLX local model tags to the published base-model name for the report output.
# Add entries here if your oMLX profiles use a local tag that differs from the base name.
_PUBLIC_MODEL_NAMES: dict[str, str] = {
    # "my-local-tag": "Published Base Model Name",
}


def _public(tag: str) -> str:
    """Internal serving tag → published base-model name (falls back to the tag)."""
    return _PUBLIC_MODEL_NAMES.get(tag, tag)


def render(run: dict, judge: dict | None = None, comparison: list[dict] | None = None) -> str:
    gen = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    op = run["operating_point"]
    corpus = run["corpus"]

    parts: list[str] = []
    parts.append("# Red-Team Results — Injection Detection\n")
    parts.append(f"> Generated {gen} by `python -m eval.redteam report`. Regenerate with the "
                 "bundled corpus + the frozen FP-capture snapshot (`fp_capture.frozen.jsonl`) + "
                 "cached judge labels. Local judge/guard models (e.g. via oMLX) must be "
                 "configured separately; the DeBERTa baseline runs in-process.\n")

    # Status banner — be explicit about the validation state of the judge.
    if judge and judge.get("vs_gold"):
        k = judge["vs_gold"]["cohens_kappa"]
        parts.append(f"**Status:** judge validated against a human gold set "
                     f"(n={judge['vs_gold']['n']}, Cohen's κ={k:.2f}, {interpret_kappa(k)}).\n")
    else:
        parts.append("**Status:** _preliminary — human gold-set validation pending._ "
                     "Metrics below use known-source labels; the judge-vs-human κ that "
                     "anchors the labeling chain is not yet computed.\n")

    parts.append(_METHODOLOGY)

    # Detector comparison (optional) — heuristic baseline vs. model-backed guard.
    if comparison:
        parts.append("\n## Detector comparison\n")
        parts.append(f"Same corpus, same harness, at the operating threshold "
                     f"{op['threshold']}. The model-backed guard runs in-process (ONNX, no "
                     "torch/server/egress).\n")
        parts.append(_table(
            ["detector", "recall", "precision", "FP rate", "evasions caught", "surprise FN"],
            [[c["detector"], f"{c['recall']:.0%}", f"{c['precision']:.0%}", f"{c['fp_rate']:.0%}",
              f"{c['evasions_caught']}/{c['evasions_total']}", c["surprise_fn"]]
             for c in comparison],
        ))
        parts.append(f"\n*The body below details the **{corpus['detector']}** detector — the "
                     "deployed guard; the heuristic row is the regex baseline.*\n")

    # Corpus composition.
    parts.append("\n## Corpus\n")
    parts.append(_table(
        ["metric", "value"],
        [["labeled items", corpus["total_labeled"]],
         ["positives", corpus["positives"]],
         ["negatives", corpus["negatives"]],
         ["sources", ", ".join(corpus["sources"])]],
    ))

    # Operating point.
    parts.append(f"\n## Catch-rate at the live operating point (threshold = {op['threshold']})\n")
    parts.append(_table(
        ["recall (catch-rate)", "precision", "false-positive rate", "F1", "TP", "FN", "FP", "TN"],
        [[f"{op['recall']:.0%}", f"{op['precision']:.0%}", f"{op['fp_rate']:.0%}",
          f"{op['f1']:.2f}", op["tp"], op["fn"], op["fp"], op["tn"]]],
    ))

    # Threshold sweep.
    parts.append("\n## Threshold sweep (recall / FP tradeoff)\n")
    parts.append(_table(
        ["threshold", "recall", "precision", "FP rate", "TP", "FN", "FP"],
        [[f"{r['threshold']:.2f}", f"{r['recall']:.0%}", f"{r['precision']:.0%}",
          f"{r['fp_rate']:.0%}", r["tp"], r["fn"], r["fp"]] for r in run["sweep"]],
    ))

    # Per-category recall.
    parts.append("\n## Per-category recall\n")
    if _is_model(corpus["detector"]):
        parts.append("Rows prefixed `garak:` are generated single-turn probes (NVIDIA garak); "
                     "rows prefixed `seedmutate:` are the seed-and-mutate harness — "
                     "`doc_embedded`/`tool_result` are local-attacker-model indirect-injection "
                     "attacks (the indirect-injection blind spot), and `enc_*` are the obfuscation "
                     "control. "
                     "Items tagged `expected_miss` — encoding/obfuscation + low-resource "
                     "translation — target the classifier's **tokenization** blind spot, so a "
                     "miss there is documented rather than surprising. Plain rows (the "
                     "document-embedded `latentinjection` / `seedmutate:doc_embedded` / "
                     "`seedmutate:tool_result` families) are forms the guard is "
                     "expected to catch; low recall there is the real signal.\n")
    else:
        parts.append("Categories suffixed `-evasion` / `-nonenglish` / `encoded-` are "
                     "**designed to defeat the heuristic regexes** — low recall there is "
                     "expected and documents the known blind spots a local-model classifier "
                     "should close.\n")
    parts.append(_table(
        ["category", "caught", "total", "recall"],
        [[cat, v["caught"], v["total"], f"{v['recall']:.0%}"]
         for cat, v in run["per_category"].items()],
    ))

    # Honest findings: surprise false negatives.
    em = run["expected_misses"]
    is_model = _is_model(corpus["detector"])
    parts.append("\n## Identified gaps\n")
    if is_model:
        parts.append(f"- **Expected-miss evasions:** {em['total']} obfuscation/encoding items "
                     f"targeting the classifier's tokenization blind spot; "
                     f"{em['caught_anyway']} caught anyway (so the disguise often leaves enough "
                     "legible payload to flag).\n")
    else:
        parts.append(f"- **Expected-miss evasions:** {em['total']} crafted to defeat the "
                     f"regexes; {em['caught_anyway']} caught anyway.\n")
    surprises = run["missed_surprises"]
    if surprises:
        # Break down by source so a large public-dataset miss-rate is legible at a glance.
        by_source: dict[str, int] = {}
        for s in surprises:
            by_source[s.get("source", "?")] = by_source.get(s.get("source", "?"), 0) + 1
        breakdown = ", ".join(f"{src}: {n}" for src, n in sorted(by_source.items()))
        if is_model:
            parts.append(f"- **Surprise false negatives ({len(surprises)})** — plain positives "
                         f"the deployed guard missed (by source: {breakdown}). These are "
                         "genuine misses, not documented blind spots — the real signal. "
                         "Sample:\n")
        else:
            parts.append(f"- **Surprise false negatives ({len(surprises)})** — plain positives "
                         f"that slipped through (by source: {breakdown}). These are the real "
                         "heuristic gaps; the bulk on real-world data is the core argument for "
                         "a local-model classifier. Sample:\n")
        shown = surprises[:_MAX_FN_ROWS]
        parts.append(_table(
            ["source", "category", "missed text (truncated)"],
            # Flatten whitespace + escape pipes so a multi-line / pipe-bearing payload can't
            # break the markdown table row.
            [[s.get("source", "?"), s["category"],
              "`" + " ".join(s["text"].split()).replace("|", "\\|") + "`"]
             for s in shown],
        ))
        if len(surprises) > _MAX_FN_ROWS:
            parts.append(f"\n_…showing {_MAX_FN_ROWS} of {len(surprises)}; full list in the "
                         "run JSON under `eval/redteam/runs/`._\n")
    else:
        parts.append("- No surprise false negatives — every non-evasion positive was caught.\n")

    # Judge validation detail.
    if judge:
        parts.append("\n## Judge validation\n")
        parts.append(f"Judge model: `{_public(judge.get('model', '?'))}` (local, via oMLX or "
                     f"compatible endpoint), labeled {judge.get('n_labeled', 0)} items.\n")
        if judge.get("vs_known"):
            vk = judge["vs_known"]
            parts.append(f"- **vs. known-source labels** (sanity): agreement "
                         f"{vk['observed_agreement']:.0%}, κ={vk['cohens_kappa']:.2f} "
                         f"(n={vk['n']}).\n")
        if judge.get("vs_gold"):
            vg = judge["vs_gold"]
            parts.append(f"- **vs. human gold set** (the credibility anchor): agreement "
                         f"{vg['observed_agreement']:.0%}, κ={vg['cohens_kappa']:.2f} "
                         f"— {interpret_kappa(vg['cohens_kappa'])} (n={vg['n']}).\n")
        else:
            parts.append("- **vs. human gold set:** pending — run "
                         "`python -m eval.redteam sample`, hand-label the emitted file into "
                         "`corpus/gold_set.jsonl`, then re-run `report`.\n")

    # Local-LLM guard section — the 3-model recall/FP table (Gemma / Qwen / DeBERTa),
    # cross-model robustness probes, and latency. Kept as a committed markdown fragment
    # (the Gemma column has no single reproducible run JSON); bundling it keeps `report`
    # regenerate-stable for this section.
    guard_frag = Path(__file__).resolve().parent / "sections" / "local_llm_guard.md"
    if guard_frag.exists():
        parts.append("\n" + guard_frag.read_text(encoding="utf-8").rstrip("\n") + "\n")

    return "\n".join(parts) + "\n"
