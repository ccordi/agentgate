# Red-team eval suite

Measures how well the injection guard actually catches attacks — including the ones it
misses. Everything here is offline unless a command says otherwise; the corpus files are
committed snapshots so a fresh clone reproduces the offline numbers with no keys and no
model downloads.

## The two tracks

```
  corpus generation                          labeling chain
  ─────────────────                          ──────────────
  authored payloads ─────┐                   known-source labels
  deepset (vendored) ────┤                        │
  garak probes ──────────┼──▶ corpus/*.jsonl ──▶  ├── independent LLM judge
  seed-and-mutate ───────┤        (loader)        │   (different model family
  capture tap ───────────┘                        │    than the scanner)
                                                  │
                                                  └── human gold set ──▶ κ
                                                        (validates the judge)
                                                              │
                                    score ◀─────────────────── ┘
                                      │
                                      └──▶ report ──▶ docs/redteam-results.md
```

Two errors can inflate a scanner evaluation. **Circularity:** the scanner's own verdict is
never ground truth — labels come from the chain above, and the judge uses a different
model family to reduce, though not eliminate, correlated-error risk. **False-negative
sampling:** if you only inspect what the scanner
flagged, you can't measure what slipped through, so recall is computed over labeled
positives and the gold set is sampled across flagged *and* unflagged content.

`harness.py` also splits false negatives two ways: an *expected miss* (tagged
`meta.expected_miss` — a documented blind spot like base64 or low-resource translation)
is reported in its own bucket, separate from unanticipated misses. `gen/probe_map.py`
defines that split and explains each choice.

## Commands

All of these are `uv run python -m eval.redteam <cmd>`.

| Command | What it does | Needs |
| --- | --- | --- |
| `score [--detector heuristic\|deberta\|llm-guard]` | offline metrics over the labeled corpus; writes a run JSON | nothing (deberta needs the `guard` extra + model) |
| `report [--out PATH]` | renders `docs/redteam-results.md` from scores + judge cache + gold set | nothing |
| `sample [-n N] [--seed S]` | emits a stratified gold-set template for a human to label | nothing |
| `classifier-eval [--json]` | sensitivity-classifier over-fire report, with the offending substrings | nothing |
| `judge` | runs the independent LLM judge to label items | `AGENTGATE_JUDGE_API_KEY` |
| `llm-eval` | local-LLM guard vs the DeBERTa baseline, by sub-corpus | local model server |
| `probes-eval` | obfuscation + security-meta probes | local model server |
| `route-eval [--per-tier N]` | drives the sensitivity corpus through a live gateway; routing is then read out of the audit DB | running gateway |
| `traffic --smoke\|--volume` | drives real OSS content through a live gateway | running gateway |
| `gen sensitivity [--out PATH]` | regenerates the synthetic sensitivity corpus (deterministic) | nothing |
| `gen seed-mutate [-n N]` | regenerates the seed-and-mutate attacks (**not** deterministic) | local attacker model |

Each module keeps its own `main(args)`, so `python -m eval.redteam.<module>` still works
and runs the same code. Point the traffic drivers somewhere else with
`AGENTGATE_EVAL_GATEWAY_URL`.

Four scripts stay outside the CLI on purpose. `gen/run-garak.sh`, `gen/dump_prompts.py`
and `gen/export.py` straddle two venvs — garak needs its own Python 3.12 install and is
deliberately not a project dependency — and `corpus/public/_vendor_deepset.py` is a
one-off vendoring script. Their headers explain the boundary. The gold-set labeler is a
Streamlit app, so it has its own launcher too:

```bash
uv run --extra label streamlit run eval/redteam/label_app.py
```

## Reproducing the published numbers

| Claim | Command | Offline? |
| --- | --- | --- |
| corpus-wide heuristic / DeBERTa recall, precision, FP | `score`, `score --detector deberta` | yes |
| garak `latentinjection` recall (LLM guard vs DeBERTa) | `llm-eval` | needs a local model server |
| 0/109 FP on the scanned (tool-output) channel | `llm-eval` — the benign split of the same run | needs a local model server |
| sensitivity-classifier over-fire rate | `classifier-eval` | yes |
| overhead / latency | `bench/run.py` → `docs/benchmarks.md` | needs k6 |

`score` and `report` carry 95% Wilson intervals next to recall and FP rate. The point
estimates are unchanged; the interval is there because several of these sub-corpora are
small enough (n=28 to n=153) that the bare percentage reads more precise than the sample
supports. Each statistic these commands emit has a single implementation: the Wilson
helpers in `common.py`, κ in `agreement.py`, the confusion-matrix arithmetic in
`harness.py`.

The corpus files are frozen snapshots — `corpus/public/SOURCES.md` records where each one
came from, which are re-fetchable byte-for-byte and which are not, and the sha256 of each.
`corpus/fp_capture.frozen.jsonl` is anonymized real traffic and is frozen outright.
The numbers in `sections/local_llm_guard.md` are a measured snapshot stapled into the
report; `report` does not regenerate them, and its header says so.

Two caveats matter when quoting these results:

- The synthetic sensitivity corpus validates itself against the same `classify()` it is
  used to exercise. That circularity is real, is stated in `gen/sensitivity_corpus.py` and
  again in `classifier_eval.py`, and means the public tier is off-limits for any
  false-positive claim.
- seed_mutate recall has the same circularity risk — the attacker model and the guard
  can be the same family. Treat the independent garak result as stronger evidence.

## Layout

```
loader.py schema.py      corpus I/O and the on-disk JSONL contract
harness.py               scoring: confusion matrices, threshold sweep, FN split
judge.py agreement.py    the independent judge and Cohen's κ
report.py sections/      markdown rendering + the committed frozen fragment
label_app.py             Streamlit gold-set labeler (blind: text only, no scores)
common.py                shared paths, gateway URL, percentile, Wilson interval
gen/                     corpus generators (garak export, seed-and-mutate, sensitivity)
corpus/                  the committed snapshots — see corpus/public/SOURCES.md
runs/                    run JSONs and caches (gitignored)
```
