"""Sensitivity-classifier validation + over-fire measurement (offline).

Deterministic and offline: no LLM, network access, or database writes. Run it with:

    uv run python -m eval.redteam classifier-eval          # full report
    uv run python -m eval.redteam classifier-eval --json   # machine-readable

Two halves:

  1. Over-fire on public content. Run classify() over every OSS capture
     (`loader.FP_CAPTURE` — the committed frozen snapshot unless
     AGENTGATE_FP_CAPTURE_PATH points elsewhere; expected sensitivity = none). Every non-none verdict is
     an over-fire. Reported by resulting class and firing hit_type, with the exact
     offending substring so each is root-causable to (a) correct-detect/blunt-policy vs
     (b) genuine detector false-positive.

  2. Detection-floor consistency check on the synthetic sensitivity corpus. The
     generator asserted every item against the same classify()/detect() functions, so
     agreement is true by construction and is not independent validation. Do not use
     the public tier for false-positive claims; it is reported only as a consistency
     check.

The audit DB is read READ-ONLY (sqlite3 immutable URI) purely to corroborate that the
offline over-fire reflects live behaviour; it is skipped cleanly if the DB is absent.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter

from agentgate.redaction import explain
from agentgate.sensitivity import classify

from . import loader
from .common import REPO_ROOT

_DB_PATH = REPO_ROOT / "data" / "agentgate.db"  # not bundled; section [3] skipped if absent

# Tier (`sensitivity` field) -> the classifier class it is constructed to produce.
_TIER_EXPECT = {"public": "none", "sensitive_doc": "pii", "secret_bearing": "secret"}


def _offenders(text: str, hit_type: str) -> list[str]:
    """Return the literal substring(s) that caused `hit_type` to fire in `text`.

    `redaction.explain` owns the which-detector-fired-on-what walk; this only decides
    how to render it. Entropy hits print bare (the token *is* the evidence); pattern hits
    print with surrounding context, so a match can be judged as correct-detect vs
    genuine false-positive without opening the capture.
    """
    spans = [s for t, s in explain(text) if t == hit_type]
    if hit_type == "high_entropy_token":
        return spans[:3]
    out = []
    cursor = 0
    for span in spans[:3]:
        # explain() yields one pattern's matches in textual order, so a forward-only
        # cursor recovers each match's real position (repeated identical spans included).
        s = text.find(span, cursor)
        if s < 0:  # unreachable in practice; degrade to the bare span rather than crash
            out.append(repr(span))
            continue
        e = s + len(span)
        cursor = e
        ctx = text[max(0, s - 25): e + 25].replace("\n", " ")
        out.append(f"{span!r}  …in: …{ctx}…")
    return out


def measure_overfire() -> dict:
    """classify() over every real OSS capture; expected sensitivity = none."""
    if not loader.FP_CAPTURE.exists():
        raise SystemExit(
            f"no FP capture corpus at {loader.FP_CAPTURE}; point "
            "AGENTGATE_FP_CAPTURE_PATH at a capture file, or unset it to use the "
            "committed frozen snapshot."
        )
    items = list(loader.load_jsonl(loader.FP_CAPTURE))
    by_class: Counter = Counter()
    by_hit: Counter = Counter()
    overfires = []
    for it in items:
        r = classify(it.text)
        if r.is_sensitive:
            by_class[str(r.sensitivity)] += 1
            for h in r.hit_types:
                by_hit[h] += 1
            overfires.append(
                {
                    "id": it.id,
                    "class": str(r.sensitivity),
                    "hit_types": r.hit_types,
                    "vector": it.meta.get("vector"),
                    "len": len(it.text),
                    "offenders": {h: _offenders(it.text, h) for h in r.hit_types},
                }
            )
    n = len(items)
    j = len(overfires)
    return {
        "n": n,
        "overfires": j,
        "rate": j / n if n else 0.0,
        "by_class": dict(by_class),
        "by_hit_type": dict(by_hit),
        "detail": overfires,
    }


def detection_floor() -> dict:
    """Consistency check — CIRCULAR, by construction. Sanity floor only."""
    items = list(loader.load_jsonl(loader.SENSITIVITY_CORPUS))
    rows: dict[str, Counter] = {}
    agree = Counter()
    total = Counter()
    for it in items:
        tier = it.sensitivity  # 'public' | 'sensitive_doc' | 'secret_bearing'
        expect = _TIER_EXPECT.get(tier)
        got = str(classify(it.text).sensitivity)
        rows.setdefault(tier, Counter())[got] += 1
        total[tier] += 1
        if got == expect:
            agree[tier] += 1
    return {
        "by_tier": {t: dict(c) for t, c in rows.items()},
        "agreement": {t: f"{agree[t]}/{total[t]}" for t in total},
    }


def db_corroboration() -> dict | None:
    """READ-ONLY live-traffic split, to corroborate the offline rate. None if no DB."""
    if not _DB_PATH.exists():
        return None
    uri = f"file:{_DB_PATH}?mode=ro&immutable=1"
    con = sqlite3.connect(uri, uri=True)
    try:
        cur = con.execute(
            "SELECT agent_id, sensitivity_class, COUNT(*) FROM requests "
            "GROUP BY agent_id, sensitivity_class"
        )
        split: dict[str, Counter] = {}
        for agent, cls, n in cur.fetchall():
            split.setdefault(agent or "(none)", Counter())[cls or "?"] += n
    finally:
        con.close()
    out = {}
    for agent, c in split.items():
        tot = sum(c.values())
        pii = c.get("pii", 0) + c.get("secret", 0) + c.get("private_repo", 0)
        out[agent] = {"split": dict(c), "sensitive": pii, "total": tot,
                      "rate": pii / tot if tot else 0.0}
    return out


def _print_report(of: dict, floor: dict, db: dict | None) -> None:
    p = print
    p("=" * 72)
    p("SENSITIVITY CLASSIFIER VALIDATION (offline, deterministic)")
    p("=" * 72)

    p(f"\n[1] OVER-FIRE on REAL public content  ({loader.FP_CAPTURE.name})")
    p(f"    expected sensitivity = none for all {of['n']} captures")
    p(f"    over-fires: {of['overfires']}/{of['n']} = {of['rate'] * 100:.1f}%")
    p(f"    by resulting class : {of['by_class'] or '{}'}")
    p(f"    by firing hit_type : {of['by_hit_type'] or '{}'}")
    if of["detail"]:
        p("\n    offending captures (root-cause evidence):")
        for d in of["detail"]:
            p(f"      - id={d['id']} {d['class']} {d['hit_types']} "
              f"(vector={d['vector']}, len={d['len']})")
            for h, offs in d["offenders"].items():
                for o in offs:
                    p(f"          [{h}] {o}")

    p("\n[2] DETECTION-FLOOR consistency check  (synthetic sensitivity corpus)")
    p("    ⚠️  CIRCULAR by construction — the generator asserted each item against this")
    p("        same classify(); agreement is guaranteed, NOT independent validation.")
    p("        The `public` tier is OFF-LIMITS for any false-positive claim.")
    for tier, conf in floor["by_tier"].items():
        p(f"    {tier:16s} -> {conf}   agreement {floor['agreement'][tier]}")

    p("\n[3] LIVE-TRAFFIC corroboration  (audit DB, read-only)")
    if db is None:
        p("    (DB absent — skipped)")
    else:
        for agent, s in db.items():
            p(f"    {agent:12s} sensitive {s['sensitive']:>3d}/{s['total']:<3d} "
              f"= {s['rate'] * 100:4.1f}%   split={s['split']}")
        p("    NOTE: the DB classifies whole REQUESTS (classify_request joins the full")
        p("    messages array, <=20k chars); the captures file is PER-TURN fragments.")
        p("    Different unit of analysis -> the two rates are not directly comparable.")
    p("")


def register(sub) -> None:
    """Register the `classifier-eval` subcommand on `python -m eval.redteam`."""
    p = sub.add_parser("classifier-eval", help="sensitivity-classifier over-fire report (offline)")
    p.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    p.set_defaults(fn=main)


def main(args) -> None:
    of = measure_overfire()
    floor = detection_floor()
    db = db_corroboration()

    if args.json:
        print(json.dumps({"overfire": of, "detection_floor": floor,
                          "db_corroboration": db}, indent=2))
    else:
        _print_report(of, floor, db)


if __name__ == "__main__":  # direct invocation delegates to the same main()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    main(ap.parse_args())
