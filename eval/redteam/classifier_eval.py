"""Sensitivity-classifier validation + over-fire measurement (offline).

Deterministic, offline, re-runnable. NO LLM, NO network, NO DB writes. Run it with:

    uv run python -m eval.redteam.classifier_eval          # full report
    uv run python -m eval.redteam.classifier_eval --json   # machine-readable

Two halves:

  1. OVER-FIRE on REAL public content. Run classify() over every real OSS capture
     (fp_capture.jsonl, expected sensitivity = none). Every non-none verdict is
     an over-fire. Reported by resulting class and by firing hit_type, WITH the exact
     offending substring so each is root-causable to (a) correct-detect/blunt-policy vs
     (b) genuine detector false-positive.

  2. DETECTION-FLOOR consistency check on the synthetic sensitivity corpus. ⚠️ This is
     CIRCULAR: the generator asserted every item against this same classify()/detect(), so
     the agreement is true BY CONSTRUCTION and is NOT independent validation. The public
     tier is OFF-LIMITS for any false-positive claim. Reported here only as a sanity floor.

The audit DB is read READ-ONLY (sqlite3 immutable URI) purely to corroborate that the
offline over-fire reflects live behaviour; it is skipped cleanly if the DB is absent.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

from agentgate.security.classifier import classify
from agentgate.security.redaction import (
    _B64_BLOB_RE,
    _ENTROPY_BITS,
    _PII_PATTERNS,
    _SECRET_PATTERNS,
    _TOKEN_RE,
    _is_b64_secret,
    _shannon_bits,
)

from . import loader

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DB_PATH = _REPO_ROOT / "data" / "agentgate.db"  # not bundled; section [3] skipped if absent

# Tier (`sensitivity` field) -> the classifier class it is constructed to produce.
_TIER_EXPECT = {"public": "none", "sensitive_doc": "pii", "secret_bearing": "secret"}

# Map a firing hit_type to the detector that produced it, so we can pull the exact
# offending substring out of an over-firing text for root-causing.
_HIT_RE = {name: pat for pat, name in (*_SECRET_PATTERNS, *_PII_PATTERNS)}


def _offenders(text: str, hit_type: str) -> list[str]:
    """Return the literal substring(s) that caused `hit_type` to fire in `text`."""
    if hit_type == "high_entropy_token":
        # detect() flags this via TWO branches; mirror both so the offender is never
        # blank: (a) a generic alnum/_/- run over the entropy floor, and (b) a base64
        # blob carrying `+`/`/` (e.g. `nv/<hex>`) that _is_b64_secret accepts.
        a = [t for t in _TOKEN_RE.findall(text) if _shannon_bits(t) >= _ENTROPY_BITS]
        b = [t for t in _B64_BLOB_RE.findall(text) if _is_b64_secret(t)]
        return (a + b)[:3]
    pat = _HIT_RE.get(hit_type)
    if pat is None:
        return []
    out = []
    for m in pat.finditer(text):
        s, e = m.span()
        ctx = text[max(0, s - 25): e + 25].replace("\n", " ")
        out.append(f"{m.group(0)!r}  …in: …{ctx}…")
    return out[:3]


def measure_overfire() -> dict:
    """classify() over every real OSS capture; expected sensitivity = none."""
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

    p("\n[1] OVER-FIRE on REAL public content  (fp_capture.jsonl)")
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args()

    of = measure_overfire()
    floor = detection_floor()
    db = db_corroboration()

    if args.json:
        print(json.dumps({"overfire": of, "detection_floor": floor,
                          "db_corroboration": db}, indent=2))
    else:
        _print_report(of, floor, db)


if __name__ == "__main__":
    main()
