"""Turn garak's dumped prompts (raw JSON) → ``corpus/public/garak.jsonl``. Runs in agentgate's
venv. This is the agentgate side of the Path-1 handoff; the raw JSON comes from
``dump_prompts.py`` (garak's venv).

Each garak prompt becomes a ``CorpusItem`` with:
    label        = 1            (generated attacks are positive by construction)
    label_origin = KNOWN        (the source guarantees the label; survives dedup at rank 2)
    category     = garak:<probe>  (its own per-category recall row)
    source       = garak
    meta         = {probe, expected_miss}  (the split — see probe_map.py)

The output lands in ``corpus/public/`` so ``loader.default_sources()`` globs it in
automatically — no loader edit. Re-running overwrites the file (deterministic given a fixed
PER_PROBE_CAP), so the corpus stays reproducible. Usage (see run-garak.sh):

    uv run python -m eval.redteam.gen.export <raw.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ..loader import PUBLIC_DIR, write_jsonl
from ..schema import CorpusItem, LabelOrigin
from .probe_map import PROBES, category_for

OUT = PUBLIC_DIR / "garak.jsonl"


def build_items(dump: dict[str, list[str]]) -> list[CorpusItem]:
    items: list[CorpusItem] = []
    for probe, prompts in dump.items():
        expected_miss = PROBES.get(probe, False)
        for text in prompts:
            items.append(CorpusItem(
                source="garak",
                text=text,
                label=1,
                label_origin=LabelOrigin.KNOWN,
                category=category_for(probe),
                meta={"probe": probe, "expected_miss": expected_miss},
            ))
    return items


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m eval.redteam.gen.export <raw.json>")
    dump = json.loads(Path(sys.argv[1]).read_text())
    items = build_items(dump)
    n = write_jsonl(OUT, items)
    em = sum(1 for i in items if i.meta["expected_miss"])
    print(f"wrote {n} garak items → {OUT}  ({em} expected_miss / {n - em} plain)")


if __name__ == "__main__":
    main()
