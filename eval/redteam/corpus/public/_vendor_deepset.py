"""One-off vendoring script — pins a snapshot of `deepset/prompt-injections`.

Run once to (re)generate `deepset_prompt_injections.jsonl`; the snapshot is committed so
all harness runs stay offline and reproducible. Uses the HF datasets-server REST API (JSON)
so we avoid a heavyweight `datasets`/`pyarrow` dependency.

    python -m eval.redteam.corpus.public._vendor_deepset

License of the source dataset: Apache-2.0 (see SOURCES.md).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import httpx

from eval.redteam.loader import write_jsonl
from eval.redteam.schema import CorpusItem, LabelOrigin

DATASET = "deepset/prompt-injections"
API = "https://datasets-server.huggingface.co/rows"
OUT = Path(__file__).resolve().parent / "deepset_prompt_injections.jsonl"
PAGE = 100


def _fetch_split(client: httpx.Client, split: str) -> list[CorpusItem]:
    items: list[CorpusItem] = []
    offset = 0
    while True:
        r = client.get(API, params={"dataset": DATASET, "config": "default",
                                     "split": split, "offset": offset, "length": PAGE})
        r.raise_for_status()
        payload = r.json()
        for row in payload["rows"]:
            rec = row["row"]
            items.append(CorpusItem(
                source=DATASET,
                text=rec["text"],
                label=int(rec["label"]),
                label_origin=LabelOrigin.KNOWN,
                category="public",
                meta={"split": split},
            ))
        offset += PAGE
        if offset >= payload["num_rows_total"]:
            break
    return items


def main() -> None:
    with httpx.Client(timeout=30.0) as client:
        items = _fetch_split(client, "train") + _fetch_split(client, "test")
    n = write_jsonl(OUT, items)
    sha = hashlib.sha256(OUT.read_bytes()).hexdigest()
    pos = sum(1 for i in items if i.label == 1)
    print(f"wrote {n} items ({pos} positives / {n - pos} negatives) → {OUT.name}")
    print(f"sha256: {sha}")


if __name__ == "__main__":
    main()
