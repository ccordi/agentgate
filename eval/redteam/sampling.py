"""Stratified sampling for the human gold set.

The gold set validates the judge, so it must include content the scanner did not flag.
Sampling is stratified by scanner prediction and source, then drawn deterministically.
"""

from __future__ import annotations

import random

from agentgate import guards
from agentgate.guards import heuristic

from .schema import CorpusItem, LabelOrigin


def _stratum(item: CorpusItem) -> tuple[str, bool]:
    """Bucket key: (source, predicted-positive-by-scanner-at-flag-threshold)."""
    flagged = heuristic.scan_text(item.text).score >= guards.FLAG_THRESHOLD
    return (item.source, flagged)


def stratified_sample(
    items: list[CorpusItem],
    n: int,
    *,
    seed: int = 0,
    exclude_ids: set[str] | None = None,
) -> list[CorpusItem]:
    """Pick ~``n`` items spread evenly across strata, reproducibly.

    Round-robins across strata (so flagged & unflagged are both represented even when the
    corpus is imbalanced) and shuffles within each stratum by ``seed``.
    """
    exclude_ids = exclude_ids or set()
    rng = random.Random(seed)

    strata: dict[tuple[str, bool], list[CorpusItem]] = {}
    for it in items:
        if it.id in exclude_ids:
            continue
        strata.setdefault(_stratum(it), []).append(it)
    for rows in strata.values():
        rng.shuffle(rows)

    # Round-robin draw across strata until we have n (or exhaust the pool).
    order = sorted(strata.keys())
    picked: list[CorpusItem] = []
    while len(picked) < n and any(strata[k] for k in order):
        for k in order:
            if strata[k]:
                picked.append(strata[k].pop())
                if len(picked) >= n:
                    break
    return picked


def to_gold_template(item: CorpusItem) -> CorpusItem:
    """A blind gold-set line for a human to label from its text.

    Category and scanner score are omitted to avoid revealing the expected label.
    ``id`` is retained so category and source can be joined after labeling.
    """
    return CorpusItem(
        id=item.id,
        source=item.source,  # provenance only; not label-revealing
        text=item.text,
        label=None,  # human fills 0 or 1
        label_origin=LabelOrigin.HUMAN,
        category=None,  # withheld to keep labeling blind
        meta={"_fill": "set label to 0 (benign) or 1 (injection) from the text alone"},
    )
