"""Corpus I/O — read/merge/dedup the JSONL corpus and gold set.

File-based by design (see ``schema.py``). Sources merge by stable content id; when the same
text appears in two sources we keep the higher-provenance label (human > known > judge >
unlabeled) so a labeled copy never loses to an unlabeled one.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from pathlib import Path

from .schema import CorpusItem, LabelOrigin, stable_id

PKG_DIR = Path(__file__).resolve().parent
CORPUS_DIR = PKG_DIR / "corpus"
RUNS_DIR = PKG_DIR / "runs"

AUTHORED = CORPUS_DIR / "authored.jsonl"
PUBLIC_DIR = CORPUS_DIR / "public"
# FP-capture corpus. The live tap keeps the default file growing; set
# AGENTGATE_FP_CAPTURE_PATH to point score/judge/report at a frozen snapshot
# (e.g. a deduped+scrubbed freeze) without mutating or pausing the live tap.
FP_CAPTURE = Path(_p) if (_p := os.environ.get("AGENTGATE_FP_CAPTURE_PATH")) else CORPUS_DIR / "fp_capture.jsonl"
GOLD_SET = CORPUS_DIR / "gold_set.jsonl"
# The Streamlit labeler writes here by default; used as the gold set if the canonical
# gold_set.jsonl isn't present, so labels feed the report without a rename step.
GOLD_SET_LABELED = CORPUS_DIR / "gold_set_labeled.jsonl"
SENSITIVITY_CORPUS = CORPUS_DIR / "sensitivity_corpus.jsonl"

# Higher = more authoritative when merging duplicate texts.
_ORIGIN_RANK = {
    LabelOrigin.HUMAN: 3,
    LabelOrigin.KNOWN: 2,
    LabelOrigin.JUDGE: 1,
    LabelOrigin.UNLABELED: 0,
}


def load_jsonl(path: Path) -> Iterator[CorpusItem]:
    """Yield validated ``CorpusItem``s from a JSONL file (skips blank lines)."""
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield CorpusItem.model_validate_json(line)
            except Exception as exc:  # surface the offending line, don't swallow
                raise ValueError(f"{path.name}:{lineno}: invalid corpus item: {exc}") from exc


def _merge(items: Iterable[CorpusItem]) -> list[CorpusItem]:
    """Dedup by id, keeping the highest-provenance copy of each text."""
    best: dict[str, CorpusItem] = {}
    for item in items:
        prev = best.get(item.id)
        if prev is None or _ORIGIN_RANK[item.label_origin] > _ORIGIN_RANK[prev.label_origin]:
            best[item.id] = item
    return list(best.values())


def default_sources(include_fp_capture: bool = False) -> list[Path]:
    """Committed recall/FP corpus: authored + any vendored public snapshots."""
    paths = [AUTHORED]
    if PUBLIC_DIR.is_dir():
        paths.extend(sorted(PUBLIC_DIR.glob("*.jsonl")))
    if include_fp_capture and FP_CAPTURE.exists():
        paths.append(FP_CAPTURE)
    return [p for p in paths if p.exists()]


def load_corpus(
    paths: list[Path] | None = None,
    *,
    include_fp_capture: bool = True,
    labeled_only: bool = False,
) -> list[CorpusItem]:
    """Load and merge the corpus. ``labeled_only`` drops unlabeled (e.g. raw captures)."""
    if paths is None:
        paths = default_sources(include_fp_capture=include_fp_capture)
    merged = _merge(item for p in paths for item in load_jsonl(p))

    # Apply cached judge labels to unlabeled items (e.g. FP captures). Gate ONLY
    # on `label is None`: an item that already carries a label is gold and must never be
    # silently overwritten by the judge. (`label_origin` defaults to UNLABELED, so a
    # JSONL row that sets `label` but omits `label_origin` would otherwise have its real
    # label flipped to the judge's — corrupting the reported recall/FP numbers.)
    try:
        from .judge import JudgeConfig, cached_labels
        cfg = JudgeConfig()
        labels = cached_labels(cfg.model)
        for item in merged:
            if item.label is None and item.id in labels:
                item.label = labels[item.id].label
                item.label_origin = LabelOrigin.JUDGE
    except Exception as exc:  # judge not configured / cache unreadable — degrade, but say so
        import logging
        logging.getLogger(__name__).warning("judge-label application skipped: %s", exc)

    if labeled_only:
        merged = [i for i in merged if i.is_labeled]
    # Stable ordering for reproducible runs/sampling.
    merged.sort(key=lambda i: i.id)
    return merged


def resolve_gold_path() -> Path:
    """Canonical gold_set.jsonl if present, else the labeler's gold_set_labeled.jsonl."""
    return GOLD_SET if GOLD_SET.exists() else GOLD_SET_LABELED


def load_gold_set(path: Path | None = None) -> dict[str, int]:
    """Map item-id → human label from the gold set (empty if absent).

    Only binary (0/1) labels count — 'Unknown' items (label null) are skipped, so they
    don't pollute the judge-vs-human κ.
    """
    path = path or resolve_gold_path()
    if not path.exists():
        return {}
    gold: dict[str, int] = {}
    for item in load_jsonl(path):
        if item.label is not None:
            gold[item.id] = item.label
    return gold


def write_jsonl(path: Path, items: Iterable[CorpusItem]) -> int:
    """Write items as JSONL (one compact object per line). Returns count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(item.model_dump_json(exclude_defaults=False))
            fh.write("\n")
            n += 1
    return n


def append_capture(path: Path, item: CorpusItem) -> None:
    """Append a single capture line (FP capture). Caller ensures id/text are set."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(item.model_dump_json())
        fh.write("\n")


__all__ = [
    "CORPUS_DIR", "RUNS_DIR", "AUTHORED", "PUBLIC_DIR", "FP_CAPTURE", "GOLD_SET",
    "SENSITIVITY_CORPUS",
    "load_jsonl", "load_corpus", "load_gold_set", "default_sources",
    "write_jsonl", "append_capture", "stable_id",
]
