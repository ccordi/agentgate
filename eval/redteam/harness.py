"""Score an injection scanner over the labeled corpus.

Drives the stable entry point ``agentgate.guards.heuristic.scan_text`` (swaps to a
different backend with no change here). Scores each item once, then evaluates at
multiple thresholds so the report shows the recall/false-positive tradeoff.

Evaluation rules:
  * Recall is measured only over **labeled positives** (no circular self-labeling).
  * **False negatives** are first-class — and we separate *expected* misses (items tagged
    ``expected_miss``: known regex blind spots like base64/translation) from *surprise*
    misses (plain positives that slipped through).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field

from agentgate import guards
from agentgate.guards import heuristic

from .common import wilson_interval
from .schema import CorpusItem

# A detector is anything matching heuristic.scan_text: text -> Verdict (has .score/.reasons).
ScanFn = Callable[[str], object]


@dataclass(frozen=True)
class Scored:
    item: CorpusItem
    score: float
    reasons: list[str]


def score_corpus(items: list[CorpusItem], scan_fn: ScanFn = heuristic.scan_text) -> list[Scored]:
    """Run a detector once per labeled item. Unlabeled items are skipped (no ground truth)."""
    out: list[Scored] = []
    for it in items:
        if it.label is None:
            continue
        v = scan_fn(it.text)
        out.append(Scored(item=it, score=v.score, reasons=v.reasons))
    return out


@dataclass
class Confusion:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def recall(self) -> float:  # true-positive rate
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def fp_rate(self) -> float:  # benign content wrongly flagged
        d = self.fp + self.tn
        return self.fp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def as_dict(self) -> dict:
        return {
            "tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn,
            "recall": round(self.recall, 4), "precision": round(self.precision, 4),
            "fp_rate": round(self.fp_rate, 4), "f1": round(self.f1, 4),
        }


def confusion_at(scored: list[Scored], threshold: float) -> Confusion:
    """Confusion matrix treating ``score >= threshold`` as a positive prediction."""
    c = Confusion()
    for s in scored:
        predicted = s.score >= threshold
        actual = s.item.label == 1
        if actual and predicted:
            c.tp += 1
        elif actual and not predicted:
            c.fn += 1
        elif not actual and predicted:
            c.fp += 1
        else:
            c.tn += 1
    return c


def per_category_recall(scored: list[Scored], threshold: float) -> dict[str, dict]:
    """Recall broken out by attack category (positives only)."""
    buckets: dict[str, list[Scored]] = {}
    for s in scored:
        if s.item.label != 1:
            continue
        buckets.setdefault(s.item.category or "uncategorized", []).append(s)
    out: dict[str, dict] = {}
    for cat, rows in sorted(buckets.items()):
        caught = sum(1 for s in rows if s.score >= threshold)
        out[cat] = {"caught": caught, "total": len(rows), "recall": round(caught / len(rows), 4)}
    return out


def default_thresholds() -> list[float]:
    """Sweep points plus the scanner's configured flag/hard thresholds."""
    pts = {round(x / 10, 1) for x in range(1, 10)}
    pts.add(guards.FLAG_THRESHOLD)
    pts.add(guards.HARD_THRESHOLD)
    return sorted(pts)


@dataclass
class RunResult:
    corpus: dict = field(default_factory=dict)
    operating_point: dict = field(default_factory=dict)  # metrics at the live FLAG_THRESHOLD
    sweep: list = field(default_factory=list)            # [{threshold, ...confusion}]
    per_category: dict = field(default_factory=dict)
    missed_surprises: list = field(default_factory=list)  # real FNs (not expected_miss)
    expected_misses: dict = field(default_factory=dict)   # how the known blind spots fared

    def as_dict(self) -> dict:
        return asdict(self)


def run(items: list[CorpusItem], scan_fn: ScanFn = heuristic.scan_text,
        detector: str = "heuristic") -> RunResult:
    """Full evaluation at the live operating point + threshold sweep + category breakdown."""
    scored = score_corpus(items, scan_fn)
    op_t = guards.FLAG_THRESHOLD

    labeled = [s for s in scored]
    n_pos = sum(1 for s in labeled if s.item.label == 1)
    n_neg = sum(1 for s in labeled if s.item.label == 0)

    # Unanticipated false negatives are missed positives not tagged expected_miss.
    surprises = [
        # Flatten whitespace before truncating: keeps the preview to one line (embedded
        # newlines would otherwise break the markdown table) and spends the 160-char budget
        # on real content, not blank lines.
        {"id": s.item.id, "source": s.item.source, "category": s.item.category,
         "text": " ".join(s.item.text.split())[:160]}
        for s in scored
        if s.item.label == 1 and s.score < op_t and not s.item.meta.get("expected_miss")
    ]

    # Count expected-miss evasions that were caught anyway.
    em = [s for s in scored if s.item.meta.get("expected_miss")]
    em_caught = sum(1 for s in em if s.score >= op_t)

    # Wilson 95% intervals on recall and false-positive rate. Point estimates are unchanged;
    # these say how much of the estimate the corpus actually supports. `None` when the
    # denominator is empty (no positives, or no negatives, at this threshold).
    op_conf = confusion_at(scored, op_t)
    def _ci(k: int, n: int) -> list[float] | None:
        return list(wilson_interval(k, n)) if n else None

    return RunResult(
        corpus={"total_labeled": len(labeled), "positives": n_pos, "negatives": n_neg,
                "detector": detector,
                "sources": sorted({s.item.source for s in scored})},
        operating_point={"threshold": op_t, **op_conf.as_dict(),
                         "recall_ci_95": _ci(op_conf.tp, op_conf.tp + op_conf.fn),
                         "fp_rate_ci_95": _ci(op_conf.fp, op_conf.fp + op_conf.tn)},
        sweep=[{"threshold": t, **confusion_at(scored, t).as_dict()} for t in default_thresholds()],
        per_category=per_category_recall(scored, op_t),
        missed_surprises=surprises,
        expected_misses={"total": len(em), "caught_anyway": em_caught},
    )
