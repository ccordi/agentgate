"""Inter-rater agreement — Cohen's κ, hand-rolled (no sklearn).

This is the credibility number for the whole chain: it quantifies how well the independent
judge agrees with the human gold set. Reporting κ (not just raw agreement) accounts for
agreement that would happen by chance, which matters when one class dominates.

Used two ways:
  * judge vs **human gold set** → validates the judge (the headline).
  * judge vs **known-source labels** → a sanity check on authored/public items.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Agreement:
    n: int
    observed: float          # raw agreement rate (p_o)
    expected: float          # chance agreement (p_e)
    kappa: float
    confusion: dict          # {"1_1","1_0","0_1","0_0"} counts (rater_a _ rater_b)

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "observed_agreement": round(self.observed, 4),
            "expected_agreement": round(self.expected, 4),
            "cohens_kappa": round(self.kappa, 4),
            "confusion": self.confusion,
        }


def cohen_kappa(a: dict[str, int], b: dict[str, int]) -> Agreement:
    """Cohen's κ between two raters keyed by item-id, over their shared ids.

    Labels are binary (0/1). Raises if there are no shared ids.
    """
    ids = sorted(set(a) & set(b))
    n = len(ids)
    if n == 0:
        raise ValueError("no overlapping ids between the two raters")

    counts = {"1_1": 0, "1_0": 0, "0_1": 0, "0_0": 0}
    for i in ids:
        counts[f"{int(a[i])}_{int(b[i])}"] += 1

    agree = counts["1_1"] + counts["0_0"]
    p_o = agree / n

    # Marginals → expected agreement by chance.
    a_pos = (counts["1_1"] + counts["1_0"]) / n
    a_neg = 1 - a_pos
    b_pos = (counts["1_1"] + counts["0_1"]) / n
    b_neg = 1 - b_pos
    p_e = a_pos * b_pos + a_neg * b_neg

    if p_e >= 1.0:  # both raters constant → κ undefined; use the degenerate convention
        kappa = 1.0 if p_o == 1.0 else 0.0
    else:
        kappa = (p_o - p_e) / (1 - p_e)

    return Agreement(n=n, observed=p_o, expected=p_e, kappa=kappa, confusion=counts)


def interpret_kappa(kappa: float) -> str:
    """Landis & Koch (1977) qualitative bands — for the report's prose."""
    if kappa < 0:
        return "poor (worse than chance)"
    if kappa < 0.20:
        return "slight"
    if kappa < 0.40:
        return "fair"
    if kappa < 0.60:
        return "moderate"
    if kappa < 0.80:
        return "substantial"
    return "almost perfect"
