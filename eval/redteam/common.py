"""Shared plumbing for the eval scripts — paths, the gateway target, small stats.

The harness proper (``loader``/``schema``/``harness``/``report``) does not need this;
it exists so the traffic drivers and the standalone evals stop each re-deriving the
repo root, the gateway URL, an OpenAI-POST helper, and a percentile function.

Deliberately dependency-light: stdlib + ``httpx`` (already a runtime dep). Nothing here
imports ``bench`` — the two trees stay independent (see ``percentile``).
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence

import httpx

from .loader import PKG_DIR

# Repo root — one definition, imported by __main__, classifier_eval, llm_eval.
REPO_ROOT = PKG_DIR.parents[1]

# The local gateway the traffic drivers point at. Override for a non-default port.
GATEWAY_URL = os.environ.get("AGENTGATE_EVAL_GATEWAY_URL", "http://127.0.0.1:4100")

# Placeholder cloud model for traffic drivers. It only has to exist in
# ``agentgate.pricing.PRICE_TABLE`` so the audit rows carry a cost; the actual
# upstream is expected to be a local mock during volume runs.
EVAL_CLOUD_MODEL = "gemini-2.5-flash"


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile (``p`` in 0–100), matching numpy's default method.

    Twin of ``bench.stats.percentile``. Copied rather than imported on purpose: the eval
    tree does not depend on the bench tree. Keep the two in sync if either changes.
    """
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    rank = (p / 100.0) * (len(s) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    frac = rank - lo
    return float(s[lo] + (s[hi] - s[lo]) * frac)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion ``k/n`` (default 95%).

    Preferred over the normal approximation precisely at the edges this repo reports:
    it stays inside [0, 1] and stays informative at k=0 and k=n, where the Wald
    interval collapses to a point. For 0/109 the upper bound comes out ≈ 0.034 —
    the rule-of-three intuition (≈ 3/n) with the arithmetic done properly.
    """
    if n == 0:
        raise ValueError("wilson_interval is undefined for n == 0")
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def fmt_rate_ci(k: int, n: int, *, counts: bool = True) -> str:
    """``98.6% (71/72, 95% CI 92.5–99.8%)`` — a rate with its Wilson interval.

    The single formatter for every CI-bearing cell, so the report and the standalone
    evals can't drift in how they present uncertainty. ``n == 0`` degrades to a dash.
    """
    if n == 0:
        return "—"
    lo, hi = wilson_interval(k, n)
    inner = f"{k}/{n}, " if counts else ""
    return f"{k / n:.1%} ({inner}95% CI {lo * 100:.1f}–{hi:.1%})"


async def chat_completion(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    **kw,
) -> str:
    """POST one non-streaming Chat Completions request; return the message content.

    Deliberately dumb — no retries, no streaming, no error translation. The callers that
    measure transport behaviour (``route_eval``, ``oss_traffic_gen``) keep their own
    readers, because how they drain a stream is part of what they measure.
    """
    resp = await client.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "messages": messages, **kw},
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]
