"""Drive the sensitivity corpus through the live gateway with routing enabled.

Sends each item to the gateway as a chat-completions request under an unpinned
agent_id, so the router forks it by sensitivity: sensitive (pii/secret) → local oMLX,
public (none) → the cloud branch. The gateway audits route_provider / route_is_local /
tokens — this script does NOT self-tally verdicts; the audit DB is the source of truth.

⚠️ Point the cloud branch at a local mock (zero real egress) before running volume tests.
Drains every SSE stream fully (no fire-and-forget early-close → no keepalive stubs).

    uv run python -m eval.redteam.route_eval --agent route-eval            # full 150
    uv run python -m eval.redteam.route_eval --agent route-eval-smoke --per-tier 1
"""

from __future__ import annotations

import argparse
import time
from collections import Counter

import httpx

from . import loader

GATEWAY = "http://127.0.0.1:4100"


def _items(per_tier: int | None):
    items = list(loader.load_jsonl(loader.SENSITIVITY_CORPUS))
    if per_tier is None:
        return items
    seen: Counter = Counter()
    out = []
    for it in items:
        tier = it.sensitivity
        if seen[tier] < per_tier:
            out.append(it)
            seen[tier] += 1
    return out


def _send(client: httpx.Client, agent: str, text: str) -> tuple[int, int]:
    """POST one item, DRAIN the stream fully. Returns (status, sse_data_lines)."""
    url = f"{GATEWAY}/a/{agent}/v1/chat/completions"
    payload = {
        "model": "gemini-2.5-flash",  # documents the counterfactual cloud model
        "messages": [{"role": "user", "content": text}],
        "stream": True,
        "stream_options": {"include_usage": True},  # make oMLX emit a usage chunk to tap
    }
    headers = {"Authorization": "Bearer eval-2c", "content-type": "application/json"}
    lines = 0
    with client.stream("POST", url, json=payload, headers=headers, timeout=300.0) as r:
        status = r.status_code
        for line in r.iter_lines():  # drain to completion — no early close
            if line:
                lines += 1
    return status, lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agent", default="route-eval", help="unpinned agent_id (NOT capture)")
    ap.add_argument("--per-tier", type=int, default=None, help="cap items per tier (smoke test)")
    ap.add_argument("--sleep", type=float, default=0.3, help="inter-request sleep (oMLX concurrency=1)")
    args = ap.parse_args()

    items = _items(args.per_tier)
    by_tier = Counter(it.sensitivity for it in items)
    print(f"driving {len(items)} items through {GATEWAY}/a/{args.agent} — tiers {dict(by_tier)}")
    print("(sensitive→local oMLX generation; public→mock canned. Local tier dominates runtime.)")

    t0 = time.perf_counter()
    status_counts: Counter = Counter()
    client = httpx.Client()
    try:
        for i, it in enumerate(items, 1):
            try:
                status, lines = _send(client, args.agent, it.text)
            except Exception as exc:  # noqa: BLE001 — report and continue, don't abort the run
                print(f"  [{i}/{len(items)}] {it.sensitivity:14s} ERROR {exc!r}")
                status_counts["error"] += 1
                continue
            status_counts[status] += 1
            if i % 10 == 0 or args.per_tier is not None:
                print(f"  [{i}/{len(items)}] {it.sensitivity:14s} status={status} sse_lines={lines}")
            time.sleep(args.sleep)
    finally:
        client.close()
    dt = time.perf_counter() - t0
    print(f"done in {dt:.0f}s — status counts {dict(status_counts)}")
    print(f"verify in the audit DB (read-only): query the requests table, filter agent_id={args.agent!r} and inspect route_provider / route_is_local")


if __name__ == "__main__":
    main()
