"""Benchmark orchestrator → `docs/benchmarks.md`.

Spins up an **isolated** bench gateway (separate port/DB/provider) in front of the
deterministic mock upstream, runs k6 load phases against both the gateway and the mock
directly (baseline), reads the gateway's own per-stage latency from its audit DB, and
renders the report. **Never touches a running gateway instance.**

    uv run python -m bench.run            # full run (needs k6: `brew install k6`)
    uv run python -m bench.run --quick    # shorter durations for a smoke run

Topology:
    k6 → bench agentgate (:4300, provider=mock, data/bench.db) → mock_upstream (:4200)
    k6 → mock_upstream (:4200) directly         [baseline: the upstream+transport floor]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import httpx

from bench import report, stats

REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = REPO_ROOT / "data" / "bench.db"
RUNS_DIR = REPO_ROOT / "bench" / "runs"
ARTIFACT = REPO_ROOT / "docs" / "benchmarks.md"
SCRIPT = REPO_ROOT / "bench" / "loadtest.js"

MOCK_PORT, GW_PORT = 4200, 4300
MOCK_URL = f"http://127.0.0.1:{MOCK_PORT}/v1/chat/completions"
GW_URL = f"http://127.0.0.1:{GW_PORT}/v1/chat/completions"


def _wait_http(url: str, *, want_200: bool, tries: int = 100) -> bool:
    """Poll until the URL answers (any status) or returns 200 if want_200."""
    for _ in range(tries):
        try:
            r = httpx.get(url, timeout=1.0)
            if not want_200 or r.status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    return False


def _start_processes() -> list[subprocess.Popen]:
    if DB_PATH.exists():
        for suffix in ("", "-wal", "-shm"):
            Path(str(DB_PATH) + suffix).unlink(missing_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    mock = subprocess.Popen(
        ["uv", "run", "python", "-m", "bench.mock_upstream"],
        cwd=REPO_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    gw_env = {
        **os.environ,
        "AGENTGATE_PORT": str(GW_PORT),
        "AGENTGATE_DEFAULT_PROVIDER": "mock",
        # CRITICAL: disable the rules-table router. It is enabled by default, and when
        # enabled it — not AGENTGATE_DEFAULT_PROVIDER — chooses the upstream. A benign
        # bench request falls through to the `default` rule (prefer_cloud → "gemini"), so
        # the gateway would forward every request to the *real* Google Gemini endpoint,
        # which 400s on the fake bench token. That silently routed the whole benchmark at
        # a live cloud API and made every gateway phase show fail% 100%. With routing off,
        # settings.provider() honors AGENTGATE_DEFAULT_PROVIDER=mock. See _smoke_check().
        "AGENTGATE_ROUTING__ENABLED": "false",
        "AGENTGATE_DATABASE_URL": f"sqlite+aiosqlite:///{DB_PATH}",
    }
    gw = subprocess.Popen(
        ["uv", "run", "agentgate"],
        cwd=REPO_ROOT, env=gw_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if not _wait_http(f"http://127.0.0.1:{MOCK_PORT}/", want_200=False):
        raise RuntimeError("mock upstream did not come up on :4200")
    if not _wait_http(f"http://127.0.0.1:{GW_PORT}/healthz", want_200=True):
        raise RuntimeError("bench gateway did not come up on :4300")
    _smoke_check()
    return [gw, mock]


def _smoke_check() -> None:
    """Fail loudly if the gateway isn't actually streaming the mock's SSE back.

    /healthz only proves the process is up — it does NOT exercise the forward path, so a
    mis-routed gateway (e.g. forwarding to a real cloud upstream that rejects the bench
    token) sails past it and silently poisons every number in the report. One real
    streaming request closes that gap: it must return 200, text/event-stream, and the
    terminal `data: [DONE]` from the mock — exactly what the k6 phases check for.
    """
    body = {
        "model": "m",
        "stream": True,
        "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": "smoke"}],
    }
    headers = {"Content-Type": "application/json", "Authorization": "Bearer benchtoken"}
    with httpx.Client(timeout=10.0) as client:
        with client.stream("POST", GW_URL, json=body, headers=headers) as r:
            ctype = r.headers.get("content-type", "")
            text = "".join(r.iter_text())
    if r.status_code != 200 or "text/event-stream" not in ctype or "[DONE]" not in text:
        raise RuntimeError(
            "bench gateway smoke check failed — it is not streaming the mock upstream "
            f"(status={r.status_code}, content-type={ctype!r}, "
            f"saw_DONE={'[DONE]' in text}). The gateway is likely routing elsewhere; "
            "AGENTGATE_ROUTING__ENABLED must be false so AGENTGATE_DEFAULT_PROVIDER=mock "
            f"is honored. First 200 bytes:\n{text[:200]!r}"
        )


def _stop_processes(procs: list[subprocess.Popen]) -> None:
    for p in procs:
        p.terminate()
    for p in procs:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()


def _run_k6(phase: dict) -> dict:
    """Run one k6 phase; return parsed summary metrics."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    summary = RUNS_DIR / f"summary-{phase['name']}.json"
    env = {
        **os.environ,
        "TARGET_URL": phase["target"],
        "MODE": phase.get("mode", "rate"),
        "RATE": str(phase.get("rate", 100)),
        "VUS": str(phase.get("vus", 20)),
        "DURATION": phase["duration"],
        "PAYLOAD": phase.get("payload", "small"),
        "SUMMARY_OUT": str(summary),
    }
    subprocess.run(
        ["k6", "run", "--quiet", str(SCRIPT)],
        cwd=REPO_ROOT, env=env, check=False,  # thresholds may set exit 99; summary still written
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if not summary.exists():
        raise RuntimeError(f"k6 produced no summary for {phase['name']} (is k6 installed?)")
    return _parse_summary(json.loads(summary.read_text()))


def _parse_summary(data: dict) -> dict:
    m = data.get("metrics", {})

    def v(name: str) -> dict:
        return m.get(name, {}).get("values", {})

    dur, wait = v("http_req_duration"), v("http_req_waiting")
    reqs, failed = v("http_reqs"), v("http_req_failed")
    return {
        "reqs": int(reqs.get("count", 0)),
        "rps": round(reqs.get("rate", 0.0), 1),
        "fail_rate": round(failed.get("rate", failed.get("value", 0.0)), 4),
        "dur_p50": dur.get("med"), "dur_p90": dur.get("p(90)"),
        "dur_p95": dur.get("p(95)"), "dur_p99": dur.get("p(99)"), "dur_max": dur.get("max"),
        "ttfb_p50": wait.get("med"), "ttfb_p99": wait.get("p(99)"),
    }


def _per_stage_from_db() -> dict:
    con = sqlite3.connect(DB_PATH)
    try:
        rows = con.execute(
            "SELECT latency_total_ms, latency_upstream_ms, latency_inject_ms FROM requests"
        ).fetchall()
    finally:
        con.close()
    total = [r[0] for r in rows]
    upstream = [r[1] for r in rows]
    inject = [r[2] for r in rows]
    return {
        "rows": len(rows),
        "total_ms": stats.summarize(total),
        "upstream_ms": stats.summarize(upstream),
        "inject_ms": stats.summarize(inject),
        "gateway_internal_ms": stats.summarize(stats.gateway_overhead_ms(total, upstream)),
    }


def _phases(quick: bool) -> list[dict]:
    d = "6s" if quick else "12s"
    return [
        {"name": "gateway_rate_50", "target": GW_URL, "kind": "gateway", "rate": 50, "duration": d},
        {"name": "gateway_rate_100", "target": GW_URL, "kind": "gateway", "rate": 100, "duration": d},
        {"name": "gateway_rate_200", "target": GW_URL, "kind": "gateway", "rate": 200, "duration": d},
        {"name": "gateway_large_100", "target": GW_URL, "kind": "gateway", "rate": 100,
         "duration": d, "payload": "large"},
        {"name": "baseline_rate_100", "target": MOCK_URL, "kind": "baseline", "rate": 100, "duration": d},
    ]


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m bench.run")
    ap.add_argument("--quick", action="store_true", help="shorter durations (smoke run)")
    args = ap.parse_args()

    if shutil.which("k6") is None:
        sys.exit("k6 not found on PATH — install it: `brew install k6`")

    procs = _start_processes()
    try:
        phases = _phases(args.quick)
        for ph in phases:
            print(f"… {ph['name']} ({ph.get('payload', 'small')}, {ph.get('rate')} rps, {ph['duration']})")
            ph["result"] = _run_k6(ph)
        per_stage = _per_stage_from_db()
    finally:
        _stop_processes(procs)

    ARTIFACT.write_text(report.render(phases, per_stage))
    print(f"\nwrote {ARTIFACT.relative_to(REPO_ROOT)}  ({per_stage['rows']} gateway requests measured)")


if __name__ == "__main__":
    main()
