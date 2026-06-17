"""Passive traffic capture tap.

When the tagged capture agent runs (identified by the ``/a/<agent_id>`` tagged route), append
the untrusted content it encountered — tool/web outputs and the user turn — to the eval corpus
for later judge-labeling. This is the benign / false-positive sampling path: real, in-the-wild
traffic from one intentionally-non-sensitive agent, used to characterize the scanner's
false-positive rate.

Scope guardrails:
  * **Off by default** (``AGENTGATE_CAPTURE_ENABLED``); only the configured capture agent
    is captured. The capture agent fetches public pages and is told to ignore embedded
    instructions, so its content is intentionally non-sensitive.
  * **Off the hot path** — called fire-and-forget; file I/O runs in a worker thread.
  * **No dependency on the eval package** — agentgate must not import eval tooling, so we
    write the corpus JSONL shape directly (kept in sync with eval/redteam/schema.py).
  * Captured items are **unlabeled** (``label_origin: ""``); the judge labels them later.
    The scanner's own verdict is recorded only as telemetry, never as a label (avoids the
    circularity the red-team methodology is built to prevent).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path

from . import injection

log = logging.getLogger("agentgate.capture")


def _stable_id(text: str) -> str:
    # Mirrors eval.redteam.schema.stable_id so captures dedup against the rest of the corpus.
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _append(path: str, lines: list[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


async def capture(path: str, agent_id: str, messages: list[dict]) -> None:
    """Append the request's untrusted content to the FP-capture JSONL.

    Captures even when the request is blocked inbound — a tripped scanner on real web
    content is exactly the high-value sample. Errors are swallowed (best-effort telemetry).
    """
    try:
        lines: list[str] = []
        for vector, text in injection.extract_untrusted(messages):
            if not text:
                continue
            v = injection.scan_text(text)
            rec = {
                "id": _stable_id(text),
                "source": "capture",
                "text": text,
                "label": None,
                "label_origin": "",
                "category": None,
                "meta": {
                    "agent_id": agent_id,
                    "vector": vector,  # "tool_output" | "user"
                    "scanner_score": round(v.score, 3),
                    "scanner_flagged": v.flagged,
                    "ts": time.time(),
                },
            }
            lines.append(json.dumps(rec, ensure_ascii=False))
        if lines:
            await asyncio.to_thread(_append, path, lines)
    except Exception as exc:  # never let capture break a request
        log.warning("traffic capture failed: %s", exc)
