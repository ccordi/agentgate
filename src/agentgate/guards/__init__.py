"""Inbound prompt-injection guards: one verdict type, one driver, three backends.

`scan(backend, messages)` is the only entry point the pipeline needs. It owns
extraction — deciding *which* text each backend sees — because the three backends
deliberately do not agree on that:

- `heuristic` (always available) and `deberta` scan `content.extract_untrusted`:
  the trailing batch of tool outputs **and** the newest user turn.
- `llm` scans only `content.tool_output_texts` — never the operator's own turn.

That asymmetry is measured and documented (docs/index.md, "How I measured this");
it is a property of the design, not an oversight. Selecting a backend therefore
selects a scan surface, a coverage bound, and a verdict shape all at once — see the
per-backend module docstrings.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass

from agentgate import content

# Recognized backend names — shared by `guard_backend`/`guard_backend_overrides`
# validation (config.py imports this) and by `scan`'s dispatch below.
BACKENDS = frozenset({"deberta", "llm", "combined", "heuristic"})

# Which optional model dependency each backend needs loaded. `heuristic` needs none, and
# `combined` needs both. Startup warmup asks this table rather than testing backend names
# itself — a new backend declares its needs in one place.
_REQUIRES = {
    "deberta": frozenset({"deberta"}),
    "llm": frozenset({"llm"}),
    "combined": frozenset({"deberta", "llm"}),
}

# Score at/above this blocks the request inbound (cheap, fully-buffered side).
HARD_THRESHOLD = 0.7
# Score at/above this flags-and-logs but forwards.
FLAG_THRESHOLD = 0.4


@dataclass
class Verdict:
    flagged: bool
    score: float
    reasons: list[str]
    hard: bool  # True -> block the request inbound

    @classmethod
    def clean(cls) -> Verdict:
        return cls(flagged=False, score=0.0, reasons=[], hard=False)


def requires(settings, dependency: str) -> bool:
    """Whether any backend this process can resolve to needs ``dependency``.

    "Can resolve to" is the global default plus every per-key override — a key pinned to
    `llm` still has to work when the default is `heuristic`. Callers ask about a
    dependency, never about a backend name, so adding a backend touches `_REQUIRES` and
    nothing else.
    """
    configured = {settings.guard_backend, *settings.guard_backend_overrides.values()}
    return any(dependency in _REQUIRES.get(b, frozenset()) for b in configured)


def resolve_backend(settings, key_id: str | None, deberta_available: bool) -> str:
    """Resolve the effective backend for ``key_id`` (override, else default), applying
    the per-request deberta-availability fallback. Shared by `scan` (dispatch) and the
    pipeline (audit) so the audited backend always matches what actually ran."""
    backend = settings.guard_backend_overrides.get(key_id, settings.guard_backend)
    if not deberta_available:
        if backend == "combined":
            backend = "llm"
        elif backend == "deberta":
            backend = "heuristic"
    return backend


def worst(verdicts: Iterable[tuple[str, Verdict]]) -> Verdict:
    """The strongest verdict from [(source, verdict)], with reasons source-prefixed."""
    out = Verdict.clean()
    for source, v in verdicts:
        if v.score > out.score:
            out = Verdict(v.flagged, v.score, [f"{source}:{r}" for r in v.reasons], v.hard)
    return out


def combine(a: Verdict, b: Verdict) -> Verdict:
    """Merge two backends' verdicts: flag/hard by OR, score by max, reasons concatenated."""
    return Verdict(
        flagged=a.flagged or b.flagged,
        score=max(a.score, b.score),
        reasons=a.reasons + b.reasons,
        hard=a.hard or b.hard,
    )


async def scan(backend: str, messages: list[dict]) -> Verdict:
    """Scan a request's untrusted content with ``backend``; return the strongest verdict.

    The model backends run in a worker thread so their ~10-30 ms (deberta) or ~1.4 s
    (llm) inference never blocks the event loop; the heuristic is inline. Backend
    modules are imported lazily — deberta and the LLM guard have optional deps.
    """
    if backend == "deberta":
        from agentgate.guards import deberta
        return await asyncio.to_thread(_scan_sync, deberta.scan_text, messages)
    if backend == "llm":
        from agentgate.guards import local_llm
        return await asyncio.to_thread(_scan_sync, local_llm.scan_text, messages, tool_only=True)
    if backend == "combined":
        from agentgate.guards import deberta, local_llm
        deb, llm = await asyncio.gather(
            asyncio.to_thread(_scan_sync, deberta.scan_text, messages),
            asyncio.to_thread(_scan_sync, local_llm.scan_text, messages, tool_only=True),
        )
        return combine(deb, llm)

    from agentgate.guards import heuristic
    return _scan_sync(heuristic.scan_text, messages)


def _scan_sync(scan_text, messages: list[dict], *, tool_only: bool = False) -> Verdict:
    """Apply a per-text scanner across this request's untrusted content.

    ``tool_only`` selects the LLM guard's narrower surface (tool output, no user turn).
    """
    extract = content.tool_output_texts if tool_only else content.extract_untrusted
    return worst((source, scan_text(text)) for source, text in extract(messages))
