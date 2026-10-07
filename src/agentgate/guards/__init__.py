"""Inbound prompt-injection guards: one verdict type, one driver, three backends.

`scan(backend, messages)` is the only entry point the pipeline needs. It owns
extraction — deciding *which* text each backend sees — because the three backends
deliberately do not agree on that:

- `heuristic` (always available) and `deberta` scan `content.extract_untrusted`:
  the trailing batch of tool outputs **and** the newest user turn.
- `llm` scans only `content.tool_output_texts` — never the operator's own turn.

That asymmetry is measured and documented (docs/threat-model.md); it is a property
of the design, not an oversight. Selecting a backend therefore selects a scan surface,
a coverage bound, and a verdict shape all at once — see the per-backend module
docstrings.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field

from agentgate import content

log = logging.getLogger("agentgate.guards")

# Recognized backend names — shared by `guard_backend`/`guard_backend_overrides`
# validation (config.py imports this) and by `scan`'s dispatch below.
BACKENDS = frozenset({"deberta", "llm", "combined", "heuristic"})

# Which optional model dependency each backend needs loaded. `heuristic` needs none, and
# `combined` needs both. Startup warmup and /readyz both ask this table rather than
# testing backend names themselves — a new backend declares its needs in one place.
_REQUIRES = {
    "deberta": frozenset({"deberta"}),
    "llm": frozenset({"llm"}),
    "combined": frozenset({"deberta", "llm"}),
}

# The heuristic scanner's thresholds; the classifier sets its own (`guards.deberta`).
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
    # How many untrusted items the backend actually examined. A clean verdict over
    # 0 items ("scanned nothing") must stay distinguishable from a clean verdict over 3
    # ("scanned and found nothing"). `worst` stamps it from the driver's own extractor,
    # since the backends deliberately scan different surfaces.
    scanned_items: int = 0
    # `gap_abandoned:<pattern>:<distance>` per bounded-gap heuristic pattern whose
    # prefix was seen with its suffix past the bound. Never in `reasons`, never in the
    # score — an audit caveat, concatenated across items and backends by `worst` and
    # `combine`.
    gap_abandoned: list[str] = field(default_factory=list)

    @classmethod
    def clean(cls) -> Verdict:
        return cls(flagged=False, score=0.0, reasons=[], hard=False)


class GuardUnavailable(Exception):
    """A blocking safety control could not run.

    Fail closed *with a distinguishable status*: a control that cannot run must not look
    like a control that ran and found nothing, and must not surface as a bare framework
    500 either.
    """

    def __init__(self, backend: str, cause: BaseException) -> None:
        super().__init__(f"guard backend {backend!r} unavailable: {cause!r}")
        self.backend = backend
        self.cause = cause


def requires(settings, dependency: str) -> bool:
    """Whether any backend this process can resolve to needs ``dependency``.

    "Can resolve to" is the global default plus every per-key override — a key pinned to
    `llm` still has to work when the default is `heuristic`. Callers ask about a
    dependency, never about a backend name, so adding a backend touches `_REQUIRES` and
    nothing else.
    """
    configured = {settings.guard_backend, *settings.guard_backend_overrides.values()}
    return any(dependency in _REQUIRES.get(b, frozenset()) for b in configured)


def resolve_backend(settings, key_id: str | None) -> str:
    """The effective backend for ``key_id``: its override, else the global default.

    Shared by `scan` (dispatch) and the pipeline (audit) so the audited backend always
    matches what actually ran. No availability fallback: a backend whose model will not
    load refuses startup (`app.lifespan`).
    """
    return settings.guard_backend_overrides.get(key_id, settings.guard_backend)


def worst(verdicts: Iterable[tuple[str, Verdict]]) -> Verdict:
    """The strongest verdict from [(source, verdict)], with reasons source-prefixed.

    ``scanned_items`` is stamped on the way out, not inside the loop: the loop only
    replaces the running best on a HIGHER score, so a request where nothing scored would
    otherwise report 0 items — the exact ambiguity the field exists to remove. The same
    goes for ``gap_abandoned``: it is collected from every item, not just the winner.
    """
    out = Verdict.clean()
    n = 0
    gaps: list[str] = []
    for source, v in verdicts:
        n += 1
        gaps.extend(v.gap_abandoned)
        if v.score > out.score:
            out = Verdict(v.flagged, v.score, [f"{source}:{r}" for r in v.reasons], v.hard)
    out.scanned_items = n
    out.gap_abandoned = gaps
    return out


def combine(a: Verdict, b: Verdict) -> Verdict:
    """Merge two backends' verdicts: flag/hard by OR, score by max, reasons concatenated."""
    return Verdict(
        flagged=a.flagged or b.flagged,
        score=max(a.score, b.score),
        reasons=a.reasons + b.reasons,
        hard=a.hard or b.hard,
        # The two surfaces differ (deberta: tool batch + newest user turn; llm: tool
        # output only), so max() answers the question the column is for — did anything
        # get looked at — rather than double-counting the overlap.
        scanned_items=max(a.scanned_items, b.scanned_items),
        gap_abandoned=a.gap_abandoned + b.gap_abandoned,
    )


async def scan(backend: str, messages: list[dict]) -> Verdict:
    """Scan a request's untrusted content with ``backend``; return the strongest verdict.

    The model backends run in a worker thread so their inference (latency on the
    Results page) never blocks the event loop; the heuristic is inline. Backend modules
    are imported lazily — deberta needs the optional `guard` extra.

    Raises :class:`GuardUnavailable` when a model-backed path fails: the local model
    server down/slow/507, an unparseable judge response, an unset judge key, an
    onnxruntime failure at scan time. Fail closed, but say so.
    """
    try:
        if backend == "deberta":
            from agentgate.guards import deberta
            return await asyncio.to_thread(_scan_sync, deberta.scan_text, messages)
        if backend == "llm":
            from agentgate.guards import local_llm
            return await asyncio.to_thread(
                _scan_sync, local_llm.scan_text, messages, tool_only=True)
        if backend == "combined":
            from agentgate.guards import deberta, local_llm
            deb, llm = await asyncio.gather(
                asyncio.to_thread(_scan_sync, deberta.scan_text, messages),
                asyncio.to_thread(_scan_sync, local_llm.scan_text, messages, tool_only=True),
            )
            return combine(deb, llm)
    except Exception as exc:
        log.error("guard backend %s failed: %r", backend, exc)
        raise GuardUnavailable(backend, exc) from exc

    # The heuristic is pure regex over a bounded string with no dependency to lose, so it
    # is deliberately outside the try: if *it* raises, that is a bug, not an outage.
    from agentgate.guards import heuristic
    return _scan_sync(heuristic.scan_text, messages)


def _scan_sync(scan_text, messages: list[dict], *, tool_only: bool = False) -> Verdict:
    """Apply a per-text scanner across this request's untrusted content.

    ``tool_only`` selects the LLM judge's narrower surface (tool output, no user turn).
    That backend is where ``scanned_items`` earns its keep: a request whose newest tool
    batch has already been answered — or that carries none at all — leaves the surface
    empty, and forwards on a clean verdict having been scanned by nothing.
    """
    extract = content.tool_output_texts if tool_only else content.extract_untrusted
    return worst((source, scan_text(text)) for source, text in extract(messages))
