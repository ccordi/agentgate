"""Fire-and-forget background tasks.

The audit write, the spend record, and the content-capture tap all happen *after* the
response has been handed to the client, so none of them may be awaited on the hot path.
`asyncio.create_task` alone isn't enough — without a strong reference the loop may GC a
task mid-flight — so keep one until the task is done.

Lives outside `app` so the pipeline and `egress/api.py` can use it without importing
back into the application module.
"""

from __future__ import annotations

import asyncio
import logging

log = logging.getLogger("agentgate.tasks")

_background: set[asyncio.Task] = set()


def spawn_background(coro) -> None:
    """Schedule ``coro`` off the hot path, holding a reference until it finishes.

    The done-callback logs anything that escaped. Today every spawned coroutine swallows
    its own errors by design, so this should never fire — which is exactly why it is
    here: without it, the *next* fire-and-forget coroutine anyone adds vanishes silently.
    """
    task = asyncio.create_task(coro)
    _background.add(task)

    def _done(t: asyncio.Task) -> None:
        _background.discard(t)
        if not t.cancelled() and t.exception() is not None:
            log.error("background task failed: %r", t.exception())

    task.add_done_callback(_done)


async def drain_background(timeout_s: float = 5.0) -> None:
    """Await in-flight fire-and-forget writes at shutdown.

    Without this, writes still in flight at shutdown are lost on every restart, not only
    on a crash. Bounded so a stuck write cannot hang shutdown.
    """
    if not _background:
        return
    log.info("draining %d in-flight background write(s)…", len(_background))
    try:
        await asyncio.wait_for(
            asyncio.gather(*list(_background), return_exceptions=True), timeout=timeout_s)
    except TimeoutError:
        log.warning("background drain timed out; %d write(s) abandoned", len(_background))
