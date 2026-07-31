"""Fire-and-forget background tasks.

The audit write, the spend record, and the content-capture tap all happen *after* the
response has been handed to the client, so none of them may be awaited on the hot path.
`asyncio.create_task` alone isn't enough — without a strong reference the loop may GC a
task mid-flight — so keep one until the task is done.
"""

from __future__ import annotations

import asyncio

_background: set[asyncio.Task] = set()


def spawn_background(coro) -> None:
    """Schedule ``coro`` off the hot path, holding a reference until it finishes."""
    task = asyncio.create_task(coro)
    _background.add(task)
    task.add_done_callback(_background.discard)
