"""Per-key admission control: bounded concurrency at the front door.

One noisy key must not consume the capacity every other key needs, and the process
must not accept more concurrent work than it can carry. Both are concurrency
questions, so both are answered by holding a slot rather than counting events.

Two levels, acquired in this order:

1. **Per-key semaphore**, cap K. A request that finds the key at its cap waits in a
   bounded FIFO with a deadline: a full queue sheds immediately, and a wait that
   outlives the deadline sheds. Queueing briefly is normal; queueing forever is a
   client-visible hang, so the wait is bounded on both axes.
2. **Global ceiling**, try-acquire only — never waits. On saturation the per-key slot
   is handed back and the request is shed. A waiting request therefore never holds a
   global slot, so one key's queued burst cannot occupy capacity it is not using.

The slot spans admission until the response has fully drained (a streamed answer can
hold one for tens of seconds), so the caller releases the `Lease`, not this module.

State is per-process and dies with the process. Across replicas the effective per-key
bound is K x replicas; exact cluster-wide bounds are a quota question, not a
gateway-local one.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field

# Shed reasons. The caller maps these to a status and an error type.
QUEUE_FULL = "queue_full"
DEADLINE = "deadline"
GLOBAL = "global"


class Shed(Exception):
    """Admission refused. `reason` is one of QUEUE_FULL / DEADLINE / GLOBAL."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass
class _KeyState:
    """One key's live slots. Removed from the registry once both are empty."""

    holders: int = 0
    waiters: deque[asyncio.Future] = field(default_factory=deque)


class Lease:
    """A held admission slot.

    `release` is idempotent because the release paths overlap: a stream can finish
    normally, be abandoned by a disconnecting client, or unwind through an exception,
    and a double release would hand out capacity that is still in use.
    """

    __slots__ = ("_controller", "_key_id", "_released")

    def __init__(self, controller: AdmissionController, key_id: str) -> None:
        self._controller = controller
        self._key_id = key_id
        self._released = False

    @property
    def released(self) -> bool:
        return self._released

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._controller._release(self._key_id)


class AdmissionController:
    """The per-process registry of per-key slots plus the global ceiling.

    Every method runs to completion without awaiting between reading and updating the
    counters, so the checks cannot interleave on one event loop.
    """

    def __init__(
        self,
        *,
        per_key: int,
        global_cap: int,
        queue_depth: int,
        wait_deadline_s: float,
    ) -> None:
        self.per_key = per_key
        self.global_cap = global_cap
        self.queue_depth = queue_depth
        self.wait_deadline_s = wait_deadline_s
        self._keys: dict[str, _KeyState] = {}
        self._global_held = 0

    @classmethod
    def from_settings(cls, settings) -> AdmissionController:
        return cls(
            per_key=settings.admission_per_key,
            global_cap=settings.admission_global,
            queue_depth=settings.admission_queue_depth,
            wait_deadline_s=settings.admission_wait_deadline_s,
        )

    # --- introspection (tests, and anything that wants to report state) ---

    @property
    def global_held(self) -> int:
        return self._global_held

    @property
    def tracked_keys(self) -> list[str]:
        return list(self._keys)

    def holders(self, key_id: str) -> int:
        st = self._keys.get(key_id)
        return st.holders if st is not None else 0

    def queued(self, key_id: str) -> int:
        st = self._keys.get(key_id)
        return len(st.waiters) if st is not None else 0

    # --- acquire / release ---

    async def acquire(self, key_id: str) -> Lease:
        """Take a slot for `key_id`, or raise `Shed`.

        Returns only when both levels are held; every failure path leaves the
        controller exactly as it found it.
        """
        st = self._keys.get(key_id)
        if st is None:
            st = self._keys[key_id] = _KeyState()

        if st.holders < self.per_key and not st.waiters:
            st.holders += 1
        else:
            # Waiting for a queued slot is the only await in here.
            await self._wait_for_key_slot(key_id, st)

        # Global ceiling: a test-and-take with nothing between the two, so this is a
        # try-acquire and never a wait.
        if self._global_held >= self.global_cap:
            self._release_key(key_id)
            raise Shed(GLOBAL)
        self._global_held += 1
        return Lease(self, key_id)

    async def _wait_for_key_slot(self, key_id: str, st: _KeyState) -> None:
        if len(st.waiters) >= self.queue_depth:
            self._evict_if_idle(key_id, st)
            raise Shed(QUEUE_FULL)

        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        st.waiters.append(fut)
        try:
            async with asyncio.timeout(self.wait_deadline_s):
                await fut
        except TimeoutError:
            self._abandon_wait(key_id, st, fut)
            raise Shed(DEADLINE) from None
        except asyncio.CancelledError:
            self._abandon_wait(key_id, st, fut)
            raise
        # Resumed with the slot already transferred by the releasing request:
        # `holders` was never decremented, so there is nothing to take here.

    def _abandon_wait(self, key_id: str, st: _KeyState, fut: asyncio.Future) -> None:
        """Give up a queued wait.

        The slot can arrive in the same event-loop step the deadline fires or the
        request is cancelled; when it did, pass it on rather than leak it.
        """
        if fut.done() and not fut.cancelled():
            self._release_key(key_id)
            return
        try:
            st.waiters.remove(fut)
        except ValueError:
            pass
        self._evict_if_idle(key_id, st)

    def _release(self, key_id: str) -> None:
        """Release both levels. Called only through `Lease.release`."""
        self._global_held -= 1
        self._release_key(key_id)

    def _release_key(self, key_id: str) -> None:
        st = self._keys.get(key_id)
        if st is None:
            return
        # Hand the slot straight to the longest-waiting request. `holders` stays put
        # across a handoff, which is what stops a fresh arrival from jumping the queue.
        while st.waiters:
            fut = st.waiters.popleft()
            if not fut.done():
                fut.set_result(True)
                return
        st.holders -= 1
        self._evict_if_idle(key_id, st)

    def _evict_if_idle(self, key_id: str, st: _KeyState) -> None:
        """Drop a key with nothing in flight.

        `key_id` is whatever the caller's header hashes to — verified against issued
        keys only when that default-off gate is on — so in the default posture an
        unbounded number of them is mintable; a registry that only grows is a
        memory-growth surface.
        """
        if st.holders <= 0 and not st.waiters and self._keys.get(key_id) is st:
            del self._keys[key_id]
