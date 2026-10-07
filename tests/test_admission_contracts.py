"""Admission control under load: ordering, cascades, and the one-step races.

`test_admission.py` pins each contract once, on the smallest case that shows it. This file
attacks the same contracts where they are most likely to come apart: a queue several
requests deep rather than one, a global ceiling that sheds a whole queue one handoff at a
time, the single event-loop step in which a handed-off slot meets an expiring deadline,
and a seeded mix of every operation with the invariants re-checked after each step.

Interleavings are built out of event-loop steps, not elapsed time. The two deadline races
are the exception, and there the deadline is the thing under test: the loop is held inside
one step past the moment the timer comes due, so the step the timer finally fires in is
the step the test chose for it.
"""

from __future__ import annotations

import asyncio
import random

import pytest

from agentgate.limits.admission import (
    DEADLINE,
    GLOBAL,
    QUEUE_FULL,
    AdmissionController,
    Lease,
    Shed,
)


def _controller(**overrides) -> AdmissionController:
    kwargs = {"per_key": 2, "global_cap": 10, "queue_depth": 4, "wait_deadline_s": 0.5}
    kwargs.update(overrides)
    return AdmissionController(**kwargs)


async def _settle() -> None:
    """Let every task that is ready to run reach its next await."""
    await asyncio.sleep(0)
    await asyncio.sleep(0)


async def _drain(rounds: int = 8) -> None:
    """Run the loop until a chain of handoffs has finished rippling through the queue.

    One handoff costs a step: the released slot wakes a waiter, the waiter resumes and
    either takes it or hands it on again, and each link runs in the step after the last.
    """
    for _ in range(rounds):
        await asyncio.sleep(0)


def _block_loop_until(loop: asyncio.AbstractEventLoop, when: float) -> None:
    """Hold the loop inside the current step until `when`.

    Busy-waiting instead of sleeping is the whole point: while this runs the loop cannot
    reach an expired deadline's callback, so the deadline stays due-but-unfired until the
    caller has set up the step it wants it to fire in.
    """
    while loop.time() < when:
        pass


async def _serve(
    controller: AdmissionController,
    key_id: str,
    tag: int,
    order: list[int],
    leases: dict[int, Lease],
) -> Lease:
    """Acquire, recording the order requests are actually served in."""
    lease = await controller.acquire(key_id)
    order.append(tag)
    leases[tag] = lease
    return lease


# --- arrival order ----------------------------------------------------------------------

async def test_five_waiters_are_served_in_arrival_order():
    """The per-key wait is FIFO the whole way down the queue, not just for the first."""
    c = _controller(per_key=1, queue_depth=5, wait_deadline_s=30)
    order: list[int] = []
    leases: dict[int, Lease] = {}
    held = await c.acquire("A")

    waiters = []
    for tag in range(5):
        waiters.append(asyncio.create_task(_serve(c, "A", tag, order, leases)))
        await _settle()  # one arrival per step, so the tags are the arrival order
    assert c.queued("A") == 5

    current = held
    for expected in range(5):
        current.release()
        await _drain()
        assert order == list(range(expected + 1))
        current = leases[expected]
    current.release()

    assert order == [0, 1, 2, 3, 4]
    assert all(waiter.done() for waiter in waiters)
    assert c.global_held == 0
    assert c.tracked_keys == []


async def test_arrival_order_holds_when_several_slots_free_in_one_step():
    """Three slots released back to back inside one step go to the first three waiters."""
    c = _controller(per_key=3, queue_depth=6, wait_deadline_s=30)
    order: list[int] = []
    leases: dict[int, Lease] = {}
    holders = [await c.acquire("A") for _ in range(3)]

    waiters = []
    for tag in range(5):
        waiters.append(asyncio.create_task(_serve(c, "A", tag, order, leases)))
        await _settle()
    assert c.queued("A") == 5

    for holder in holders:
        holder.release()  # three handoffs, none of them yet resumed
    await _drain()

    assert order == [0, 1, 2]
    assert c.holders("A") == 3
    assert c.queued("A") == 2

    leases[0].release()
    leases[1].release()
    await _drain()
    assert order == [0, 1, 2, 3, 4]
    assert all(waiter.done() for waiter in waiters)

    for lease in leases.values():
        lease.release()
    assert c.global_held == 0
    assert c.tracked_keys == []


async def test_a_fresh_arrival_queues_behind_the_existing_waiters():
    """A request that shows up last is served last, however short the queue is."""
    c = _controller(per_key=1, queue_depth=4, wait_deadline_s=30)
    order: list[int] = []
    leases: dict[int, Lease] = {}
    held = await c.acquire("A")

    early = []
    for tag in (0, 1):
        early.append(asyncio.create_task(_serve(c, "A", tag, order, leases)))
        await _settle()
    latecomer = asyncio.create_task(_serve(c, "A", 2, order, leases))
    await _settle()
    assert c.queued("A") == 3

    held.release()
    await _drain()
    assert order == [0]
    assert not latecomer.done()

    leases[0].release()
    await _drain()
    assert order == [0, 1]

    leases[1].release()
    await _drain()
    assert order == [0, 1, 2]
    assert all(waiter.done() for waiter in [*early, latecomer])

    leases[2].release()
    assert c.global_held == 0
    assert c.tracked_keys == []


async def test_arrivals_that_land_as_a_slot_frees_still_go_to_the_back():
    """Arrival order is service order across a mix of releases and new requests.

    Each newcomer arrives in the same step a slot is handed off, which is the moment the
    key looks free: `holders` is only left at the cap by the handoff itself.
    """
    c = _controller(per_key=2, queue_depth=6, wait_deadline_s=30)
    order: list[int] = []
    leases: dict[int, Lease] = {}
    holders = [await c.acquire("A") for _ in range(2)]

    requests = []
    for tag in (0, 1):
        requests.append(asyncio.create_task(_serve(c, "A", tag, order, leases)))
        await _settle()

    holders[0].release()
    requests.append(asyncio.create_task(_serve(c, "A", 2, order, leases)))  # mid-handoff
    await _drain()
    assert order == [0]
    assert c.queued("A") == 2

    holders[1].release()
    requests.append(asyncio.create_task(_serve(c, "A", 3, order, leases)))
    await _drain()
    assert order == [0, 1]
    assert c.queued("A") == 2

    leases[0].release()
    await _drain()
    leases[1].release()
    await _drain()

    assert order == [0, 1, 2, 3]
    assert all(request.done() for request in requests)
    leases[2].release()
    leases[3].release()
    assert c.global_held == 0
    assert c.tracked_keys == []


async def test_holders_is_unchanged_across_a_handoff():
    """The mechanism the queue rests on: a handoff transfers the slot without ever
    dropping `holders`, which is what leaves no gap for an arrival to take."""
    c = _controller(per_key=2, queue_depth=4, wait_deadline_s=30)
    first = await c.acquire("A")
    second = await c.acquire("A")
    waiting = asyncio.create_task(c.acquire("A"))
    await _settle()
    assert (c.holders("A"), c.queued("A")) == (2, 1)

    first.release()
    # The waiter has not resumed yet, but its slot is already spoken for.
    assert (c.holders("A"), c.queued("A")) == (2, 0)

    lease = await asyncio.wait_for(waiting, 1.0)
    assert c.holders("A") == 2

    lease.release()
    second.release()
    assert c.global_held == 0
    assert c.tracked_keys == []


# --- the global ceiling -------------------------------------------------------------------

async def test_a_global_shed_hands_the_key_slot_down_the_queue():
    """A waiter that reaches the global gate and sheds passes its per-key slot on rather
    than dropping it, so a saturated ceiling drains the queue without losing a slot."""
    c = _controller(per_key=1, global_cap=2, queue_depth=4, wait_deadline_s=30)
    order: list[int] = []
    leases: dict[int, Lease] = {}
    on_a = await c.acquire("A")
    on_b = await c.acquire("B")  # the ceiling is now full
    waiters = []
    for tag in range(3):
        waiters.append(asyncio.create_task(_serve(c, "A", tag, order, leases)))
        await _settle()
    assert c.queued("A") == 3

    on_a.release()  # frees a global slot and hands A's key slot to the first waiter
    # `acquire` on an unseen key never awaits, so this runs to completion inside the
    # current step and takes the freed global slot before any waiter can resume.
    stealer = await c.acquire("C")
    assert c.global_held == 2
    await _drain(12)

    for waiter in waiters:
        with pytest.raises(Shed) as shed:
            waiter.result()
        assert shed.value.reason == GLOBAL
    assert order == []

    # Every shed rolled its state back: the key slot walked the whole queue and was
    # given up exactly once at the end of it.
    assert c.holders("A") == 0
    assert "A" not in c.tracked_keys
    assert c.global_held == 2

    on_b.release()
    stealer.release()
    assert c.global_held == 0
    assert c.tracked_keys == []


async def test_a_global_slot_freed_mid_cascade_stops_the_shedding():
    """The cascade is a chain of ordinary handoffs, so room appearing part-way through it
    admits the next waiter in line instead of shedding it."""
    c = _controller(per_key=1, global_cap=2, queue_depth=4, wait_deadline_s=30)
    order: list[int] = []
    leases: dict[int, Lease] = {}
    on_a = await c.acquire("A")
    on_b = await c.acquire("B")
    waiters = []
    for tag in range(3):
        waiters.append(asyncio.create_task(_serve(c, "A", tag, order, leases)))
        await _settle()

    on_a.release()
    stealer = await c.acquire("C")  # ceiling full again before waiter 0 resumes
    await asyncio.sleep(0)  # waiter 0 sheds and hands its key slot to waiter 1
    with pytest.raises(Shed) as shed:
        waiters[0].result()
    assert shed.value.reason == GLOBAL

    stealer.release()  # room again, still before waiter 1 resumes
    await _drain(12)

    assert order == [1]
    assert c.holders("A") == 1
    assert c.queued("A") == 1  # waiter 2 is still behind waiter 1
    assert c.global_held == 2

    leases[1].release()
    await _drain()
    assert order == [1, 2]

    leases[2].release()
    on_b.release()
    assert c.global_held == 0
    assert c.tracked_keys == []


# --- the deadline / handoff race ----------------------------------------------------------

async def test_a_handoff_in_the_step_a_deadline_comes_due_serves_the_waiter():
    """Handoff side of the race: the slot arrives first, so the waiter is admitted and
    the deadline never fires."""
    c = _controller(per_key=1, queue_depth=4, wait_deadline_s=0.05)
    loop = asyncio.get_running_loop()
    held = await c.acquire("A")
    armed_at = loop.time()
    waiting = asyncio.create_task(c.acquire("A"))
    await _settle()
    assert c.queued("A") == 1
    assert not waiting.done()

    # Releasing from inside the blocked step queues the waiter's resumption ahead of the
    # timer callback, which the loop only reaches once this step ends.
    _block_loop_until(loop, armed_at + 0.06)
    held.release()
    await _drain()

    lease = waiting.result()
    assert c.holders("A") == 1
    assert c.global_held == 1

    lease.release()
    assert c.global_held == 0
    assert c.tracked_keys == []


async def test_a_deadline_firing_on_an_already_granted_slot_passes_it_on():
    """Deadline side of the race: the slot is granted, the deadline fires before the
    waiter can resume, and the slot it never used goes to the next in line."""
    c = _controller(per_key=1, queue_depth=4, wait_deadline_s=0.05)
    loop = asyncio.get_running_loop()
    held = await c.acquire("A")
    armed_at = loop.time()
    first = asyncio.create_task(c.acquire("A"))
    await _settle()
    assert c.queued("A") == 1
    assert not first.done()

    _block_loop_until(loop, armed_at + 0.06)
    # Next step, in order: `second` queues with a fresh deadline, the release hands the
    # slot to `first`, and only then does `first`'s expired timer run.
    second = asyncio.create_task(c.acquire("A"))
    loop.call_soon(held.release)
    await _drain(12)

    with pytest.raises(Shed) as shed:
        first.result()
    assert shed.value.reason == DEADLINE

    lease = second.result()
    assert c.holders("A") == 1
    assert c.global_held == 1

    lease.release()
    assert c.global_held == 0
    assert c.tracked_keys == []


# --- cancellation -------------------------------------------------------------------------

async def test_cancellation_storm_keeps_the_survivors_in_arrival_order():
    """Requests going away mid-queue, while slots are being handed down it, must not
    reorder or strand the ones that stay."""
    rng = random.Random(20260805)
    c = _controller(per_key=1, global_cap=20, queue_depth=16, wait_deadline_s=30)
    order: list[int] = []
    leases: dict[int, Lease] = {}
    held = await c.acquire("A")

    tasks: dict[int, asyncio.Task] = {}
    for tag in range(12):
        tasks[tag] = asyncio.create_task(_serve(c, "A", tag, order, leases))
        await _settle()
    assert c.queued("A") == 12

    cancelled: set[int] = set()
    to_release: Lease | None = held
    while to_release is not None:
        # Cancel part of the queue in the step before the next slot is handed down it.
        for tag, task in tasks.items():
            if tag not in cancelled and not task.done() and rng.random() < 0.25:
                task.cancel()
                cancelled.add(tag)
        served_before = len(order)
        to_release.release()
        await _drain(12)
        to_release = leases[order[-1]] if len(order) > served_before else None

    for task in tasks.values():
        task.cancel()
    await asyncio.gather(*tasks.values(), return_exceptions=True)

    # The tags are the arrival order, so sorted order *is* arrival order.
    assert order == sorted(order)
    assert cancelled.isdisjoint(order)
    assert len(order) >= 3 and len(cancelled) >= 3  # the run exercised both outcomes
    # Nobody was stranded: the queue emptied because every request was either served or
    # cancelled, not because a slot went missing part-way down it.
    assert set(order) | cancelled == set(range(12))
    for tag in cancelled:
        assert tasks[tag].cancelled()

    for lease in leases.values():
        lease.release()
    await _drain()
    assert c.global_held == 0
    assert c.tracked_keys == []


# --- everything at once -------------------------------------------------------------------

async def test_a_random_operation_mix_never_breaks_an_invariant():
    """A seeded mix of acquires, releases, cancellations and expiries across six keys,
    with every invariant re-checked after each settled step."""
    rng = random.Random(20260805)
    c = _controller(per_key=2, global_cap=4, queue_depth=3, wait_deadline_s=0.01)
    keys = [f"k{i}" for i in range(6)]
    pending: list[asyncio.Task] = []
    leases: list[Lease] = []

    def harvest() -> None:
        """Take the leases off finished acquires, and consume their sheds."""
        unfinished = []
        for task in pending:
            if not task.done():
                unfinished.append(task)
            elif not task.cancelled():
                exc = task.exception()
                if exc is None:
                    leases.append(task.result())
                else:
                    assert isinstance(exc, Shed), exc
        pending[:] = unfinished

    def check(step: int) -> None:
        live = [lease for lease in leases if not lease.released]
        assert 0 <= c.global_held <= c.global_cap, step
        assert len(live) == c.global_held, step  # only unreleased leases hold the ceiling
        for key in keys:
            assert 0 <= c.holders(key) <= c.per_key, (step, key)
            assert 0 <= c.queued(key) <= c.queue_depth, (step, key)
            # Nobody waits on a key with room to spare, which is what leaves no gap for
            # an arrival to jump into.
            assert not (c.queued(key) and c.holders(key) < c.per_key), (step, key)
        for key in c.tracked_keys:
            assert c.holders(key) or c.queued(key), (step, key)

    for step in range(420):
        roll = rng.random()
        if roll < 0.45:
            pending.append(asyncio.create_task(c.acquire(rng.choice(keys))))
        elif roll < 0.72:
            live = [lease for lease in leases if not lease.released]
            if live:
                rng.choice(live).release()
        elif roll < 0.88:
            if pending:
                rng.choice(pending).cancel()
        elif roll < 0.97:
            await _drain(rng.randint(1, 3))
        else:
            await asyncio.sleep(0.015)  # long enough for the queued waits to expire
        await _drain(rng.randint(1, 3))
        harvest()
        check(step)

    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    harvest()
    for lease in leases:
        lease.release()
    await _drain(20)

    assert c.global_held == 0
    assert c.tracked_keys == []


# --- registry growth ----------------------------------------------------------------------

async def test_no_shed_path_leaves_a_key_behind():
    """Key ids are whatever the caller's header hashes to, so every path that creates key
    state has to drop it again — including the three that end in a shed."""
    c = _controller(per_key=1, global_cap=2, queue_depth=1, wait_deadline_s=0.02)
    blocker = await c.acquire("blocker")

    for i in range(200):
        (await c.acquire(f"served-{i}")).release()
    assert c.tracked_keys == ["blocker"]

    for i in range(50):
        key = f"full-{i}"
        held = await c.acquire(key)
        waiting = asyncio.create_task(c.acquire(key))
        await _settle()
        with pytest.raises(Shed) as shed:
            await c.acquire(key)
        assert shed.value.reason == QUEUE_FULL
        held.release()
        (await asyncio.wait_for(waiting, 1.0)).release()
    assert c.tracked_keys == ["blocker"]

    for i in range(5):
        key = f"late-{i}"
        held = await c.acquire(key)
        with pytest.raises(Shed) as shed:
            await asyncio.wait_for(c.acquire(key), 1.0)
        assert shed.value.reason == DEADLINE
        held.release()
    assert c.tracked_keys == ["blocker"]

    filler = await c.acquire("filler")  # the ceiling is now full
    for i in range(200):
        with pytest.raises(Shed) as shed:
            await c.acquire(f"global-{i}")
        assert shed.value.reason == GLOBAL
    assert sorted(c.tracked_keys) == ["blocker", "filler"]

    blocker.release()
    filler.release()
    assert c.global_held == 0
    assert c.tracked_keys == []


async def test_a_shed_request_never_took_a_global_slot():
    """Each shed happens either before the global gate or with the take rolled back, so
    none of them can be the reason the next request finds the ceiling full."""
    c = _controller(per_key=1, global_cap=4, queue_depth=1, wait_deadline_s=0.02)
    held = await c.acquire("A")
    assert c.global_held == 1

    waiting = asyncio.create_task(c.acquire("A"))
    await _settle()
    with pytest.raises(Shed) as queue_full:
        await c.acquire("A")
    assert queue_full.value.reason == QUEUE_FULL
    assert c.global_held == 1

    with pytest.raises(Shed) as deadline:
        await asyncio.wait_for(waiting, 1.0)
    assert deadline.value.reason == DEADLINE
    assert c.global_held == 1

    held.release()
    assert c.global_held == 0
    assert c.tracked_keys == []
