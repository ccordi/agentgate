"""Admission control: the per-key slots, and where the slot is released.

Two halves. The first drives `AdmissionController` directly — acquire/release, per-key
isolation, the shed paths, and the two invariants that are easy to break silently (a
waiting request never holds a global slot; a slot handed to a request that is going away
is passed on, not leaked). The second half goes through the gateway, because the thing
worth pinning there is *when* the slot comes back: a streamed answer holds it until the
stream drains, long after the handler returned.
"""

from __future__ import annotations

import asyncio
import contextlib

import httpx
import pytest

from agentgate.config import Settings
from agentgate.limits.admission import (
    DEADLINE,
    GLOBAL,
    QUEUE_FULL,
    AdmissionController,
    Shed,
)
from agentgate.limits.spend import key_id_from_auth
from tests.support import HARD_INJECTION, sse_handler

SSE_HEADERS = {"content-type": "text/event-stream"}
BODY = {"model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]}


def _controller(**overrides) -> AdmissionController:
    kwargs = {"per_key": 2, "global_cap": 10, "queue_depth": 4, "wait_deadline_s": 0.5}
    kwargs.update(overrides)
    return AdmissionController(**kwargs)


async def _settle() -> None:
    """Let every task that is ready to run reach its next await."""
    await asyncio.sleep(0)
    await asyncio.sleep(0)


# --- controller ------------------------------------------------------------------------

async def test_acquire_takes_both_levels_and_release_returns_them():
    c = _controller(per_key=1, global_cap=1)
    lease = await c.acquire("A")

    assert c.holders("A") == 1
    assert c.global_held == 1

    lease.release()
    assert c.holders("A") == 0
    assert c.global_held == 0


async def test_release_is_idempotent():
    c = _controller(per_key=1, global_cap=1)
    lease = await c.acquire("A")
    lease.release()
    lease.release()

    assert c.global_held == 0
    # The second release must not have credited a slot that was never held: with a cap
    # of one, an over-credited controller would let two requests in here.
    again = await c.acquire("A")
    assert c.global_held == 1
    assert c.holders("A") == 1
    again.release()


async def test_one_saturated_key_does_not_block_another():
    c = _controller(per_key=1, queue_depth=0, global_cap=10)
    held = await c.acquire("A")

    with pytest.raises(Shed) as shed:
        await c.acquire("A")
    assert shed.value.reason == QUEUE_FULL

    other = await c.acquire("B")  # unaffected by A's saturation
    assert c.holders("B") == 1
    other.release()
    held.release()


async def test_queue_full_sheds_without_waiting():
    c = _controller(per_key=1, queue_depth=1, wait_deadline_s=30)
    held = await c.acquire("A")
    waiting = asyncio.create_task(c.acquire("A"))
    await _settle()
    assert c.queued("A") == 1

    # The queue is full, so this one must come back now rather than on the deadline.
    with pytest.raises(Shed) as shed:
        await asyncio.wait_for(c.acquire("A"), 1.0)
    assert shed.value.reason == QUEUE_FULL

    held.release()
    (await waiting).release()


async def test_wait_deadline_sheds():
    c = _controller(per_key=1, queue_depth=4, wait_deadline_s=0.05)
    held = await c.acquire("A")

    with pytest.raises(Shed) as shed:
        await c.acquire("A")
    assert shed.value.reason == DEADLINE
    assert c.queued("A") == 0  # the expired waiter left the queue

    held.release()
    assert c.tracked_keys == []


async def test_released_slot_is_handed_to_the_waiter():
    c = _controller(per_key=1, queue_depth=4, wait_deadline_s=30)
    held = await c.acquire("A")
    waiting = asyncio.create_task(c.acquire("A"))
    await _settle()
    assert c.queued("A") == 1

    held.release()
    lease = await asyncio.wait_for(waiting, 1.0)
    assert c.holders("A") == 1
    assert c.global_held == 1
    lease.release()
    assert c.tracked_keys == []


async def test_global_shed_rolls_back_the_per_key_slot():
    c = _controller(per_key=2, global_cap=1, queue_depth=0)
    held = await c.acquire("A")

    with pytest.raises(Shed) as shed:
        await c.acquire("B")
    assert shed.value.reason == GLOBAL

    # B's per-key slot was taken before the global try-acquire failed; it must be gone.
    assert c.holders("B") == 0
    assert "B" not in c.tracked_keys
    assert c.global_held == 1

    held.release()
    assert c.global_held == 0


async def test_a_waiting_request_holds_no_global_slot():
    c = _controller(per_key=1, global_cap=10, queue_depth=4, wait_deadline_s=30)
    held = await c.acquire("A")
    waiting = asyncio.create_task(c.acquire("A"))
    await _settle()

    assert c.queued("A") == 1
    assert c.global_held == 1  # the holder's, and only the holder's

    held.release()
    lease = await asyncio.wait_for(waiting, 1.0)
    assert c.global_held == 1
    lease.release()


async def test_cancelled_waiter_leaves_no_trace():
    c = _controller(per_key=1, queue_depth=4, wait_deadline_s=30)
    held = await c.acquire("A")
    waiting = asyncio.create_task(c.acquire("A"))
    await _settle()
    assert c.queued("A") == 1

    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert c.queued("A") == 0

    held.release()
    assert c.tracked_keys == []
    assert c.global_held == 0


async def test_slot_handed_to_a_cancelled_waiter_is_passed_on():
    """The race the cancellation path exists for: the slot arrives, then the request
    goes away before it can use it. Releasing to nobody would strand the slot."""
    c = _controller(per_key=1, queue_depth=4, wait_deadline_s=30)
    held = await c.acquire("A")
    first = asyncio.create_task(c.acquire("A"))
    await _settle()
    second = asyncio.create_task(c.acquire("A"))
    await _settle()
    assert c.queued("A") == 2

    held.release()   # hands the slot to `first`, which has not resumed yet
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

    lease = await asyncio.wait_for(second, 1.0)  # the slot reached the next in line
    assert c.holders("A") == 1
    lease.release()
    assert c.tracked_keys == []
    assert c.global_held == 0


async def test_idle_key_is_evicted_but_a_queued_one_is_kept():
    c = _controller(per_key=1, queue_depth=4, wait_deadline_s=30)
    held = await c.acquire("A")
    waiting = asyncio.create_task(c.acquire("A"))
    await _settle()

    assert c.tracked_keys == ["A"]  # holder + waiter: still live
    held.release()
    assert c.tracked_keys == ["A"]  # the waiter now holds it

    (await asyncio.wait_for(waiting, 1.0)).release()
    assert c.tracked_keys == []


def test_unusable_caps_are_rejected_at_construction():
    """Each of these sheds every request the moment admission is armed, which reads as
    a throttled gateway rather than a typo."""
    for override in ({"admission_per_key": 0}, {"admission_global": 0},
                     {"admission_queue_depth": -1}, {"admission_wait_deadline_s": 0.0}):
        with pytest.raises(ValueError):
            Settings(**override)


# --- through the gateway ---------------------------------------------------------------

@pytest.fixture(autouse=True)
def no_leftover_controller():
    """The FastAPI app is module-global, so a controller left on `app.state` by one test
    would be the one the next test's requests are admitted by."""
    from agentgate.app import app as gateway_app

    for state in (gateway_app.state,):
        if hasattr(state, "admission"):
            delattr(state, "admission")
    yield
    if hasattr(gateway_app.state, "admission"):
        delattr(gateway_app.state, "admission")


def _arm(gateway, **overrides) -> AdmissionController:
    """Turn admission on for this test and install the controller its requests use."""
    gateway.settings.admission_enabled = True
    controller = _controller(**overrides)
    gateway.app.state.admission = controller
    return controller


def _gated_upstream(started: asyncio.Event, gate: asyncio.Event):
    """An upstream whose SSE stream stops mid-flight until `gate` is set."""
    async def body():
        started.set()
        yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        await gate.wait()
        yield b'data: {"usage":{"prompt_tokens":1,"completion_tokens":1}}\n\n'
        yield b"data: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body(), headers=SSE_HEADERS)

    return handler


class _Held:
    """A request parked mid-stream, holding its admission slot."""

    def __init__(self, task: asyncio.Task, gate: asyncio.Event) -> None:
        self._task = task
        self._gate = gate

    async def finish(self) -> httpx.Response:
        self._gate.set()
        return await asyncio.wait_for(self._task, 5)


@contextlib.asynccontextmanager
async def _in_flight(gateway, auth: str):
    """Open a request and pause it mid-stream, for as long as the block runs.

    The request is always finished on the way out — a failed assertion inside the block
    should fail the test, not leave a live request wedging the fixture teardown.
    """
    started, gate = asyncio.Event(), asyncio.Event()
    gateway.set_upstream(_gated_upstream(started, gate))
    task = asyncio.create_task(gateway.client.post(
        "/v1/chat/completions", json=BODY, headers={"authorization": auth}))
    held = _Held(task, gate)
    try:
        await asyncio.wait_for(started.wait(), 5)
        yield held
    finally:
        gate.set()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(task, 5)


async def test_admission_is_off_by_default(gateway):
    assert Settings().admission_enabled is False
    assert gateway.settings.admission_enabled is False

    r = await gateway.client.post("/v1/chat/completions", json=BODY,
                                  headers={"authorization": "Bearer t"})

    assert r.status_code == 200
    # Nothing in the request path built one, so nothing consulted one.
    assert getattr(gateway.app.state, "admission", None) is None


async def test_slot_is_held_until_the_stream_drains(gateway):
    controller = _arm(gateway, per_key=1, global_cap=4, queue_depth=0)
    key_id = key_id_from_auth("Bearer t", None)

    async with _in_flight(gateway, "Bearer t") as held:
        # The handler has returned its response by now — the upstream body is only
        # pulled from inside the streaming response — and the slot is still held.
        assert controller.holders(key_id) == 1
        assert controller.global_held == 1
        assert (await held.finish()).status_code == 200

    assert controller.global_held == 0
    assert controller.tracked_keys == []


async def test_second_concurrent_request_for_a_key_sheds_429(gateway):
    controller = _arm(gateway, per_key=1, global_cap=4, queue_depth=0)

    async with _in_flight(gateway, "Bearer t") as held:
        shed = await gateway.client.post("/v1/chat/completions", json=BODY,
                                         headers={"authorization": "Bearer t"})
        assert shed.status_code == 429
        assert shed.json()["error"]["type"] == "admission_queue_full"

        # A second key is admitted while the first is saturated.
        gateway.set_upstream(sse_handler())
        other = await gateway.client.post("/v1/chat/completions", json=BODY,
                                          headers={"authorization": "Bearer other"})
        assert other.status_code == 200
        assert (await held.finish()).status_code == 200

    assert controller.tracked_keys == []


async def test_global_ceiling_sheds_503(gateway):
    controller = _arm(gateway, per_key=4, global_cap=1, queue_depth=0)

    async with _in_flight(gateway, "Bearer t") as held:
        shed = await gateway.client.post("/v1/chat/completions", json=BODY,
                                         headers={"authorization": "Bearer other"})
        assert shed.status_code == 503
        assert shed.json()["error"]["type"] == "admission_capacity"
        # The shed request's per-key slot was rolled back, not left behind.
        assert "other" not in controller.tracked_keys
        assert (await held.finish()).status_code == 200

    assert controller.global_held == 0


async def test_abandoned_stream_releases_the_slot_exactly_once(gateway):
    """A client that hangs up mid-stream must not strand its slot — nor free two."""
    controller = _arm(gateway, per_key=1, global_cap=1, queue_depth=0)
    started, gate = asyncio.Event(), asyncio.Event()
    gateway.set_upstream(_gated_upstream(started, gate))

    request = asyncio.create_task(gateway.client.post(
        "/v1/chat/completions", json=BODY, headers={"authorization": "Bearer t"}))
    await asyncio.wait_for(started.wait(), 5)
    assert controller.global_held == 1

    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request
    await _settle()

    # Zero, not -1: a slot released twice would show up here as an over-credit.
    assert controller.global_held == 0
    assert controller.tracked_keys == []


async def test_blocked_request_releases_its_slot(gateway):
    """A rejection answers with a complete response, so its slot comes back at once."""
    controller = _arm(gateway, per_key=1, global_cap=4, queue_depth=0)

    r = await gateway.client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": HARD_INJECTION}]},
        headers={"authorization": "Bearer t"},
    )

    assert r.status_code == 400
    assert controller.global_held == 0
    assert controller.tracked_keys == []
