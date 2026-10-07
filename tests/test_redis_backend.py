"""RedisBackend correctness against a real Redis, including two-process behavior.

The limits stack (spend cap, kill switch) is only as shared as its backend; these
tests run every cross-instance claim against a real server. Two backend instances on
separate connections stand in for two gateway replicas.

Skipped unless AGENTGATE_TEST_REDIS_URL is set — `bash scripts/test_redis.sh` starts a
disposable container and runs the suite with it set. Read at import time: the autouse
`hermetic_settings` fixture strips AGENTGATE_* inside tests.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from agentgate.limits.spend import SpendConfig, SpendExceeded, SpendTracker

TEST_REDIS_URL = os.environ.get("AGENTGATE_TEST_REDIS_URL")

pytestmark = pytest.mark.skipif(
    TEST_REDIS_URL is None, reason="AGENTGATE_TEST_REDIS_URL not set (scripts/test_redis.sh)"
)


@pytest.fixture
async def redis_pair():
    """Two RedisBackend instances on separate connections, over a flushed DB."""
    import redis.asyncio as aioredis

    from agentgate.limits.backend import RedisBackend

    clients = [aioredis.from_url(TEST_REDIS_URL, decode_responses=True) for _ in range(2)]
    await clients[0].flushdb()
    try:
        yield RedisBackend(clients[0]), RedisBackend(clients[1])
    finally:
        for c in clients:
            await c.aclose()


async def test_missing_key_reads_zero_and_false(redis_pair):
    a, _ = redis_pair
    assert await a.get("spend:usd:nobody") == 0.0
    assert await a.get_flag("kill:nobody") is False


async def test_incr_is_shared_and_exact_at_the_cap_boundary(redis_pair):
    """Ten increments of $0.10 across two instances read back as exactly 1.0.

    Pins two things: counters are shared (both instances see the same total), and
    INCRBYFLOAT's decimal accumulation hits the cap boundary exactly — the same
    boundary MemoryBackend rounds to agree with (backend.py).
    """
    a, b = redis_pair
    total = 0.0
    for i in range(10):
        total = await (a if i % 2 == 0 else b).incr("spend:usd:k1", 0.10, ttl_s=100)
    assert total == 1.0
    assert await a.get("spend:usd:k1") == 1.0
    assert await b.get("spend:usd:k1") == 1.0


async def test_window_ttl_is_fixed_at_first_write(redis_pair):
    """The spend window must start at the first request and expire whole — a later
    increment must not extend it (expire NX in RedisBackend.incr)."""
    import redis.asyncio as aioredis

    a, _ = redis_pair
    await a.incr("spend:usd:k2", 0.10, ttl_s=100)
    raw = aioredis.from_url(TEST_REDIS_URL, decode_responses=True)
    try:
        first = await raw.ttl("spend:usd:k2")
        await a.incr("spend:usd:k2", 0.10, ttl_s=100)
        second = await raw.ttl("spend:usd:k2")
    finally:
        await raw.aclose()
    assert 0 < first <= 100
    assert second <= first


async def test_flag_propagates_and_clears_across_instances(redis_pair):
    a, b = redis_pair
    await a.set_flag("kill:k3")
    assert await b.get_flag("kill:k3") is True
    await b.clear("kill:k3")
    assert await a.get_flag("kill:k3") is False


async def test_flag_ttl_expires(redis_pair):
    a, _ = redis_pair
    await a.set_flag("kill:k4", ttl_s=1)
    assert await a.get_flag("kill:k4") is True
    for _ in range(40):  # up to ~2 s; TTL is 1 s
        if not await a.get_flag("kill:k4"):
            break
        await asyncio.sleep(0.05)
    assert await a.get_flag("kill:k4") is False


async def test_spend_cap_tripped_on_one_replica_kills_the_key_on_the_other(redis_pair):
    """The cross-replica claim end to end: spend recorded through replica A trips the kill
    switch, and replica B — a separate connection, a separate tracker — refuses the
    key. Then an operator's clear on B is honored by A."""
    a, b = redis_pair
    cfg = SpendConfig(cloud_usd_cap=1.0, window_s=100)
    tracker_a, tracker_b = SpendTracker(a, cfg), SpendTracker(b, cfg)

    await tracker_a.record("key9", is_local=False, cost_usd=1.0)  # hits the cap exactly

    assert await tracker_b.is_killed("key9") is True
    with pytest.raises(SpendExceeded) as exc:
        await tracker_b.check("key9", is_local=False)
    assert exc.value.killed is True

    await tracker_b.clear_kill("key9")
    assert await tracker_a.is_killed("key9") is False


async def test_local_request_cap_is_shared(redis_pair):
    """Local-route request counting is a shared counter too: replicas split the
    window's budget rather than each getting their own."""
    a, b = redis_pair
    cfg = SpendConfig(local_request_cap=4, window_s=100)
    tracker_a, tracker_b = SpendTracker(a, cfg), SpendTracker(b, cfg)

    for i in range(4):
        await (tracker_a if i % 2 == 0 else tracker_b).record("key10", is_local=True, cost_usd=0)
    with pytest.raises(SpendExceeded):
        await tracker_b.check("key10", is_local=True)
