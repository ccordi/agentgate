"""Counter backend for limits (spend cap, rate limit, kill switch).

Two implementations behind one interface:

- ``RedisBackend`` — the real path; counters persist across restarts and processes.
- ``MemoryBackend`` — in-process fallback for single-process deployments without
  Docker/Redis. Used automatically when Redis can't be reached at startup.

The interface is the minimum the limiters need: an atomic float increment with TTL,
a read, and a boolean flag with TTL (the kill switch).
"""

from __future__ import annotations

import logging
import time
from typing import Protocol

log = logging.getLogger("agentgate.limits")


class CounterBackend(Protocol):
    async def incr(self, key: str, amount: float, ttl_s: int) -> float: ...
    async def get(self, key: str) -> float: ...
    async def set_flag(self, key: str, ttl_s: int | None = None) -> None: ...
    async def get_flag(self, key: str) -> bool: ...
    async def clear(self, key: str) -> None: ...


class MemoryBackend:
    """Single-process counters with expiry. Not for multi-worker deployments."""

    def __init__(self) -> None:
        self._vals: dict[str, tuple[float, float | None]] = {}  # key -> (value, expires_at)
        self._flags: dict[str, float | None] = {}  # key -> expires_at

    def _expired(self, expires_at: float | None) -> bool:
        return expires_at is not None and time.monotonic() >= expires_at

    async def incr(self, key: str, amount: float, ttl_s: int) -> float:
        val, exp = self._vals.get(key, (0.0, None))
        if self._expired(exp):
            val = 0.0
            exp = None
        if exp is None:
            exp = time.monotonic() + ttl_s
        val += amount
        self._vals[key] = (val, exp)
        return val

    async def get(self, key: str) -> float:
        val, exp = self._vals.get(key, (0.0, None))
        if self._expired(exp):
            self._vals.pop(key, None)
            return 0.0
        return val

    async def set_flag(self, key: str, ttl_s: int | None = None) -> None:
        self._flags[key] = (time.monotonic() + ttl_s) if ttl_s else None

    async def get_flag(self, key: str) -> bool:
        if key not in self._flags:
            return False
        if self._expired(self._flags[key]):
            self._flags.pop(key, None)
            return False
        return True

    async def clear(self, key: str) -> None:
        self._vals.pop(key, None)
        self._flags.pop(key, None)


class RedisBackend:
    def __init__(self, redis) -> None:
        self._r = redis

    async def incr(self, key: str, amount: float, ttl_s: int) -> float:
        # INCRBYFLOAT then set TTL only on first write (NX) so the window is fixed.
        val = await self._r.incrbyfloat(key, amount)
        await self._r.expire(key, ttl_s, nx=True)
        return float(val)

    async def get(self, key: str) -> float:
        val = await self._r.get(key)
        return float(val) if val is not None else 0.0

    async def set_flag(self, key: str, ttl_s: int | None = None) -> None:
        await self._r.set(key, "1", ex=ttl_s)

    async def get_flag(self, key: str) -> bool:
        return await self._r.exists(key) == 1

    async def clear(self, key: str) -> None:
        await self._r.delete(key)


async def make_backend(redis_url: str) -> CounterBackend:
    """Try Redis; fall back to in-process memory if it's unreachable."""
    try:
        import redis.asyncio as aioredis

        client = aioredis.from_url(redis_url, decode_responses=True)
        await client.ping()
        log.info("limits backend: redis (%s)", redis_url)
        return RedisBackend(client)
    except Exception as exc:  # noqa: BLE001
        log.warning("redis unavailable (%s); using in-process memory backend", exc)
        return MemoryBackend()
