"""Spend cap + runaway kill switch (provider-aware).

Guards against runaway agent loops. Two enforcement modes:

- **Cloud** routes accrue an estimated USD spend in a rolling window; crossing the
  cap **trips the kill switch** for that key (a sticky stop, not just a one-request
  rejection — a runaway loop should be halted, not throttled).
- **Local** routes accrue a request count in the window (a rate/compute guard);
  no dollars, so the guard is request volume.

The kill switch can also be tripped manually (ops action) and cleared the same way.
Keys are identified by a short hash of the inbound auth token so raw keys are never
used as Redis keys or logged.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

from agentgate.limits.backend import CounterBackend

log = logging.getLogger("agentgate.limits")


class SpendExceeded(Exception):
    """Raised when a request must be rejected for spend/kill reasons."""

    def __init__(self, reason: str, *, killed: bool = False) -> None:
        super().__init__(reason)
        self.reason = reason
        self.killed = killed


@dataclass
class SpendConfig:
    cloud_usd_cap: float = 5.0  # per window, per key
    local_request_cap: int = 10_000  # per window, per key
    window_s: int = 3600  # rolling window length
    kill_ttl_s: int | None = None  # None = sticky until manually cleared


def key_id_from_auth(authorization: str | None, goog_key: str | None) -> str:
    """Stable short id for a credential, without storing the credential."""
    secret = authorization or goog_key or "anonymous"
    return hashlib.sha256(secret.encode()).hexdigest()[:16]


class SpendTracker:
    def __init__(self, backend: CounterBackend, config: SpendConfig | None = None) -> None:
        self._b = backend
        self.cfg = config or SpendConfig()

    def _spend_key(self, key_id: str) -> str:
        return f"spend:usd:{key_id}"

    def _rate_key(self, key_id: str) -> str:
        return f"spend:local:{key_id}"

    def _kill_key(self, key_id: str) -> str:
        return f"kill:{key_id}"

    async def check(self, key_id: str, is_local: bool) -> None:
        """Pre-forward gate. Raises SpendExceeded if killed or over cap."""
        if await self._b.get_flag(self._kill_key(key_id)):
            raise SpendExceeded("kill switch active for this key", killed=True)

        if is_local:
            count = await self._b.get(self._rate_key(key_id))
            if count >= self.cfg.local_request_cap:
                raise SpendExceeded(
                    f"local request cap reached ({self.cfg.local_request_cap}/window)"
                )
        else:
            spent = await self._b.get(self._spend_key(key_id))
            if spent >= self.cfg.cloud_usd_cap:
                # Trip the kill switch so a runaway loop is halted, not just throttled.
                await self.kill(key_id)
                raise SpendExceeded(
                    f"cloud spend cap reached (${spent:.4f} >= ${self.cfg.cloud_usd_cap})",
                    killed=True,
                )

    async def record(self, key_id: str, is_local: bool, cost_usd: float) -> None:
        """Post-forward accounting. Trips the kill switch if this request pushed the
        accumulated spend over the cap."""
        if is_local:
            await self._b.incr(self._rate_key(key_id), 1, self.cfg.window_s)
            return
        spent = await self._b.incr(self._spend_key(key_id), cost_usd, self.cfg.window_s)
        if spent >= self.cfg.cloud_usd_cap:
            await self.kill(key_id)
            log.warning("spend cap tripped kill switch: key=%s spent=$%.4f", key_id, spent)

    async def kill(self, key_id: str) -> None:
        await self._b.set_flag(self._kill_key(key_id), self.cfg.kill_ttl_s)

    async def is_killed(self, key_id: str) -> bool:
        return await self._b.get_flag(self._kill_key(key_id))

    async def clear_kill(self, key_id: str) -> None:
        await self._b.clear(self._kill_key(key_id))
