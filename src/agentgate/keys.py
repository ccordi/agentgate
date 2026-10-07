"""Gateway-issued API keys: mint, verify, revoke.

The verified half of the `key_id` identity. Everything downstream of `key_id`
(admission, spend/kill, guard-backend overrides, audit) works per key; this
module makes the id underneath it verifiable. With `AGENTGATE_REQUIRE_ISSUED_KEYS`
off (the default) nothing here runs on the request path.

Verification is a straight indexed point-lookup per request — deliberately no
in-memory cache: a cache needs an invalidation protocol the moment there are two
replicas on shared Postgres, while the lookup is correct there for free.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import func, select

from agentgate.audit.models import IssuedKey
from agentgate.audit.store import AuditStore
from agentgate.limits.spend import key_id_from_auth

_PREFIX = "ag_"


def _hash(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


def canonical_key_id(secret: str) -> str:
    """The downstream `key_id` for a verified key.

    Derived from the canonical presentation form (`Bearer <secret>`), so it equals
    what `key_id_from_auth` produces for a well-behaved client — and is the same
    however the credential actually arrived (header casing, `x-goog-api-key`).
    """
    return key_id_from_auth(f"Bearer {secret}", None)


@dataclass
class MintedKey:
    """Mint result — the only place the plaintext secret ever appears."""

    secret: str
    key_id: str


class KeyStore:
    """Issued-keys table access, riding the audit store's engine/pool."""

    def __init__(self, store: AuditStore) -> None:
        self._store = store

    @property
    def _sessions(self):
        return self._store.sessionmaker

    async def mint(self, label: str | None = None) -> MintedKey:
        secret = _PREFIX + secrets.token_urlsafe(32)
        row = IssuedKey(key_hash=_hash(secret), key_id=canonical_key_id(secret), label=label)
        async with self._sessions() as session:
            session.add(row)
            await session.commit()
        return MintedKey(secret=secret, key_id=row.key_id)

    async def verify(self, secret: str) -> IssuedKey | None:
        """The row whose hash matches, revoked or not — the caller distinguishes
        revoked from unknown (they get different metrics, the same 401)."""
        async with self._sessions() as session:
            return await session.scalar(
                select(IssuedKey).where(IssuedKey.key_hash == _hash(secret))
            )

    async def revoke(self, key_id: str) -> bool:
        """Revoke by canonical key_id. False if no such key; idempotent otherwise."""
        async with self._sessions() as session:
            row = await session.scalar(select(IssuedKey).where(IssuedKey.key_id == key_id))
            if row is None:
                return False
            if row.revoked_at is None:
                row.revoked_at = datetime.now(UTC)
                await session.commit()
            return True

    async def list_keys(self) -> list[IssuedKey]:
        async with self._sessions() as session:
            result = await session.execute(
                select(IssuedKey).order_by(IssuedKey.created_at.desc())
            )
            return list(result.scalars().all())

    async def count_active(self) -> int:
        async with self._sessions() as session:
            return (
                await session.scalar(
                    select(func.count()).select_from(IssuedKey).where(
                        IssuedKey.revoked_at.is_(None)
                    )
                )
            ) or 0


def key_store(app) -> KeyStore:
    """The app's key store — built per call, never cached. It is a stateless view
    over the audit store's engine (the sessionmaker is the shared resource), and
    caching it on app state would pin whichever audit store existed first, which
    outlives per-test stores on the module-global app."""
    return KeyStore(app.state.audit)
