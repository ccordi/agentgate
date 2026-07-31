"""Tests for the content tier — crypto, capture decisions, sweeper."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from cryptography.fernet import Fernet

from agentgate.audit.crypto import ContentCipher
from agentgate.audit.store import AuditStore, ContentSampleAudit
from agentgate.pipeline import capture_content
from tests.support import FAKE_OPENAI_KEY, SECRET_PROMPT, SOFT_INJECTION, wait_for_audit_row

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _store(db_path) -> AuditStore:
    return AuditStore(f"sqlite+aiosqlite:///{db_path}")


def _sample(request_id=None, expires_at=None, reason="random") -> ContentSampleAudit:
    now = datetime.now(UTC)
    return ContentSampleAudit(
        request_id=request_id or uuid.uuid4(),
        ts=now,
        role="user",
        redacted_content_enc=b"placeholder",
        sampled_reason=reason,
        expires_at=expires_at or (now + timedelta(days=30)),
    )


# ---------------------------------------------------------------------------
# ContentCipher tests
# ---------------------------------------------------------------------------

def test_cipher_round_trip():
    key = Fernet.generate_key().decode()
    c = ContentCipher(key)
    assert c.enabled
    enc = c.encrypt("hello world")
    assert isinstance(enc, bytes)
    assert c.decrypt(enc) == "hello world"


def test_cipher_disabled_when_no_key():
    c = ContentCipher(None)
    assert not c.enabled


def test_cipher_raises_when_disabled():
    c = ContentCipher(None)
    with pytest.raises(RuntimeError):
        c.encrypt("x")
    with pytest.raises(RuntimeError):
        c.decrypt(b"x")


def test_cipher_different_plaintexts_produce_different_tokens():
    key = Fernet.generate_key().decode()
    c = ContentCipher(key)
    assert c.encrypt("aaa") != c.encrypt("bbb")


# ---------------------------------------------------------------------------
# AuditStore.write_content_sample / purge_expired_content
# ---------------------------------------------------------------------------

async def test_write_and_count_content_sample(tmp_path):
    store = _store(tmp_path / "c.db")
    await store.init()
    await store.write_content_sample(_sample())
    assert await store.count_content_samples() == 1
    await store.close()


async def test_purge_deletes_only_expired(tmp_path):
    store = _store(tmp_path / "p.db")
    await store.init()
    now = datetime.now(UTC)
    # One expired, one future
    await store.write_content_sample(_sample(expires_at=now - timedelta(seconds=1)))
    await store.write_content_sample(_sample(expires_at=now + timedelta(days=30)))
    deleted = await store.purge_expired_content()
    assert deleted == 1
    assert await store.count_content_samples() == 1
    await store.close()


async def test_purge_returns_zero_when_nothing_expired(tmp_path):
    store = _store(tmp_path / "z.db")
    await store.init()
    await store.write_content_sample(_sample())
    assert await store.purge_expired_content() == 0
    await store.close()


async def test_purge_deletes_all_expired(tmp_path):
    store = _store(tmp_path / "all.db")
    await store.init()
    now = datetime.now(UTC)
    for _ in range(3):
        await store.write_content_sample(_sample(expires_at=now - timedelta(seconds=1)))
    deleted = await store.purge_expired_content()
    assert deleted == 3
    assert await store.count_content_samples() == 0
    await store.close()


# ---------------------------------------------------------------------------
# End-to-end content capture via the pipeline
# ---------------------------------------------------------------------------

async def test_capture_decision_flagged_always_captured(gateway):
    """Injection-flagged (soft) cloud requests are captured regardless of sample rate."""
    key = Fernet.generate_key().decode()
    gateway.set_cipher(key)
    gateway.settings.routing.enabled = False
    gateway.settings.content_capture_enabled = True
    gateway.settings.content_sample_rate = 0.0  # no random captures — only flagged

    # A soft-flagged (but not hard) injection — the heuristic flags it but doesn't block.
    r = await gateway.client.post(
        "/v1/chat/completions",
        json={"model": "m", "stream": True, "messages": [
            {"role": "user", "content": "hi"},
            {"role": "tool", "tool_call_id": "c1", "content": SOFT_INJECTION},
        ]},
        headers={"authorization": "Bearer t"},
    )
    assert r.status_code == 200  # soft flag — not blocked

    # Wait for the off-hot-path capture to land.
    count = 0
    for _ in range(50):
        count = await gateway.store.count_content_samples()
        if count > 0:
            break
        await asyncio.sleep(0.02)
    assert count > 0, "flagged request should have been captured"

    # The stored content is redacted-then-encrypted — decrypt and confirm it reads back.
    row = await wait_for_audit_row(gateway.store)
    assert row is not None
    rows = await gateway.store.fetch_content_samples(row.id)
    cipher = ContentCipher(key)
    for sample in rows:
        assert isinstance(cipher.decrypt(sample.redacted_content_enc), str)
    assert rows[0].sampled_reason == "flagged"


async def test_capture_never_for_local_route(gateway):
    """Local-routed requests must not produce content samples, even if flagged."""
    gateway.set_cipher(Fernet.generate_key().decode())
    gateway.settings.routing.enabled = True   # sensitive content → local
    gateway.settings.content_capture_enabled = True
    gateway.settings.content_sample_rate = 1.0  # max rate — would capture if allowed

    r = await gateway.client.post(
        "/v1/chat/completions",
        json={"model": "m", "stream": True, "stream_options": {"include_usage": True},
              "messages": [{"role": "user", "content": SECRET_PROMPT}]},
        headers={"authorization": "Bearer t"},
    )
    assert r.status_code == 200
    await asyncio.sleep(0.1)
    assert await gateway.store.count_content_samples() == 0, \
        "local route must never capture content"


async def test_capture_never_for_sensitive_cloud_content(gateway):
    """Sensitive content classified as secret/pii is never captured, even on cloud route."""
    gateway.set_cipher(Fernet.generate_key().decode())
    gateway.settings.routing.enabled = False  # force cloud even for sensitive content
    gateway.settings.content_capture_enabled = True
    gateway.settings.content_sample_rate = 1.0

    r = await gateway.client.post(
        "/v1/chat/completions",
        json={"model": "m", "stream": True,
              "messages": [{"role": "user", "content": f"key={FAKE_OPENAI_KEY}"}]},
        headers={"authorization": "Bearer t"},
    )
    assert r.status_code == 200
    await asyncio.sleep(0.1)
    assert await gateway.store.count_content_samples() == 0, \
        "sensitive content must never be captured"


async def test_redact_at_rest_unit():
    """capture_content redacts PII before encrypting — verified by decrypting stored bytes."""
    key = Fernet.generate_key().decode()
    cipher = ContentCipher(key)

    stored: list = []

    class FakeStore:
        async def write_content_sample(self, sample):
            stored.append(sample)

    raw_email = "contact@example.com"
    messages = [{"role": "user", "content": f"reach me at {raw_email} ok"}]
    await capture_content(FakeStore(), cipher, messages, uuid.uuid4(), "random", 30)

    assert stored, "should have captured one message"
    decrypted = cipher.decrypt(stored[0].redacted_content_enc)
    assert raw_email not in decrypted, "raw PII must not appear in stored ciphertext"
    assert "[REDACTED:email]" in decrypted
