"""Content-tier tests — crypto, capture decisions, sweeper."""

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

def _store(url) -> AuditStore:
    return AuditStore(url)


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

async def test_write_and_count_content_sample(audit_db_url):
    store = _store(audit_db_url)
    await store.init()
    await store.write_content_sample(_sample())
    assert await store.count_content_samples() == 1
    await store.close()


async def test_purge_deletes_only_expired(audit_db_url):
    store = _store(audit_db_url)
    await store.init()
    now = datetime.now(UTC)
    # One expired, one future
    await store.write_content_sample(_sample(expires_at=now - timedelta(seconds=1)))
    await store.write_content_sample(_sample(expires_at=now + timedelta(days=30)))
    deleted = await store.purge_expired_content()
    assert deleted == 1
    assert await store.count_content_samples() == 1
    await store.close()


async def test_purge_returns_zero_when_nothing_expired(audit_db_url):
    store = _store(audit_db_url)
    await store.init()
    await store.write_content_sample(_sample())
    assert await store.purge_expired_content() == 0
    await store.close()


async def test_purge_deletes_all_expired(audit_db_url):
    store = _store(audit_db_url)
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


# ---------------------------------------------------------------------------
# Candidate selection — the tier samples what the guards scanned
# ---------------------------------------------------------------------------
#
# The guards read `content.trailing_tool_outputs` — the newest run of `tool`/`function`
# messages the assistant has not answered yet — and `capture_content` samples the same
# messages, plus the newest user turn. A narrower answer to "which messages are untrusted",
# such as the single last `role=="tool"` message anywhere in the history, diverges in four
# cases, pinned here: a batched turn, the legacy `function` role, a batch displaced by a
# later message, stale history.

async def _capture(messages: list[dict]) -> tuple[ContentCipher, list[ContentSampleAudit]]:
    """Run `capture_content` over `messages`, recording the rows it writes."""
    cipher = ContentCipher(Fernet.generate_key().decode())
    stored: list[ContentSampleAudit] = []

    class RecordingStore:
        async def write_content_sample(self, sample):
            stored.append(sample)

    await capture_content(RecordingStore(), cipher, messages, uuid.uuid4(), "flagged", 30)
    return cipher, stored


def _batched_turn(tool_role: str) -> list[dict]:
    """`assistant(tool_calls=[a,b,c])` answered by three tool-output messages."""
    return [
        {"role": "system", "content": "you are a helpful assistant"},
        {"role": "user", "content": "summarize these pages"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "fetch", "arguments": "{}"}},
            {"id": "b", "type": "function", "function": {"name": "fetch", "arguments": "{}"}},
            {"id": "c", "type": "function", "function": {"name": "fetch", "arguments": "{}"}},
        ]},
        {"role": tool_role, "tool_call_id": "a", "name": "fetch", "content": "page A body"},
        {"role": tool_role, "tool_call_id": "b", "name": "fetch", "content": "page B body"},
        {"role": tool_role, "tool_call_id": "c", "name": "fetch", "content": "page C body"},
    ]


async def test_capture_samples_whole_parallel_tool_batch():
    """A batched turn stores every tool result, not just the last (plus the user turn)."""
    cipher, stored = await _capture(_batched_turn("tool"))

    assert [s.role for s in stored] == ["tool", "tool", "tool", "user"]
    assert [cipher.decrypt(s.redacted_content_enc) for s in stored] == [
        "page A body", "page B body", "page C body", "summarize these pages",
    ]


async def test_capture_samples_legacy_function_role():
    """The legacy `function` role is tool output too — sampled, and stored as itself."""
    cipher, stored = await _capture(_batched_turn("function"))

    assert [s.role for s in stored] == ["function", "function", "function", "user"]
    assert cipher.decrypt(stored[0].redacted_content_enc) == "page A body"


async def test_capture_follows_a_displaced_tool_batch():
    """A message the agent application adds after the tool results does not move them out
    of the scanned surface, so the tier stores them — plus the newest user turn, which is
    that message."""
    messages = [*_batched_turn("tool"), {"role": "user", "content": "continue"}]
    cipher, stored = await _capture(messages)

    assert [s.role for s in stored] == ["tool", "tool", "tool", "user"]
    assert [cipher.decrypt(s.redacted_content_enc) for s in stored] == [
        "page A body", "page B body", "page C body", "continue",
    ]


async def test_capture_does_not_resample_stale_tool_history():
    """A tool message from an earlier turn was scanned on the request it arrived on; it is
    not re-sampled (and mis-attributed to this request's flag) on every later one."""
    messages = [
        {"role": "user", "content": "what is in the changelog"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "read", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "a", "content": "stale changelog body"},
        {"role": "assistant", "content": "it lists three fixes"},
        {"role": "user", "content": "and what about the release date"},
    ]
    cipher, stored = await _capture(messages)

    assert [s.role for s in stored] == ["user"]
    assert cipher.decrypt(stored[0].redacted_content_enc) == "and what about the release date"
