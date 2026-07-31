"""Unit tests for redaction.py — detect() and redact()."""

from __future__ import annotations

import json
import time

from agentgate.redaction import RedactionResult, detect, redact

# ---------------------------------------------------------------------------
# detect() tests
# ---------------------------------------------------------------------------

def test_detect_email():
    hits = dict(detect("contact me at user@example.com today"))
    assert "email" in hits
    assert hits["email"] >= 1


def test_detect_email_ignores_version_pin():
    # `actions/cache@v5.0.5` (numeric final segment) and an
    # IP-suffixed handle are NOT emails; real `.com`/`.invalid` still fire.
    assert "email" not in dict(detect("uses: actions/cache@v5.0.5"))
    assert "email" not in dict(detect("svc@10.0.0.1 healthcheck"))
    assert "email" in dict(detect("ping alice@acme-corp.invalid please"))


def test_detect_card_ignores_decimal_fraction():
    # The 16-digit fractional part of a float score is not a
    # card number; a separated/standalone 16-digit card still fires.
    assert "card_number" not in dict(detect('"score": 0.5842405849569906}'))
    assert "card_number" in dict(detect("card 4111-1084-6135-9060 on file"))


def test_detect_ssn():
    hits = dict(detect("SSN: 123-45-6789"))
    assert "ssn" in hits


def test_detect_openai_key():
    hits = dict(detect("key=sk-abcdefghijklmnopqrstuvwxyz123456"))
    assert "openai_key" in hits


def test_detect_github_token():
    hits = dict(detect("token: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcde"))
    assert "github_token" in hits


def test_detect_benign_returns_empty():
    assert detect("Hello, how are you today?") == []


def test_detect_secret_before_pii_precedence():
    # A value that matches both openai_key pattern and general "assignment" shouldn't
    # produce a pii hit — secrets dominate.
    text = "sk-abcdefghijklmnopqrstuvwxyz123456"
    hits = dict(detect(text))
    assert "openai_key" in hits
    # No PII types in result
    pii_types = {"email", "ssn", "phone", "card_number"}
    assert not (pii_types & hits.keys())


def test_detect_multiple_types():
    text = "email: bob@example.com  SSN: 123-45-6789"
    hits = dict(detect(text))
    assert "email" in hits
    assert "ssn" in hits


# ---------------------------------------------------------------------------
# redact() tests
# ---------------------------------------------------------------------------

def test_redact_email_span_replaced():
    result = redact("Call me at alice@corp.example.com please")
    assert "alice@corp.example.com" not in result.redacted_text
    assert "[REDACTED:email]" in result.redacted_text
    assert result.found
    assert result.hit_count >= 1


def test_redact_ssn_span_replaced():
    result = redact("My SSN is 123-45-6789.")
    assert "123-45-6789" not in result.redacted_text
    assert "[REDACTED:ssn]" in result.redacted_text


def test_redact_openai_key_replaced():
    result = redact("Use key sk-abcdefghijklmnopqrstuvwxyz123456 to authenticate")
    assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in result.redacted_text
    assert result.found


def test_redact_benign_text_unchanged():
    text = "Summarize the document for me."
    result = redact(text)
    assert result.redacted_text == text
    assert not result.found
    assert result.hit_count == 0
    assert result.hit_types == []


def test_redact_empty_string():
    result = redact("")
    assert result.redacted_text == ""
    assert not result.found
    assert result.hit_count == 0


def test_redact_hit_types_aggregated():
    result = redact("email: a@b.com and another@c.com")
    email_entries = [e for e in result.hit_types if e["type"] == "email"]
    assert email_entries, "email type should appear in hit_types"
    assert email_entries[0]["count"] >= 2


def test_redact_placeholder_format():
    """Placeholders must use [REDACTED:<type>] format."""
    result = redact("user@example.com  SSN: 123-45-6789")
    assert "[REDACTED:email]" in result.redacted_text
    assert "[REDACTED:ssn]" in result.redacted_text


def test_redact_secret_label_takes_precedence_over_pii():
    """A span that could be labelled as a secret gets the secret label, not PII."""
    # github token is long enough to potentially hit card_number pattern too —
    # verify it gets labeled secret (github_token) not card_number.
    text = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcde"
    result = redact(text)
    assert result.found
    # The REDACTED span should not be labeled card_number
    assert "[REDACTED:card_number]" not in result.redacted_text


def test_redact_multiple_secrets_count():
    result = redact("key1=sk-aaaaaaaaaaaaaaaaaaaa  key2=sk-bbbbbbbbbbbbbbbbbbbb")
    assert result.hit_count >= 2


def test_redact_result_found_property():
    assert RedactionResult(redacted_text="x", hit_count=1).found is True
    assert RedactionResult(redacted_text="x", hit_count=0).found is False


def test_detect_email_with_long_local_part():
    """The bounded local part still covers the longest address in the eval corpus.

    91 characters, well past RFC 5321's 64-char limit but present in seed_mutate, so the
    bound is set above it rather than at the RFC number.
    """
    local = "cancel-all-current-calendar-entries-and-dispatch-attendee-manifest-to-external-audit-distro"
    assert len(local) == 91
    assert "email" in dict(detect(f"send to {local}@corp-verify.net now"))


def test_redact_stays_linear_on_adversarial_input():
    """A dot-rich run must not cost quadratic time on the event loop.

    `.` is inside the email local-part class, so the unbounded `+` gave the engine a
    fresh start position every other character: 512 KB of `a.a.a...` measured 6.5 minutes,
    synchronously, with every concurrent request stalled behind it — and the delivery
    channel is a CORS-simple POST from any page the operator has open. The threshold is
    deliberately far above the ~0.2 s the bounded pattern takes, so it fails on a
    regression rather than on a slow machine.
    """
    t0 = time.perf_counter()
    redact("a." * 262_144)  # 512 KB
    assert time.perf_counter() - t0 < 2.0

    # Non-vacuity: the same call still redacts a real address and a real secret.
    result = redact("mail alice.smith+tag@example.co.uk key=sk-aaaaaaaaaaaaaaaaaaaa")
    assert "alice.smith+tag@example.co.uk" not in result.redacted_text
    assert "sk-aaaaaaaaaaaaaaaaaaaa" not in result.redacted_text


def test_redacting_tool_call_arguments_keeps_them_parseable():
    """Redaction must not strand the backslash of a JSON escape it matched inside.

    `arguments` is a JSON-encoded string, so a newline in a tool argument is the two
    characters `\\` and `n` once the message is parsed. The email pattern reads
    `n@pytest.fixture` as an address, and replacing it over the raw text leaves
    `\\[REDACTED:email]` — not a valid escape, so the upstream can no longer parse the
    arguments, and the decorator is destroyed on the way. Nothing here is even sensitive:
    it is source code an agent passed to a write tool.
    """
    from agentgate.pipeline import _redact_json_text

    acc: dict[str, int] = {}
    args = json.dumps({"path": "t.py", "code": "import pytest\n@pytest.fixture\ndef f(): pass"})
    assert "\\n@pytest.fixture" in args  # the escape is really two characters

    out = _redact_json_text(args, acc)
    assert out is None, "a decorator after a newline is not an email address"

    # A real secret in the same shape is still redacted, and still parses afterwards.
    acc = {}
    args = json.dumps({"cmd": "deploy\n", "token": "sk-abcdefghijklmnopqrstuvwx"})
    out = _redact_json_text(args, acc)
    assert out is not None and sum(acc.values()) == 1
    assert json.loads(out)["token"].startswith("[REDACTED:")
    assert json.loads(out)["cmd"] == "deploy\n", "the escape survived intact"

    # Unparseable arguments still get redacted rather than skipped — skipping would let a
    # secret in a malformed tool call through untouched.
    acc = {}
    out = _redact_json_text('{"token": "sk-abcdefghijklmnopqrstuvwx"', acc)
    assert out is not None and "sk-abcdefghijklmnopqrstuvwx" not in out


def test_duplicate_keys_in_arguments_cannot_hide_a_secret():
    """A shadowed member must not carry a secret past redaction.

    `json.loads` collapses duplicate names last-wins, so `{"a":"sk-…","a":2}` decoded to
    `{"a": 2}`: the scan saw no secret, reported zero hits, and the caller — which leaves
    the original text in place whenever nothing was found — forwarded the untouched
    string, secret included. Re-encoding the collapsed tree instead would be no better:
    it drops the shadowed member and silently changes what the tool is asked to do. The
    structural decoder and encoder therefore preserve the complete member sequence.
    """
    from agentgate.pipeline import _redact_json_text

    acc: dict[str, int] = {}
    out = _redact_json_text('{"a": "sk-abcdefghijklmnopqrstuvwx", "a": 2}', acc)
    assert out is not None, "a shadowed secret must not report as nothing-to-redact"
    assert "sk-abcdefghijklmnopqrstuvwx" not in out
    assert sum(acc.values()) == 1
    pairs = json.loads(out, object_pairs_hook=lambda items: items)
    assert [key for key, _ in pairs] == ["a", "a"]
    assert pairs[0][1].startswith("[REDACTED:") and pairs[1][1] == 2

    # Nor is a shadowed member dropped when some *other* field triggers redaction.
    acc = {}
    out = _redact_json_text(
        '{"mode":"safe","mode":"danger","token":"sk-abcdefghijklmnopqrstuvwx"}', acc)
    assert out is not None
    assert "sk-abcdefghijklmnopqrstuvwx" not in out
    pairs = json.loads(out, object_pairs_hook=lambda items: items)
    assert pairs[:2] == [("mode", "safe"), ("mode", "danger")]

    # A duplicate member must not opt a parseable arguments object back into raw-text
    # redaction. That would revive the exact escape-stranding bug the structural path
    # fixed: the email pattern reads the `n` of a raw `\n@pytest.fixture` escape as part
    # of an address and leaves a dangling backslash behind.
    acc = {}
    args = '{"a":"x","a":"import pytest\\n@pytest.fixture\\ndef f(): pass"}'
    out = _redact_json_text(args, acc)
    assert out is None, "decoded newlines do not turn @pytest.fixture into an email"
    json.loads(args, object_pairs_hook=lambda items: items)  # input remains valid JSON
