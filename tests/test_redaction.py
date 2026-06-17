"""Unit tests for security/redaction.py — detect() and redact()."""

from __future__ import annotations

from agentgate.security.redaction import RedactionResult, detect, redact

# ---------------------------------------------------------------------------
# detect() tests
# ---------------------------------------------------------------------------

def test_detect_email():
    hits = dict(detect("contact me at user@example.com today"))
    assert "email" in hits
    assert hits["email"] >= 1


def test_detect_email_ignores_version_pin():
    # 2b over-fire regression: `actions/cache@v5.0.5` (numeric final segment) and an
    # IP-suffixed handle are NOT emails; real `.com`/`.invalid` still fire.
    assert "email" not in dict(detect("uses: actions/cache@v5.0.5"))
    assert "email" not in dict(detect("svc@10.0.0.1 healthcheck"))
    assert "email" in dict(detect("ping alice@acme-corp.invalid please"))


def test_detect_card_ignores_decimal_fraction():
    # 2b over-fire regression: the 16-digit fractional part of a float score is not a
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
