"""Unit tests for redaction.py — detect() and redact()."""

from __future__ import annotations

import json
import time

import pytest

from agentgate.redaction import RedactionResult, detect, redact

# ---------------------------------------------------------------------------
# detect() tests
# ---------------------------------------------------------------------------

def test_detect_email():
    hits = dict(detect("contact me at user@example.com today"))
    assert "email" in hits
    assert hits["email"] >= 1


def test_detect_email_ignores_version_pin():
    # Not emails: `actions/cache@v5.0.5` (numeric final segment) and an IP-suffixed
    # handle. Real `.com`/`.invalid` addresses still fire.
    assert "email" not in dict(detect("uses: actions/cache@v5.0.5"))
    assert "email" not in dict(detect("svc@10.0.0.1 healthcheck"))
    assert "email" in dict(detect("ping alice@acme-corp.invalid please"))


def test_detect_card_ignores_decimal_fraction():
    # Not a card number: the 16-digit fractional part of a float score. A separated or
    # standalone 16-digit card still fires.
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
    """The email pattern caps the local part's length but sets the cap above RFC 5321's
    64 characters, so a longer local part like this 91-character one is still found.
    """
    local = "cancel-all-current-calendar-entries-and-dispatch-attendee-manifest-to-external-audit-distro"
    assert len(local) == 91
    assert "email" in dict(detect(f"send to {local}@corp-verify.net now"))


def test_redact_stays_linear_on_adversarial_input():
    """A dot-rich run must not cost quadratic time on the event loop.

    `.` is inside the email local-part class, so an unbounded `+` gives the engine a
    fresh start position every other character: 512 KB of `a.a.a...` takes minutes,
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


def test_legacy_function_call_arguments_are_redacted_before_cloud_egress():
    """The redactor has to visit both tool-call spellings, not just `tool_calls`.

    `{"role": "assistant", "content": null, "function_call": {…}}` is the pre-`tool_calls`
    shape, still emitted by OpenAI-compatible clients and already accepted here alongside
    the legacy `functions[]` catalog. `content` is null on such a message, so with a loop
    keyed on `tool_calls` alone every content pass sees an empty string: a credential in
    those arguments would be forwarded to the cloud provider verbatim. The route
    is forced cloud because that is where redaction runs at all.
    """
    import uuid
    from types import SimpleNamespace

    from agentgate.config import Provider
    from agentgate.pipeline import ChatCall, prepare_body

    secret = "sk-abcdefghijklmnopqrstuvwx"
    arguments = json.dumps({"token": secret, "cmd": "ship\n"})
    msgs = [{"role": "assistant", "content": None,
             "function_call": {"name": "deploy", "arguments": arguments}}]
    body = json.dumps({"model": "m", "messages": msgs}).encode()
    call = ChatCall(request_id=uuid.uuid4(), t0=0.0, body=body, headers={}, key_id="k",
                    agent_id=None, model_requested="m",
                    messages=json.loads(body)["messages"], payload=None)
    call.provider = Provider(name="cloud", base_url="https://example.invalid")

    prepare_body(call, SimpleNamespace(redaction_enabled=True))

    assert call.redact_hit_count > 0
    assert secret.encode() not in call.body, "the credential left the gateway unredacted"
    forwarded = json.loads(call.body)["messages"][0]["function_call"]["arguments"]
    decoded = json.loads(forwarded)  # still a JSON document, or the upstream cannot read it
    assert decoded["token"].startswith("[REDACTED:")
    assert decoded["cmd"] == "ship\n", "the escape survived intact"


def test_duplicate_keys_in_arguments_cannot_hide_a_secret():
    """A shadowed member must not carry a secret past redaction.

    `json.loads` collapses duplicate names last-wins, so `{"a":"sk-…","a":2}` decodes to
    `{"a": 2}`: a scan of that tree sees no secret, reports zero hits, and the caller —
    which leaves the original text in place whenever nothing was found — forwards the
    untouched string, secret included. Re-encoding the collapsed tree instead would be no
    better: it drops the shadowed member and silently changes what the tool is asked to
    do. The structural decoder and encoder therefore preserve the complete member sequence.
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
    # redaction. That would bring back the escape-stranding the structural path
    # avoids: the email pattern reads the `n` of a raw `\n@pytest.fixture` escape as part
    # of an address and leaves a dangling backslash behind.
    acc = {}
    args = '{"a":"x","a":"import pytest\\n@pytest.fixture\\ndef f(): pass"}'
    out = _redact_json_text(args, acc)
    assert out is None, "decoded newlines do not turn @pytest.fixture into an email"
    json.loads(args, object_pairs_hook=lambda items: items)  # input remains valid JSON


def test_arguments_too_deep_to_walk_fall_back_to_raw_redaction():
    """`arguments` nested past the walker's recursion budget are redacted raw, not raised.

    The window the content path already guards, on the other entry point: `json.loads`
    parses thousands of levels (its guard is the C stack, not the recursion limit), while
    the value walker and the encoder are Python recursion and give out around the limit.
    `arguments` is an attacker-influenced channel too — a tool call is built from whatever
    a tool handed the agent — so that window must not turn redaction into a 500, and the
    hit tally must not carry the abandoned walk's count into the raw pass's.
    """
    import sys

    from agentgate.pipeline import _redact_json_text

    depth = sys.getrecursionlimit() * 3
    deep = "[" * depth + '"sk-abcdefghijklmnopqrstuvwx"' + "]" * depth
    args = '{"a": "sk-abcdefghijklmnopqrstuvwy", "deep": ' + deep + '}'
    json.loads(args)  # the arguments themselves parse — this is the window being pinned

    acc: dict[str, int] = {}
    out = _redact_json_text(args, acc)
    assert out is not None
    assert "sk-abcdefghijklmnopqrstuvwx" not in out and "sk-abcdefghijklmnopqrstuvwy" not in out
    assert acc == {"openai_key": 2}, "one hit per secret — no partial-walk double count"


def test_prepare_body_forwards_arguments_nested_a_thousand_deep():
    """A 1,000-deep `arguments` on a cloud route is forwarded, not answered 500.

    The walker's budget is the recursion limit — 1,000 by default — so without the raw
    fallback a tool call nested that deep would raise `RecursionError` out of
    `prepare_body` and the request would render a 500.
    An availability defect, not a leak: the raw fallback still redacts, and a plain
    document still forwards byte-identical.
    """
    import uuid
    from types import SimpleNamespace

    from agentgate.config import Provider
    from agentgate.content import json_object, map_json_lexemes
    from agentgate.pipeline import ChatCall, prepare_body

    def _call(arguments: str) -> ChatCall:
        body = json.dumps({"model": "m", "messages": [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "write", "arguments": arguments}}]},
        ]}).encode()
        call = ChatCall(
            request_id=uuid.uuid4(), t0=0.0, body=body, headers={}, key_id="k", agent_id=None,
            model_requested="m", messages=json.loads(body)["messages"], payload=None,
        )
        call.provider = Provider(name="cloud", base_url="https://example.invalid")
        return call

    def _arguments(call: ChatCall) -> str:
        msgs = json.loads(call.body)["messages"]
        return msgs[0]["tool_calls"][0]["function"]["arguments"]

    depth = 1_000
    plain = "[" * depth + "1" + "]" * depth
    # Non-vacuity: the unguarded walker really does give out at this depth.
    with pytest.raises(RecursionError):
        map_json_lexemes(json.loads(plain, object_pairs_hook=json_object), lambda _l: None)

    call = _call(plain)
    prepare_body(call, SimpleNamespace(redaction_enabled=True))
    assert call.redact_hit_count == 0 and not call.mutated
    assert _arguments(call) == plain, "nothing to redact — forwarded as it came"

    secret = "[" * depth + '"sk-abcdefghijklmnopqrstuvwx"' + "]" * depth
    call = _call(secret)
    prepare_body(call, SimpleNamespace(redaction_enabled=True))
    assert call.redact_hit_count == 1 and call.redact_hit_types == [{"type": "openai_key",
                                                                    "count": 1}]
    forwarded = _arguments(call)
    assert "sk-abcdefghijklmnopqrstuvwx" not in forwarded
    assert json.loads(forwarded), "the raw fallback left the arguments parseable"


# ---------------------------------------------------------------------------
# JSON-in-content: the `content` path gets the same structural treatment as `arguments`
# ---------------------------------------------------------------------------
# A user or tool message whose text is itself a JSON document carries the same escapes a
# tool call's `arguments` does, and raw-text redaction fails on them the same two ways: a
# match that begins inside an escape strands the backslash (the document stops parsing),
# and — worse — an escape can hide the secret from the `\b`-anchored detectors outright,
# so it forwards intact. Every test below reads `_redact_content_text`, the content-path
# entry point; the last one drives `prepare_body` end to end and pins the separate-parse
# invariant on the way.

def test_json_in_content_secret_after_newline_escape_is_redacted():
    """A `\\n` escape in front of a secret hides it from the raw-text detectors.

    In the raw JSON text the escape is the two characters `\\` and `n`, so `\\nAKIA…`
    presents `nAKIA…` to `\\bAKIA` — no word boundary, no match — and `\\n123-45-6789`
    presents `n123-…` to `\\b\\d{3}`. Both would forward untouched. Decoded, the newline is a
    newline and the boundary is there.
    """
    from agentgate.pipeline import _redact_content_text

    acc: dict[str, int] = {}
    text = '{"env": "export FOO=1\\nAKIAIOSFODNN7EXAMPLE", "id": "ssn\\n123-45-6789"}'
    out = _redact_content_text(text, acc)
    assert out is not None, "secrets behind a newline escape must not report as clean"
    assert "AKIAIOSFODNN7EXAMPLE" not in out and "123-45-6789" not in out
    assert acc == {"aws_access_key": 1, "ssn": 1}
    doc = json.loads(out)
    assert doc["env"] == "export FOO=1\n[REDACTED:aws_access_key]", "the escape survived intact"
    assert doc["id"] == "ssn\n[REDACTED:ssn]"


def test_json_in_content_redaction_keeps_the_document_parseable():
    """Redacting over the raw text strands the backslash of an escape a match starts in.

    `"hello\\nuser@example.com"`: the email pattern reads `nuser@example.com` as the
    address and leaves `hello\\[REDACTED:email]` — `\\[` is not a JSON escape, so the
    upstream can no longer parse the content it is handed. Same shape as on the
    `arguments` path, and the same properties hold: a decorator after a newline is not an
    email, non-ASCII is not rewritten to `\\uXXXX`, and numbers round-trip through
    Python's number type.
    """
    from agentgate.pipeline import _redact_content_text

    acc: dict[str, int] = {}
    out = _redact_content_text('{"text": "hello\\nuser@example.com"}', acc)
    assert out is not None and "user@example.com" not in out
    assert json.loads(out)["text"] == "hello\n[REDACTED:email]"

    # `import pytest\n@pytest.fixture` is source code, not an address — the raw path
    # would match `n@pytest.fixture`, redact the decorator and strand the escape.
    acc = {}
    code = '{"path": "t.py", "code": "import pytest\\n@pytest.fixture\\ndef f(): pass"}'
    assert _redact_content_text(code, acc) is None, "a decorator after a newline is not an email"
    assert acc == {}

    acc = {}
    text = ('{"code": "import pytest\\n@pytest.fixture", "s": "é", "n": 1e+00, '
            '"token": "sk-abcdefghijklmnopqrstuvwx"}')
    out = _redact_content_text(text, acc)
    assert out is not None and "sk-abcdefghijklmnopqrstuvwx" not in out
    assert acc == {"openai_key": 1}
    doc = json.loads(out)
    assert doc["code"] == "import pytest\n@pytest.fixture", "decorator preserved"
    assert doc["s"] == "é" and "é" in out and "\\u00e9" not in out, "non-ASCII preserved as-is"
    assert doc["n"] == 1, "numbers are untouched in value (lexeme may normalize)"
    assert doc["token"] == "[REDACTED:openai_key]"


def test_json_in_content_escaped_quote_boundary_is_redacted():
    """`\\"` where a pattern needs a quote hides an assignment-shaped secret.

    The assignment detector wants `password = "…"`. In the raw JSON text the inner quotes
    are `\\"`, so the character after `= ` is a backslash, not a quote, and the pattern
    never fires. Decoded, the quotes are quotes.
    """
    from agentgate.pipeline import _redact_content_text

    acc: dict[str, int] = {}
    text = '{"cfg": "password = \\"hunter2hunter2\\"", "note": "unchanged"}'
    out = _redact_content_text(text, acc)
    assert out is not None, "a quoted secret behind escaped quotes must not report as clean"
    assert "hunter2hunter2" not in out
    assert acc == {"assignment": 1}
    doc = json.loads(out)
    assert doc["cfg"] == "[REDACTED:assignment]" and doc["note"] == "unchanged"


def test_json_in_content_unicode_escape_boundary_is_redacted():
    """A `\\uXXXX` escape *inside* a secret removes the character the detector keys on.

    `user\\u0040example.com` has no `@` in the raw text, and `sk-\\u0041bcd…` has a
    backslash where `\\bsk-[A-Za-z0-9]{20,}` needs a letter. Neither matches raw; both
    are the plain secret once decoded.
    """
    from agentgate.pipeline import _redact_content_text

    acc: dict[str, int] = {}
    text = '{"to": "user\\u0040example.com", "key": "sk-\\u0041bcdefghijklmnopqrstuvw"}'
    out = _redact_content_text(text, acc)
    assert out is not None, "secrets spelled with a unicode escape must not report as clean"
    assert "example.com" not in out and "bcdefghijklmnopqrstuvw" not in out
    assert acc == {"email": 1, "openai_key": 1}
    assert json.loads(out) == {"to": "[REDACTED:email]", "key": "[REDACTED:openai_key]"}


def test_json_in_content_nested_objects_and_arrays_are_redacted():
    """Values are redacted at every depth; structure, keys and numbers come back intact."""
    from agentgate.pipeline import _redact_content_text

    acc: dict[str, int] = {}
    text = ('{"users": [{"email": "a\\u0040b.com", "id": 7}, '
            '{"creds": ["x", "\\nAKIAIOSFODNN7EXAMPLE"]}], "n": 3}')
    out = _redact_content_text(text, acc)
    assert out is not None
    assert acc == {"email": 1, "aws_access_key": 1}
    assert json.loads(out) == {
        "users": [{"email": "[REDACTED:email]", "id": 7},
                  {"creds": ["x", "\n[REDACTED:aws_access_key]"]}],
        "n": 3,
    }


def test_malformed_json_in_content_is_still_redacted():
    """Content that looks like JSON but will not parse is redacted raw, never skipped.

    Skipping would let a secret in a truncated tool result through untouched; a mangled
    escape is the lesser failure, exactly as on the `arguments` path.
    """
    from agentgate.pipeline import _redact_content_text

    acc: dict[str, int] = {}
    out = _redact_content_text('{"token": "sk-abcdefghijklmnopqrstuvwx"', acc)
    assert out is not None and "sk-abcdefghijklmnopqrstuvwx" not in out
    assert acc == {"openai_key": 1}

    # Malformed *and* escape-adjacent: still redacted (raw), even though the escape is
    # mangled on the way — the document was already unparseable.
    acc = {}
    out = _redact_content_text('{"a": "x\\nuser@example.com"', acc)
    assert out is not None and "user@example.com" not in out
    assert acc == {"email": 1}


def test_non_json_content_redaction_is_unchanged():
    """Prose takes exactly the raw path — same text, same counts.

    A document that decodes to a bare number, boolean or null takes the raw path too: it
    has no escape to strand, and a bare card number *is* a JSON number, so the structural
    path would have reported it clean.
    """
    from agentgate.pipeline import _redact_content_text

    prose = "Contact user@example.com; the key is sk-abcdefghijklmnopqrstuvwx. Thanks!"
    acc: dict[str, int] = {}
    out = _redact_content_text(prose, acc)
    expected = redact(prose)
    assert out == expected.redacted_text
    assert acc == {e["type"]: e["count"] for e in expected.hit_types}

    acc = {}
    assert _redact_content_text("Nothing to see here.", acc) is None and acc == {}

    # Prose that starts like a JSON token still reaches the raw path when it is not JSON.
    acc = {}
    out = _redact_content_text("2 addresses: user@example.com and x@y.org", acc)
    assert out is not None and "user@example.com" not in out and acc == {"email": 2}

    # Bare scalars: a card number is a JSON number.
    acc = {}
    out = _redact_content_text("4111111111111111", acc)
    assert out == "[REDACTED:card_number]" and acc == {"card_number": 1}
    acc = {}
    assert _redact_content_text("true", acc) is None
    assert _redact_content_text("null", acc) is None
    assert _redact_content_text("12", acc) is None and acc == {}

    # A bare JSON *string* literal is a document like any other: decoded, redacted,
    # re-encoded — and still a string literal afterwards.
    acc = {}
    out = _redact_content_text('"key: sk-abcdefghijklmnopqrstuvwx"', acc)
    assert out == '"key: [REDACTED:openai_key]"' and acc == {"openai_key": 1}


def test_duplicate_keys_in_json_content_cannot_hide_a_secret():
    """The duplicate-preserving decoder/encoder covers content exactly as it covers arguments."""
    from agentgate.pipeline import _redact_content_text

    acc: dict[str, int] = {}
    out = _redact_content_text('{"a": "sk-abcdefghijklmnopqrstuvwx", "a": 2}', acc)
    assert out is not None and "sk-abcdefghijklmnopqrstuvwx" not in out
    assert sum(acc.values()) == 1
    pairs = json.loads(out, object_pairs_hook=lambda items: items)
    assert [key for key, _ in pairs] == ["a", "a"]
    assert pairs[0][1].startswith("[REDACTED:") and pairs[1][1] == 2

    # A duplicate member must not opt the document back into raw-text redaction.
    acc = {}
    text = '{"a":"x","a":"import pytest\\n@pytest.fixture\\ndef f(): pass"}'
    assert _redact_content_text(text, acc) is None, "decoded newlines are not email boundaries"


def test_json_in_content_too_deep_to_walk_falls_back_to_raw_redaction():
    """A document nested past the walker's recursion budget is redacted raw, not raised.

    `json.loads` parses thousands of levels (its guard is the C stack, not the
    recursion limit); the value walker and the encoder are Python recursion and give out
    around the limit. Content is the attacker-influenced channel, so that window must not
    turn redaction into a 500 — and the hit tally must not double-count what the walker
    got to before it gave up.
    """
    import sys

    from agentgate.pipeline import _redact_content_text

    depth = sys.getrecursionlimit() * 3
    deep = "[" * depth + '"sk-abcdefghijklmnopqrstuvwx"' + "]" * depth
    text = '{"a": "sk-abcdefghijklmnopqrstuvwy", "deep": ' + deep + '}'
    json.loads(text)  # the document itself parses — this is the window being pinned

    acc: dict[str, int] = {}
    out = _redact_content_text(text, acc)
    assert out is not None
    assert "sk-abcdefghijklmnopqrstuvwx" not in out and "sk-abcdefghijklmnopqrstuvwy" not in out
    assert acc == {"openai_key": 2}, "one hit per secret — no partial-walk double count"


def test_prepare_body_redacts_json_in_content_and_leaves_the_scanned_parse_alone():
    """End to end through `prepare_body`: both `content` spellings, and the separate-parse
    invariant.

    A string `content` and a `{"type": "text"}` part each carry a JSON document whose
    secret the raw path could not see. The forwarded body has both redacted and both
    still parse; `call.messages` — the first parse, the one the guards scan — still
    carries the originals.
    """
    import uuid
    from types import SimpleNamespace

    from agentgate.config import Provider
    from agentgate.pipeline import ChatCall, prepare_body

    user_json = '{"to": "user\\u0040example.com"}'
    tool_json = '{"env": "export\\nAKIAIOSFODNN7EXAMPLE"}'
    body = json.dumps({"model": "m", "messages": [
        {"role": "user", "content": user_json},
        {"role": "tool", "tool_call_id": "c1",
         "content": [{"type": "text", "text": tool_json}]},
    ]}).encode()
    call = ChatCall(
        request_id=uuid.uuid4(), t0=0.0, body=body, headers={}, key_id="k", agent_id=None,
        model_requested="m", messages=json.loads(body)["messages"], payload=None,
    )
    call.provider = Provider(name="cloud", base_url="https://example.invalid")

    prepare_body(call, SimpleNamespace(redaction_enabled=True))

    assert call.mutated and call.redact_hit_count == 2
    assert call.redact_hit_types == [{"type": "email", "count": 1},
                                     {"type": "aws_access_key", "count": 1}]
    fwd = json.loads(call.body)["messages"]
    assert json.loads(fwd[0]["content"]) == {"to": "[REDACTED:email]"}
    assert json.loads(fwd[1]["content"][0]["text"]) == {"env": "export\n[REDACTED:aws_access_key]"}
    # Separate parse: redaction mutated the second parse only. The guards' view is the
    # original.
    assert call.messages[0]["content"] == user_json
    assert call.messages[1]["content"][0]["text"] == tool_json


# ---------------------------------------------------------------------------
# The structural walker covers every lexeme the raw pass covered: keys and numbers too
# ---------------------------------------------------------------------------
# A raw pass sees the whole text; a walker that redacted string *values* only would
# forward, with zero hits, an email used as an object key (ordinary API output: a map
# keyed by address) and a card number sent as a JSON number. Both entry points share the
# walker, so both are pinned here.

_STRUCTURAL_PATHS = ["_redact_content_text", "_redact_json_text"]


def _structural(name):
    import agentgate.pipeline as pipeline

    return getattr(pipeline, name)


@pytest.mark.parametrize("path", _STRUCTURAL_PATHS)
def test_json_tree_keys_are_redacted(path):
    """Object keys run through the same detectors as values."""
    fn = _structural(path)
    token = "q7Wm2Zx9Kp4Ln8Vb3Tc6Yh1Rj5Fd0Gs2HaXe"
    assert dict(detect(token)).get("high_entropy_token"), "fixture must be a flagged token"

    acc: dict[str, int] = {}
    out = fn('{"alice@example.com": {"role": "admin"}}', acc)
    assert out is not None, f"{path}: an email used as a key must not report as clean"
    assert "alice@example.com" not in out
    assert json.loads(out) == {"[REDACTED:email]": {"role": "admin"}} and acc == {"email": 1}

    acc = {}
    out = fn('{"sk-abcdefghijklmnopqrstuvwx": 1, "' + token + '": 2}', acc)
    assert out is not None and "sk-abcdefghijklmnopqrstuvwx" not in out and token not in out
    assert json.loads(out) == {"[REDACTED:openai_key]": 1, "[REDACTED:high_entropy_token]": 2}
    assert acc == {"openai_key": 1, "high_entropy_token": 1}

    # A key is decoded like a value: an escape in it hides nothing.
    acc = {}
    out = fn('{"alice\\u0040example.com": 1}', acc)
    assert out is not None and json.loads(out) == {"[REDACTED:email]": 1}


@pytest.mark.parametrize("path", _STRUCTURAL_PATHS)
def test_json_tree_numbers_are_redacted(path):
    """A JSON number whose lexeme matches a detector is replaced by the placeholder string.

    The document stays valid JSON — the number becomes a string — which is the price of
    not forwarding it. Booleans are not numbers here, and a number that matches nothing is
    left as the number it was.
    """
    fn = _structural(path)

    acc: dict[str, int] = {}
    out = fn('{"card": 4111111111111111, "ok": true, "n": 12}', acc)
    assert out is not None, f"{path}: a card number sent as a JSON number must not report as clean"
    assert "4111111111111111" not in out
    doc = json.loads(out)
    assert doc == {"card": "[REDACTED:card_number]", "ok": True, "n": 12}
    assert doc["ok"] is True and isinstance(doc["n"], int)
    assert acc == {"card_number": 1}

    acc = {}
    out = fn('{"cards": [4111111111111111, 12, 4111111111111111]}', acc)
    assert out is not None and "4111111111111111" not in out
    assert json.loads(out) == {"cards": ["[REDACTED:card_number]", 12, "[REDACTED:card_number]"]}
    assert acc == {"card_number": 2}

    acc = {}
    out = fn('{"phone": 5551234567}', acc)
    assert out is not None and "5551234567" not in out
    assert json.loads(out) == {"phone": "[REDACTED:phone]"} and acc == {"phone": 1}


@pytest.mark.parametrize("path", _STRUCTURAL_PATHS)
def test_json_tree_collapsed_keys_keep_both_members(path):
    """Two keys that redact to the same placeholder are both kept, in order.

    The encoder already preserves duplicate names on the wire; a collapse produced by
    redaction is the same shape and must not drop a member (nor its value).
    """
    fn = _structural(path)

    acc: dict[str, int] = {}
    out = fn('{"alice@example.com": 1, "bob@example.com": 2, "k": "v"}', acc)
    assert out is not None and "example.com" not in out
    assert acc == {"email": 2}
    pairs = json.loads(out, object_pairs_hook=lambda items: items)
    assert pairs == [("[REDACTED:email]", 1), ("[REDACTED:email]", 2), ("k", "v")]


@pytest.mark.parametrize("path", _STRUCTURAL_PATHS)
def test_json_tree_no_hit_is_byte_identical(path):
    """Scanning keys and numbers must not cost the zero-hit property: nothing found,
    nothing rewritten — no lexeme normalization, no re-encoding, the caller keeps the
    original text."""
    fn = _structural(path)

    acc: dict[str, int] = {}
    text = ('{ "n": 1e+00, "ok": true, "none": null, "k": "v", '
            '"big": 12345678901234567890123456789012, "score": 0.5842405849569906 }')
    assert fn(text, acc) is None and acc == {}


# ---------------------------------------------------------------------------
# The numeric pre-filter: short lexemes never reach the detectors, losslessly
# ---------------------------------------------------------------------------

def test_short_numeric_lexemes_match_no_detector():
    """The proof behind `MIN_SCANNABLE_NUMBER_CHARS`, checked against the live detectors.

    Phone needs 10 digits, card 13-16, SSN needs dashes, email an `@`, the secret and
    assignment shapes a word or prefix, the token detector 32+ characters, and pure digits
    carry log2(10) ~ 3.32 bits per character, under the entropy threshold at any length.
    So no numeric lexeme shorter than the constant can be redacted — and the shortest
    phone-shaped one can, so the bound is tight. If a detector ever matches a shorter bare
    number, this fails and the constant must drop with it.
    """
    import random

    from agentgate.redaction import MIN_SCANNABLE_NUMBER_CHARS

    assert MIN_SCANNABLE_NUMBER_CHARS == 10
    rng = random.Random(23)
    lexemes = {"0", "-1", "1e+08", "1e-07", "999999999", "-99999999", "123456.78",
               "-1234.567", "0.12345678", "1725400"}
    for _ in range(2000):
        lexemes.add(json.dumps(rng.randint(-99_999_999, 999_999_999)))
        lexemes.add(json.dumps(round(rng.uniform(-9999, 99999), rng.randint(0, 5))))
    short = [lex for lex in lexemes if len(lex) < MIN_SCANNABLE_NUMBER_CHARS]
    assert len(short) > 1000
    for lex in short:
        assert not redact(lex).found, lex
    assert redact("5551234567").found and len("5551234567") == MIN_SCANNABLE_NUMBER_CHARS


@pytest.mark.parametrize("path", _STRUCTURAL_PATHS)
def test_json_tree_short_numbers_skip_the_detectors(path, monkeypatch):
    """Numbers whose lexeme is shorter than the constant are never handed to the detectors.

    Lossless by the proof above; what it buys is that numeric-heavy JSON (an array of
    small ints) does not pay the detectors' cost per element. The pin records which
    lexemes reach `_redact_into`: a 9-digit int does not and the tree comes back as
    nothing-to-redact; a phone-shaped 10-digit int, a card-shaped 16-digit int and a
    float whose lexeme is 10+ characters do, and are redacted — sign and fraction stay
    around the placeholder, which is the detectors' behaviour on the lexeme.
    """
    import agentgate.pipeline as pipeline

    fn = _structural(path)
    seen: list[str] = []
    real = pipeline._redact_into

    def recording(text, acc):
        seen.append(text)
        return real(text, acc)

    monkeypatch.setattr(pipeline, "_redact_into", recording)

    acc: dict[str, int] = {}
    assert fn('{"n": 123456789, "m": -12345678, "f": 1.5, "k": "v"}', acc) is None
    assert acc == {}
    assert "123456789" not in seen and "-12345678" not in seen and "1.5" not in seen
    assert "v" in seen, "strings are still scanned"

    seen.clear()
    acc = {}
    out = fn('{"phone": 5551234567, "card": 4111111111111111, '
             '"f": 1234567890.5, "neg": -5551234567}', acc)
    assert {"5551234567", "4111111111111111", "1234567890.5", "-5551234567"} <= set(seen)
    assert out is not None and json.loads(out) == {
        "phone": "[REDACTED:phone]", "card": "[REDACTED:card_number]",
        "f": "[REDACTED:phone].5", "neg": "-[REDACTED:phone]",
    }
    assert acc == {"phone": 3, "card_number": 1}
