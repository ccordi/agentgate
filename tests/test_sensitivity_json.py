"""The classifier decodes JSON-shaped content before it classifies.

Read as raw text, a secret behind a JSON escape (`{"env": "export\\nAKIA…"}`, the
`\\n` being the two characters backslash and n) would classify `none` and route to cloud
while the plain spelling classifies `secret` and stays local. Structural redaction
closes the same escape gap; these pins mirror its cases for the classifier — on the
first parse, read-only.
"""

from __future__ import annotations

import json
import sys

import pytest

from agentgate.sensitivity import Sensitivity, classify, classify_request

AWS = "AKIAIOSFODNN7EXAMPLE"
OPENAI = "sk-abcdefghijklmnopqrstuvwx"


def _user(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


def _tool(text: str) -> list[dict]:
    return [{"role": "tool", "tool_call_id": "c1", "content": text}]


def _call(*arguments: str) -> list[dict]:
    """An assistant tool-call turn whose only text is its arguments."""
    return [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": f"c{i}", "type": "function", "function": {"name": "f", "arguments": a}}
            for i, a in enumerate(arguments)
        ]},
    ]


# ---------------------------------------------------------------------------
# False negatives of a raw-text read, each for its stated reason
# ---------------------------------------------------------------------------

def test_json_in_content_secret_after_newline_escape_classifies_secret():
    """`\\n` in the raw text is backslash + n, so the `\\b`-anchored AWS pattern reads
    `nAKIA…` and misses. Decoded, the boundary is there."""
    hidden = _tool('{"env": "export\\n' + AWS + '"}')
    plain = _tool('{"env": "export ' + AWS + '"}')
    assert classify_request(plain).sensitivity is Sensitivity.SECRET
    r = classify_request(hidden)
    assert r.sensitivity is Sensitivity.SECRET, "an escape-hidden AWS key must not route to cloud"
    assert r.hit_types == ["aws_access_key"]

    # The same escape in front of an SSN: PII, and the type is named.
    r = classify_request(_user('{"id": "ssn\\n123-45-6789"}'))
    assert r.sensitivity is Sensitivity.PII and r.hit_types == ["ssn"]


def test_json_in_content_unicode_escape_inside_a_secret_classifies():
    """`\\uXXXX` inside the secret removes the character the detector keys on: `user\\u0040`
    has no `@`, `sk-\\u0041bcd…` has a backslash where the key pattern needs a letter."""
    r = classify_request(_user('{"to": "user\\u0040example.com"}'))
    assert r.sensitivity is Sensitivity.PII and r.hit_types == ["email"]
    r = classify_request(_user('{"key": "sk-\\u0041bcdefghijklmnopqrstuvw"}'))
    assert r.sensitivity is Sensitivity.SECRET and r.hit_types == ["openai_key"]


def test_json_in_content_escaped_quote_boundary_classifies_secret():
    """The assignment detector wants `password = "…"`; in the raw text the inner quotes
    are `\\"`, so the character after `= ` is a backslash and the pattern never fires."""
    r = classify_request(_user('{"cfg": "password = \\"hunter2hunter2\\"", "note": "ok"}'))
    assert r.sensitivity is Sensitivity.SECRET and r.hit_types == ["assignment"]


def test_json_in_content_nested_objects_and_arrays_classify():
    """Lexemes at every depth are read; the verdict is the most sensitive one found."""
    text = ('{"users": [{"email": "a\\u0040b.com", "id": 7}, '
            '{"creds": ["x", "\\n' + AWS + '"]}], "n": 3}')
    r = classify_request(_user(text))
    assert r.sensitivity is Sensitivity.SECRET and r.hit_types == ["aws_access_key"]
    # Without the secret the same document is PII on the escaped address alone.
    r = classify_request(_user('{"users": [{"email": "a\\u0040b.com", "id": 7}]}'))
    assert r.sensitivity is Sensitivity.PII and r.hit_types == ["email"]


def test_json_in_content_secret_in_an_object_key_classifies():
    """A key is read like a value — a tool result keyed by address or token is ordinary
    API output, and an escape in the key hides nothing."""
    r = classify_request(_user('{"alice\\u0040example.com": {"role": "admin"}}'))
    assert r.sensitivity is Sensitivity.PII and r.hit_types == ["email"]
    r = classify_request(_user('{"sk-\\u0041bcdefghijklmnopqrstuvw": 1}'))
    assert r.sensitivity is Sensitivity.SECRET and r.hit_types == ["openai_key"]


def test_duplicate_keys_in_json_content_cannot_hide_a_secret():
    """Every member is read, in either order: a plain `json.loads` keeps only the last
    value of a duplicated name, which would let `{"a": "<secret>", "a": 2}` classify clean."""
    r = classify_request(_user('{"a": "\\n' + AWS + '", "a": 2}'))
    assert r.sensitivity is Sensitivity.SECRET, "the first of two duplicate members was dropped"
    r = classify_request(_user('{"a": 2, "a": "\\n' + AWS + '"}'))
    assert r.sensitivity is Sensitivity.SECRET


def test_json_in_text_part_list_is_decoded():
    """Both `content` spellings: the `{"type": "text"}` part list is read through
    `coerce_content` exactly like the plain string."""
    msgs = [{"role": "tool", "tool_call_id": "c1",
             "content": [{"type": "text", "text": '{"env": "export\\n' + AWS + '"}'}]}]
    assert classify_request(msgs).sensitivity is Sensitivity.SECRET


def test_tool_call_arguments_are_decoded_before_classification():
    """`arguments` is a JSON document too, and read raw it has the identical false
    negative. Each call's arguments are decoded on their own: the joined arguments of two
    calls are not one document, and decoding the join would fall back to raw for exactly
    the multi-call message."""
    assert classify_request(_call('{"env": "export ' + AWS + '"}')).sensitivity \
        is Sensitivity.SECRET
    r = classify_request(_call('{"env": "export\\n' + AWS + '"}'))
    assert r.sensitivity is Sensitivity.SECRET, "an escape-hidden key in arguments routed to cloud"
    r = classify_request(_call('{"path": "a.txt"}', '{"env": "export\\n' + AWS + '"}'))
    assert r.sensitivity is Sensitivity.SECRET, "the second call's arguments were not decoded"
    r = classify_request(_call('{"to": "user\\u0040example.com"}'))
    assert r.sensitivity is Sensitivity.PII and r.hit_types == ["email"]


def test_private_repo_marker_behind_an_escape_classifies_private_repo():
    """A marker miss has no redaction backstop: proprietary content goes to cloud intact.

    The marker check is a substring test, not `\\b`-anchored, so an escape *before* the
    marker does not hide it — but an escape *inside* it does: `\\u0041CME-…` and a
    PHP-style `\\/`-escaped repository path both miss on the raw text.
    """
    markers = ["ACME-CONFIDENTIAL", "github.com/acme/private-repo"]
    for text in ('{"note": "\\u0041CME-CONFIDENTIAL"}',
                 '{"url": "https:\\/\\/github.com\\/acme\\/private-repo"}'):
        r = classify_request(_user(text), markers)
        assert r.sensitivity is Sensitivity.PRIVATE_REPO, text
        assert r.hit_types == ["marker"]
    # The plain and escape-before spellings classify on the raw text too.
    for text in ('{"note": "from the ACME-CONFIDENTIAL repo"}', '{"note": "see\\nACME-CONFIDENTIAL"}'):
        assert classify_request(_user(text), markers).sensitivity is Sensitivity.PRIVATE_REPO


# ---------------------------------------------------------------------------
# Preservation and robustness — the same on raw and decoded reads, by design
# ---------------------------------------------------------------------------

def test_malformed_json_in_content_still_classifies():
    """Text that looks like JSON but will not parse is classified raw, never skipped."""
    r = classify_request(_user('{"token": "' + OPENAI + '"'))
    assert r.sensitivity is Sensitivity.SECRET and r.hit_types == ["openai_key"]
    r = classify_request(_user('{"a": "x", "to": user@example.com}'))
    assert r.sensitivity is Sensitivity.PII


def test_non_json_content_classification_is_unchanged():
    """Prose takes exactly the raw path: the same verdict and hit types as `classify`.

    A document that decodes to a bare number, boolean or null takes the raw path too — a
    bare card number is a JSON number and must still classify PII.
    """
    prose = "Contact user@example.com; the key is " + OPENAI + ". Thanks!"
    r = classify_request(_user(prose))
    direct = classify(prose)
    assert (r.sensitivity, r.hit_types) == (direct.sensitivity, direct.hit_types)
    assert r.sensitivity is Sensitivity.SECRET

    assert classify_request(_user("Nothing to see here.")).sensitivity is Sensitivity.NONE
    assert classify_request(_user("2 addresses: user@example.com and x@y.org")).sensitivity \
        is Sensitivity.PII
    assert classify_request(_user("4111111111111111")).sensitivity is Sensitivity.PII
    for scalar in ("true", "null", "12", ""):
        assert classify_request(_user(scalar)).sensitivity is Sensitivity.NONE, scalar

    # A bare JSON string literal is a document like any other.
    assert classify_request(_user('"key: ' + OPENAI + '"')).sensitivity is Sensitivity.SECRET


def test_json_numbers_classify_as_their_lexeme():
    """A card or phone number sent as a JSON number is read as its lexeme (PII, as the
    raw pass sees it); short numbers never reach the detectors and stay clean."""
    r = classify_request(_user('{"card": 4111111111111111, "ok": true}'))
    assert r.sensitivity is Sensitivity.PII and r.hit_types == ["card_number"]
    r = classify_request(_user('{"cards": [12, 4111111111111111]}'))
    assert r.sensitivity is Sensitivity.PII and r.hit_types == ["card_number"]
    assert classify_request(_user('{"n": 123456789, "m": -12345678, "f": 1.5, "t": true}')) \
        .sensitivity is Sensitivity.NONE


def test_lexeme_boundaries_never_form_a_match():
    """The decoded surface is the document's lexemes; a boundary between two of them is a
    hard boundary, as it is for the per-lexeme redactor, so joining cannot manufacture a
    match the raw text did not have. Three short numeric strings are not a phone number,
    and a key named `password` next to a value spelled `= "…"` is not an assignment."""
    assert classify_request(_user('["555", "123", "4567"]')).sensitivity is Sensitivity.NONE
    assert classify_request(_user('{"a": "555", "b": "123", "c": "4567"}')).sensitivity \
        is Sensitivity.NONE
    assert classify_request(_user('{"password": "= \\"hunter2hunter2\\""}')).sensitivity \
        is Sensitivity.NONE
    # Non-vacuity: the same characters as one lexeme do match.
    assert classify_request(_user('["555 123 4567"]')).sensitivity is Sensitivity.PII


def test_json_in_content_too_deep_to_walk_falls_back_to_raw():
    """A document nested past the walker's recursion budget classifies raw, never raises.

    `json.loads` parses thousands of levels (its guard is the C stack); the lexeme walker
    is Python recursion and gives out near the limit. Content is the attacker-influenced
    channel, so that window must not turn classification into a 500 — and the raw path
    still sees the plain secret.
    """
    from agentgate.content import json_object, map_json_lexemes

    depth = sys.getrecursionlimit() * 3
    deep = "[" * depth + '"' + OPENAI + '"' + "]" * depth
    text = '{"a": "' + OPENAI + '", "deep": ' + deep + '}'
    tree = json.loads(text, object_pairs_hook=json_object)  # the document parses — this is the window
    with pytest.raises(RecursionError):  # non-vacuity: the unguarded walker does give out
        map_json_lexemes(tree, lambda _lexeme: None)

    r = classify_request(_user(text))
    assert r.sensitivity is Sensitivity.SECRET and r.hit_types == ["openai_key"]


def test_classify_window_applies_to_the_decoded_surface():
    """Decode per message, then join and window.

    The window counts characters of the surface the classifier reads — lexemes and
    their separators — not wire bytes, so `truncated_at` keeps its meaning (the verdict
    covers the first n characters read) and a document that only fits once decoded is
    read whole. Windowing the raw text first would cut a JSON document mid-way and hand
    the malformed remainder to the raw path, reopening the escape gap for exactly the
    tool results big enough to be truncated.
    """
    # Surface of {"k": "x"*N, "s": AWS}: "k" + sep + x*N + sep + "s" + sep + AWS = N + 25.
    fits = _tool('{"k": "' + "x" * 75 + '", "s": "' + AWS + '"}')
    assert len(fits[0]["content"]) > 100, "the raw text is over the window; the surface is not"
    r = classify_request(fits, (), max_chars=100)
    assert r.sensitivity is Sensitivity.SECRET and r.truncated_at is None

    over = _tool('{"k": "' + "x" * 80 + '", "s": "' + AWS + '"}')
    r = classify_request(over, (), max_chars=100)
    assert r.sensitivity is Sensitivity.NONE and r.truncated_at == 100, \
        "a secret past the decoded window is still missed — and marked"

    head = _tool('{"s": "' + AWS + '", "k": "' + "x" * 200 + '"}')
    r = classify_request(head, (), max_chars=100)
    assert r.sensitivity is Sensitivity.SECRET and r.truncated_at == 100


def test_classifier_and_redactor_enumerate_the_same_lexemes(monkeypatch):
    """One walker, two consumers: the lexemes the classifier reads are exactly the ones
    the structural redactor hands to the detectors, in order."""
    import agentgate.pipeline as pipeline
    from agentgate.content import LEXEME_SEPARATOR, lexeme_surface

    text = ('{"alice\\u0040example.com": {"n": 1, "big": 4111111111111111, "f": 1.5, '
            '"list": ["x", "\\n' + AWS + '", 5551234567, true, null]}, "z": "é", "z": ""}')
    seen: list[str] = []
    real = pipeline._redact_into

    def recording(lexeme, acc):
        seen.append(lexeme)
        return real(lexeme, acc)

    monkeypatch.setattr(pipeline, "_redact_into", recording)
    assert pipeline._redact_content_text(text, {}) is not None

    surface = lexeme_surface(text)
    assert surface is not None
    assert surface.split(LEXEME_SEPARATOR) == seen
    assert seen == ["alice@example.com", "n", "big", "4111111111111111", "f", "list", "x",
                    "\n" + AWS, "5551234567", "z", "é", "z", ""]


def test_classify_and_route_reads_the_decoded_surface_and_leaves_both_parses_alone():
    """End to end: `classify_and_route` routes the escape-hidden secret local, and the
    classification is read-only — `call.messages` (the guards' parse) and the body are
    byte-identical before and after, and `prepare_body`'s second parse is untouched by it."""
    import uuid

    from agentgate.config import Settings
    from agentgate.pipeline import ChatCall, classify_and_route, prepare_body

    tool_json = '{"env": "export\\n' + AWS + '"}'
    body = json.dumps({"model": "m", "messages": [
        {"role": "user", "content": '{"note": "\\u0041CME-CONFIDENTIAL"}'},
        {"role": "tool", "tool_call_id": "c1",
         "content": [{"type": "text", "text": tool_json}]},
    ]}).encode()
    messages = json.loads(body)["messages"]
    call = ChatCall(
        request_id=uuid.uuid4(), t0=0.0, body=body, headers={}, key_id="k", agent_id=None,
        model_requested="m", messages=messages, payload=None,
    )
    settings = Settings(private_repo_markers=["ACME-CONFIDENTIAL"])

    classify_and_route(call, settings)

    assert call.sensitivity_class == "secret"
    assert call.provider is not None and call.provider.is_local
    assert not call.caveats
    # Separate parse, read-only: the first parse still carries the originals, byte for byte.
    assert call.messages is messages and call.messages == json.loads(body)["messages"]
    assert call.body == body and call.payload is None and not call.mutated

    # Without the secret the marker alone keeps it local — the case with no backstop.
    marker_only = json.dumps({"model": "m", "messages": [messages[0]]}).encode()
    call2 = ChatCall(
        request_id=uuid.uuid4(), t0=0.0, body=marker_only, headers={}, key_id="k",
        agent_id=None, model_requested="m", messages=json.loads(marker_only)["messages"],
        payload=None,
    )
    classify_and_route(call2, settings)
    assert call2.sensitivity_class == "private_repo" and call2.provider.is_local

    # The second parse is `prepare_body`'s alone: on the local route nothing is redacted,
    # and the payload it builds is a fresh decode of the same bytes.
    prepare_body(call, settings)
    assert call.payload is not None and call.payload["messages"] == messages
    assert call.messages[1]["content"][0]["text"] == tool_json


def test_nothing_is_decoded_past_the_window(monkeypatch):
    """The bound on the classifier's work is still the window, not the transcript.

    Pinned by counting: a message after the window is spent is never handed to the
    decoder, and the document that straddles the edge is walked only as far as the edge.
    """
    import agentgate.content as content
    import agentgate.sensitivity as sensitivity_mod

    decoded: list[int] = []  # length of each text handed to the decoder
    visited: list[int] = []  # lexemes the walker handed over, per walk
    real_surface = content.lexeme_surface
    real_map = content.map_json_lexemes

    def counting_surface(text, limit=None):
        decoded.append(len(text))
        return real_surface(text, limit)

    depth = 0

    def counting_map(node, visit):
        nonlocal depth
        if depth:  # the walker recursing through the patched name: the same walk
            return real_map(node, visit)
        n = 0

        def counted(lexeme):
            nonlocal n
            n += 1
            return visit(lexeme)

        depth += 1
        try:
            return real_map(node, counted)
        finally:
            depth -= 1
            visited.append(n)

    doc = json.dumps({"items": [{"id": i, "name": f"item-{i}"} for i in range(200)]})
    full = real_surface(doc)  # before the patches: this walk must not be counted
    assert full is not None
    total = len(full.split(content.LEXEME_SEPARATOR))
    assert total == 601  # "items" + 3 per item: two keys and the name (ids are short numbers)
    msgs = [{"role": "tool", "tool_call_id": "c", "content": doc} for _ in range(10)]

    monkeypatch.setattr(sensitivity_mod, "lexeme_surface", counting_surface)
    monkeypatch.setattr(content, "map_json_lexemes", counting_map)

    # A window the first document overflows: one decode, a walk cut off near the edge.
    r = classify_request(msgs, (), max_chars=100)
    assert r.truncated_at == 100 and r.sensitivity is Sensitivity.NONE
    assert decoded == [len(doc)] and len(visited) == 1 and visited[0] < 30

    # A window the first document fills with 50 characters to spare: the second is
    # walked to the edge, the other eight are never decoded.
    decoded.clear()
    visited.clear()
    r = classify_request(msgs, (), max_chars=len(full) + 50)
    assert r.truncated_at == len(full) + 50
    assert decoded == [len(doc), len(doc)]
    assert visited[0] == total and visited[1] < 30

    # A window the whole transcript fits: every document decoded and walked whole.
    decoded.clear()
    visited.clear()
    r = classify_request(msgs, (), max_chars=10 * (len(full) + 1))
    assert r.truncated_at is None and decoded == [len(doc)] * 10 and visited == [total] * 10

    # Arguments get the same treatment through the same window reader.
    decoded.clear()
    call = {"role": "assistant", "content": "x" * 78, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "f",
                                                      "arguments": '{"t": "' + AWS + '"}'}}]}
    # Surface "t" + sep + AWS = 22 characters: exactly the remainder of a 100 window
    # after 78 characters of content — read whole, classified, not truncated.
    r = classify_request([call], (), max_chars=100)
    assert r.sensitivity is Sensitivity.SECRET and r.truncated_at is None
    # One character less of budget cuts the key — the bound still bounds, and says so.
    r = classify_request([call], (), max_chars=99)
    assert r.sensitivity is Sensitivity.NONE and r.truncated_at == 99


def test_json_too_deep_for_the_parser_itself_falls_back_to_raw(monkeypatch):
    """`json.loads` gives out too, not only the walker — and content is attacker-supplied.

    Catching only the walker's RecursionError would leave the *parser's* uncaught, so a
    200 KB body of nested arrays would raise out of `classify_request` and the endpoint
    would answer 500. Where the C scanner gives out is not a fixed depth: Python 3.14
    checks the thread's own stack, so 100,000 levels raise under an 8 MB stack and parse
    under 16 MB (the CI runner's). The real document is therefore checked
    for its answer, whichever budget it exhausts, and the parser's failure is forced
    explicitly.
    """
    from agentgate import content

    deep = "[" * 100_000 + "1" + "]" * 100_000
    assert classify_request([{"role": "user", "content": deep}]).sensitivity is Sensitivity.NONE

    def parser_gives_out(*_args, **_kwargs):
        raise RecursionError("maximum recursion depth exceeded while decoding a JSON array")

    monkeypatch.setattr(content.json, "loads", parser_gives_out)
    assert classify_request([{"role": "user", "content": "[1]"}]).sensitivity is Sensitivity.NONE
    # and the raw path still sees a plain secret in such a document
    text = '{"a": "' + OPENAI + '", "deep": ' + deep + "}"
    result = classify_request([{"role": "user", "content": text}])
    assert result.sensitivity is Sensitivity.SECRET and result.hit_types == ["openai_key"]


def test_raw_spelling_matches_disappear_once_decoded():
    """Where an escape's spelling fills a detector's run, a raw-text read matches an artifact.

    `\\u0040` inside a base64-looking run makes a 32+ character high-entropy token of the
    raw text; `\\u0041` inside a card-shaped key makes a 16-digit run. Decoded, neither is
    anything the model reads, and the structural redactor finds nothing in them either.
    """
    token = '"QUJDREVGR0hJSktMT\\u0040U5PUFFSU1RVVldYWVphYmNkZWY5OTk5"'
    card = '{"411111111111111\\u00411":0}'
    for text in (token, card):
        assert classify_request([{"role": "user", "content": text}]).sensitivity is Sensitivity.NONE
    # non-vacuity: the raw spellings do match, which is what a raw-text read classifies on
    assert classify(token).sensitivity is Sensitivity.SECRET
    assert classify(card).sensitivity is Sensitivity.PII
