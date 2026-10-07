"""Message/content helpers shared by the guards, the sensitivity classifier, and the pipeline.

Three questions live here: *how do I read the text out of an OpenAI message?*, *which
messages in this request are untrusted?* and *which lexemes of a JSON document can a
detector match?* The third is shared by the structural redactor (`pipeline`) and the
sensitivity classifier (`sensitivity`), and lives here because `pipeline` imports
`sensitivity` — one enumeration, two consumers, no cycle. The module also holds the
bounded-gap probe the detectors share (`bounded_gap_abandoned`).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass

from agentgate.redaction import MIN_SCANNABLE_NUMBER_CHARS


def coerce_content(content) -> str:
    """OpenAI content is either a string or a list of typed parts; flatten to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and isinstance(p.get("text"), str):
                parts.append(p["text"])
        return "\n".join(parts)
    return ""


def coerce_tool_call_args(message) -> str:
    """Flatten an assistant message's tool-call arguments to text.

    `arguments` is a JSON string the model produced, and it is where a secret the agent
    passes to a tool actually lives — on such a message `content` is usually None, so
    anything reading only `content` sees an empty string. Assistant tool-call history
    replays on every subsequent turn, so a miss here repeats for the life of the session
    rather than happening once.
    """
    return "\n".join(tool_call_arguments(message))


def tool_call_functions(message) -> list[dict]:
    """Each ``function`` dict on an assistant message that carries an ``arguments``
    string, in order — both the modern and the legacy spelling.

    Modern is `tool_calls[*].function`; legacy is the bare `function_call` object, still
    emitted by OpenAI-compatible clients and SDKs and already an accepted input shape
    here (the request parser reads the matching legacy `functions[]` catalog). Reading
    only the modern spelling would leave a legacy message's arguments unread: a
    credential there would classify `none`, route to the default cloud provider, and
    egress unredacted, while the identical content spelled `tool_calls` classifies
    `secret` and stays local.

    The containing dict rather than the string, because the cloud-egress redactor has to
    write a redacted `arguments` back where it came from. One enumeration for the
    classifier and the redactor is what stops their idea of "the arguments in this
    request" from drifting apart.

    Skips any malformed shape rather than raising: this runs on attacker-influenced
    structure.
    """
    if not isinstance(message, dict):
        return []
    functions: list[dict] = []
    calls = message.get("tool_calls")
    if isinstance(calls, list):
        for call in calls:
            if not isinstance(call, dict):
                continue
            fn = call.get("function")
            if isinstance(fn, dict) and isinstance(fn.get("arguments"), str):
                functions.append(fn)
    legacy = message.get("function_call")
    if isinstance(legacy, dict) and isinstance(legacy.get("arguments"), str):
        functions.append(legacy)
    return functions


def tool_call_arguments(message) -> list[str]:
    """Each tool call's ``arguments`` string on an assistant message, in order.

    One entry per call rather than the joined text, because each is a JSON document of
    its own: a consumer that decodes them (`sensitivity.classify_request`) must decode
    one at a time — the join of two documents is not a document.
    """
    return [fn["arguments"] for fn in tool_call_functions(message)]


# --- the lexemes of a JSON document ----------------------------------------------------
#
# A user or tool message whose text is a JSON document — an API response the agent
# fetched, a config file, a tool call's `arguments` — carries escapes, and a detector that
# reads the raw characters is blind to a secret behind one: `"\nAKIA…"` presents `nAKIA…`
# to a `\b`-anchored pattern, `user\u0040example.com` has no `@`, `password = \"…\"` has
# no quote where the assignment shape needs one. Both consumers of a decoded document —
# the cloud-egress redactor and the sensitivity classifier — must read the same lexemes,
# or a secret the classifier misses routes to cloud and only the redactor's second look
# catches it. So the enumeration is defined once, here.


@dataclass
class JSONObject:
    """Decoded JSON object that preserves member order and duplicate names."""

    pairs: list[tuple[str, object]]


def json_object(pairs: list[tuple[str, object]]) -> JSONObject:
    """`object_pairs_hook`: keep the wire object's complete member sequence instead of
    collapsing duplicates — a plain `json.loads` keeps only the last value of a repeated
    name, which would let `{"a": "<secret>", "a": 2}` hide the secret from every reader."""
    return JSONObject(pairs)


def map_json_lexemes(node, visit: Callable[[str], str | None]):
    """Walk a decoded JSON tree, handing ``visit`` every lexeme a detector can match.

    String values, object keys and numbers — everything a raw-text pass over the document
    would have seen. A key is visited exactly like a value. A number is visited as its
    JSON lexeme (`json.dumps`), and only when the lexeme is at least
    `MIN_SCANNABLE_NUMBER_CHARS` long: no detector can match a shorter bare number (proof
    and pin on the constant), and numeric-heavy JSON is mostly such numbers, so the skip
    is what keeps an array of small ints from paying the detectors' cost per element.
    Booleans and null are not lexemes here.

    ``visit`` returns a replacement or None. The tree comes back rebuilt with the
    replacements applied — a replaced number becomes the returned *string*, so the
    document stays valid JSON at the price of a changed type — and, on no replacement,
    structurally identical to what went in. A consumer that only reads (the classifier)
    passes a visitor that always returns None.

    Python recursion: gives out with `RecursionError` near the recursion limit, well
    before `json.loads` does. Callers on attacker-influenced text catch it and fall back
    to the raw text.
    """
    if isinstance(node, str):
        replacement = visit(node)
        return node if replacement is None else replacement
    if isinstance(node, JSONObject):
        return JSONObject([(map_json_lexemes(key, visit), map_json_lexemes(value, visit))
                           for key, value in node.pairs])
    if isinstance(node, list):
        return [map_json_lexemes(value, visit) for value in node]
    if isinstance(node, (int, float)) and not isinstance(node, bool):
        lexeme = json.dumps(node)
        if len(lexeme) < MIN_SCANNABLE_NUMBER_CHARS:
            return node
        replacement = visit(lexeme)
        return node if replacement is None else replacement
    return node


# What separates two lexemes in `lexeme_surface`. The redactor scans lexemes one at a
# time, so a boundary between two of them is absolute; the classifier scans the joined
# surface in one pass, and the separator has to keep that boundary absolute too — no
# detector may match or cross it. NUL is in no detector's character class: it is not
# `\w`, not `\s` (the phone shape's `[-.\s]?` and the assignment shape's `\s*` cross a
# newline, so three short numeric strings joined by `\n` would read as a phone number
# and a key named `password` would pair with a value spelled `= "…"`), not `[ -]` (the
# card shape), and `\b` holds on either side of it. The one class that admits it,
# the assignment value's `[^'"]`, needs a quote on both sides of the crossing.
LEXEME_SEPARATOR = "\x00"


class _SurfaceFull(Exception):
    """Raised by the collecting visitor once `limit` characters of surface are in hand."""


def lexeme_surface(text: str, limit: int | None = None) -> str | None:
    """The text a detector should read for ``text`` when it is a JSON document, else None.

    A document that decodes to an object, array or string yields its lexemes, in
    document order, joined by `LEXEME_SEPARATOR`. Everything else answers None and the
    caller reads the raw text: prose; malformed JSON (a secret in a truncated tool
    result must still be read); a bare number, boolean or null (no escape to hide
    behind, and a bare card number is a JSON number); and a document nested past the
    walker's *or the parser's* recursion budget — `json.loads` parses thousands of
    levels before its C scanner raises a `RecursionError` of its own, the walker is
    Python recursion, and neither window may turn a request into a 500.

    ``limit`` bounds the work to what the caller will read: the walk stops as soon as
    the joined surface is at least ``limit`` characters long, and what comes back is
    then a prefix of the full surface, of at least that length. The parse itself is
    whole either way — a document has to parse before it has lexemes — but it is C
    speed, while the walk is Python and the larger cost on a big document.
    """
    try:
        data = json.loads(text, object_pairs_hook=json_object)
    except (ValueError, RecursionError):
        return None
    if not isinstance(data, (JSONObject, list, str)):
        return None
    lexemes: list[str] = []
    collect: Callable[[str], None] = lexemes.append
    if limit is not None:
        length = -1  # the join is one separator shorter than the lexemes plus separators

        def collect(lexeme: str) -> None:
            nonlocal length
            lexemes.append(lexeme)
            length += len(lexeme) + 1
            if length >= limit:
                raise _SurfaceFull

    try:
        map_json_lexemes(data, collect)
    except RecursionError:
        return None
    except _SurfaceFull:
        pass
    return LEXEME_SEPARATOR.join(lexemes)


# Roles carrying tool output. Both the modern OpenAI `tool` role and the legacy
# `function` role (still emitted by many OpenAI-compatible clients/SDKs) are
# attacker-influenced channels and MUST be scanned — otherwise an injection delivered
# as `{"role": "function", ...}` reaches the model on an unscanned channel.
TOOL_OUTPUT_ROLES = frozenset({"tool", "function"})


def trailing_tool_outputs(messages: list[dict]) -> list[dict]:
    """The newest *unanswered* run of tool-output messages — the current turn's tool
    results, which may be a *parallel/batched* set
    (`assistant(tool_calls=[a,b,c]) → tool(a), tool(b), tool(c)`).

    Scanning only the single last tool message misses every result but the last when the
    agent makes parallel tool calls. Scanning the whole run closes that.

    Walking back from the end, the rule is:

    * trailing messages that carry no tool output and are not the assistant's — a user
      nudge, a system or developer reminder, a compaction pass's leftovers, an unknown
      role — are SKIPPED. Position in the array is the agent application's business; the
      channel the content arrived on is the security-relevant fact, and it does not stop
      being untrusted because something was appended after it. Breaking on the first
      non-tool message would let any such append move the whole batch out of the scan
      surface — and for the LLM judge, whose surface is exactly this run, nothing
      would be scanned at all.
    * the contiguous run of `TOOL_OUTPUT_ROLES` messages is then collected, and
    * the first ASSISTANT message stops the walk. Reaching one before any tool output
      returns `[]`.

    That last clause is a deliberate limit: each batch is scanned on the request where
    it arrives, and not re-scanned as history on later turns — a cost and false-positive
    choice, not a claim that one scan is sufficient. An assistant message after a batch
    means the model already answered it, so the batch is history. Nothing correlates
    content across turns; the threat model lists this limit.
    """
    block: list[dict] = []
    for m in reversed(messages):
        role = m.get("role")
        if role in TOOL_OUTPUT_ROLES:
            block.append(m)
        elif block or role == "assistant":
            # The run has ended (anything non-tool bounds it), or the assistant already
            # answered whatever came before — either way, stop.
            break
    block.reverse()
    return block


def tool_output_texts(messages: list[dict]) -> list[tuple[str, str]]:
    """The newest unanswered tool-output batch as [(source, text)] — the narrow surface.

    Its own function because it is a surface in its own right: the LLM judge scans only
    this, while `extract_untrusted` below is this plus the newest user turn. Defining it
    once is what keeps "what counts as tool-output text" from having two answers.
    """
    return [("tool_output", coerce_content(m.get("content")))
            for m in trailing_tool_outputs(messages)]


def extract_untrusted(messages: list[dict]) -> list[tuple[str, str]]:
    """Return [(source, text)] for the untrusted content in a request.

    Untrusted = the current turn's tool-output messages (`tool`/`function` role, often
    attacker-influenced — the whole unanswered batch, wherever the agent application put
    it in the array, so parallel tool calls are covered) and the newest user message.
    Earlier turns were already scanned on prior requests.

    The LLM judge deliberately scans a *narrower* surface than this — only
    `tool_output_texts`, never the user turn. See `guards.scan`.
    """
    out = tool_output_texts(messages)
    last_user = next(
        (m for m in reversed(messages) if m.get("role") == "user"), None
    )
    if last_user is not None:
        out.append(("user", coerce_content(last_user.get("content"))))
    return out


# --- bounded-gap abandon probe ---------------------------------------------------------

# Ceiling on the distance a probe reports. The recorded integer is for seeing how far
# past a bound the padding went, not for measuring it exactly; a cap keeps the audit
# value small no matter how much text an attacker supplies.
GAP_DISTANCE_CAP = 10_000


def bounded_gap_abandoned(
    text: str, prefix: re.Pattern, suffix: re.Pattern, bound: int
) -> int | None:
    """Distance past ``bound`` between a pattern's prefix and its suffix, or None.

    For a ``prefix.{0,bound}suffix`` pattern that did NOT match — the bound that keeps
    these patterns from going quadratic is also their detection limit, and this makes a
    match abandoned for distance visible. Two linear searches: the first prefix hit,
    then the first suffix hit after it. A distance over the bound is the abandoned
    case and is returned (capped at ``GAP_DISTANCE_CAP``); within the bound — a
    newline between them, which ``.`` does not cross — or no suffix at all is not, and
    returns None. Callers run it only where the bounded pattern missed, and record the
    result without ever flagging on it: this is visibility, not detection.
    """
    p = prefix.search(text)
    if p is None:
        return None
    s = suffix.search(text, p.end())
    if s is None:
        return None
    distance = s.start() - p.end()
    if distance <= bound:
        return None
    return min(distance, GAP_DISTANCE_CAP)
