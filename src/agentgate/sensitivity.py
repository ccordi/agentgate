"""Sensitivity classifier — `{none, pii, secret, private_repo}`.

Single source of truth for **both** routing (sensitive content stays local, zero cloud
egress) and audit-retention policy.  Regex/entropy detection via `redaction.detect()`;
no LLM, no egress, runs inline on the hot path.

Precedence (most-sensitive wins): secret > private_repo > pii > none.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import StrEnum

from agentgate.content import coerce_content, lexeme_surface, tool_call_arguments
from agentgate.redaction import detect


class Sensitivity(StrEnum):
    NONE = "none"
    PII = "pii"
    SECRET = "secret"
    PRIVATE_REPO = "private_repo"


# Secret hit_type names produced by redaction.detect() that map to SECRET sensitivity.
_SECRET_TYPES = frozenset({
    "private_key", "aws_access_key", "openai_key", "github_token",
    "slack_token", "google_api_key", "assignment", "high_entropy_token",
})


@dataclass
class ClassifyResult:
    sensitivity: Sensitivity
    hit_types: list[str] = field(default_factory=list)
    # The window size at which `classify_request` stopped reading, or None when it read
    # everything. Set, the verdict describes the first `truncated_at` characters only.
    truncated_at: int | None = None

    @property
    def is_sensitive(self) -> bool:
        return self.sensitivity is not Sensitivity.NONE


def classify(text: str, markers: Iterable[str] = ()) -> ClassifyResult:
    """Classify a single text blob. ``markers`` are configured private-repo indicators."""
    if not text:
        return ClassifyResult(Sensitivity.NONE)

    detected = detect(text)  # [(hit_type, count)]
    hit_type_names = [t for t, _ in detected]

    secret_hits = [t for t in hit_type_names if t in _SECRET_TYPES]
    if secret_hits:
        return ClassifyResult(Sensitivity.SECRET, secret_hits)

    repo_hits = [m for m in markers if m and m in text]
    if repo_hits:
        return ClassifyResult(Sensitivity.PRIVATE_REPO, ["marker"])

    pii_hits = [t for t in hit_type_names if t not in _SECRET_TYPES]
    if pii_hits:
        return ClassifyResult(Sensitivity.PII, pii_hits)

    return ClassifyResult(Sensitivity.NONE)


_SENSITIVITY_RANK = {
    Sensitivity.NONE: 0,
    Sensitivity.PII: 1,
    Sensitivity.PRIVATE_REPO: 2,
    Sensitivity.SECRET: 3,
}


def _merge_results(left: ClassifyResult, right: ClassifyResult) -> ClassifyResult:
    """Return the more-sensitive result, combining same-tier evidence without duplicates."""
    left_rank = _SENSITIVITY_RANK[left.sensitivity]
    right_rank = _SENSITIVITY_RANK[right.sensitivity]
    if left_rank > right_rank:
        return left
    if right_rank > left_rank:
        return right
    return ClassifyResult(left.sensitivity, list(dict.fromkeys(left.hit_types + right.hit_types)))


def _surface(text: str, limit: int) -> str:
    """What the detectors read for one message's content, or one tool call's arguments.

    A JSON document is read as its lexemes — string values, keys, numbers — through the
    enumeration the cloud-egress redactor uses (`content.lexeme_surface`); anything else
    is read as it is. On the raw text a secret behind a JSON escape is invisible to the
    `\\b`-anchored detectors, so `{"env": "export\\nAKIA…"}` would classify `none` and
    route to cloud while the plain spelling stays local. Decoding per piece,
    before the window is applied, is what keeps a truncated document from turning into
    malformed JSON that would take the raw path again. ``limit`` is how much of the
    surface the caller can still read: past it the walk stops, and the raw text is
    returned whole for the caller to cut.
    """
    if not text:
        return text
    decoded = lexeme_surface(text, limit)
    return text if decoded is None else decoded


def _arguments_surface(message: dict, limit: int) -> str:
    """The message's tool-call arguments, each decoded on its own, joined by newlines."""
    return "\n".join(_surface(arguments, limit) for arguments in tool_call_arguments(message))


def _read_window[T](items: Iterable[T], window: int, piece: Callable[[T, int], str],
                    *, skip_empty: bool = False) -> tuple[str, bool]:
    """``"\\n".join(pieces)[:window]`` and whether any of it fell outside the window.

    The same text and the same verdict as joining every piece and slicing — pinned by
    the window tests — but no piece is decoded once the window is spent, and the piece
    that straddles the edge is walked only as far as the edge: ``piece`` is handed the
    room left plus one, enough to learn that it overflows. Without this the classifier's
    work would grow with every JSON document in the transcript instead of staying bounded
    by the window.

    ``skip_empty`` reproduces the arguments join, which drops a message that has no
    arguments before joining; the content join counts every message, empty or not.
    """
    parts: list[str] = []
    room = window
    for item in items:
        text = piece(item, room + 1)
        if skip_empty and not text:
            continue
        if parts:  # a separator before every piece but the first
            if room == 0:
                return "".join(parts), True
            parts.append("\n")
            room -= 1
        if len(text) > room:
            parts.append(text[:room])
            return "".join(parts), True
        parts.append(text)
        room -= len(text)
    return "".join(parts), False


def classify_request(messages: list[dict], markers: Iterable[str] = (), *, max_chars: int = 20000) -> ClassifyResult:
    """Classify the content that would egress (all message text, bounded).

    Every message's content and every tool call's arguments are read through
    `_surface` first, so a JSON document is classified over its decoded lexemes. The
    window is then applied to what was read: ``max_chars`` counts characters of the
    classified surface, not wire bytes, so ``truncated_at`` keeps its meaning — the
    verdict covers the first n characters the classifier read — and a document that
    only fits the window once decoded is read whole. Nothing is decoded past the window
    (`_read_window`): the bound on the classifier's work is still the window, not the
    transcript.

    Tool-call arguments count as message text: they egress with the request like
    anything else, and on an assistant tool-call message they are the only text there.

    They are appended *after* all message content rather than interleaved with it, and
    only into what the window has left. Interleaved, one ordinary 20 KB `write_file` call
    would push every later message past `max_chars`, so a secret or a private-repo marker
    in those messages would go unseen. Content keeps the whole window; arguments get the
    remainder. A marker matters most here: redaction is not a backstop for it, so a miss
    routes proprietary content to cloud.

    ``truncated_at`` is set on the result when any of the input fell outside the window:
    content past ``max_chars``, arguments past the remainder, or arguments present when
    content had already spent the window. The first two are reported by `_read_window`
    at the character where the window overflowed; the last is a presence check on the
    two tool-call spellings rather than a coercion, so the arguments are never assembled
    only to be discarded.
    """
    marker_tuple = tuple(markers)
    content, truncated = _read_window(
        (coerce_content(m.get("content")) for m in messages), max_chars, _surface)
    result = classify(content, marker_tuple)
    remaining = max_chars - len(content)
    if remaining > 0:
        args, args_cut = _read_window(messages, remaining, _arguments_surface, skip_empty=True)
        if args:
            truncated = truncated or args_cut
            # Classify the two surfaces separately, then merge by the documented
            # precedence. A synthetic separator cannot consume one character of the
            # payload budget, the regex input never exceeds `max_chars` in total, and no
            # detector can manufacture a match by straddling the content/arguments edge.
            result = _merge_results(result, classify(args, marker_tuple))
    elif any(isinstance(m, dict) and (m.get("tool_calls") or m.get("function_call"))
             for m in messages):
        truncated = True  # window spent by content; the arguments went unread
    if truncated:
        result.truncated_at = max_chars
    return result
