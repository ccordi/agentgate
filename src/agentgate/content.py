"""Message/content helpers shared by the guards, the sensitivity classifier, and capture.

Two questions live here: *how do I read the text out of an OpenAI message?* and
*which messages in this request are untrusted?* Both answers are consumed by several
modules.
"""

from __future__ import annotations


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

    `tool_calls[*].function.arguments` is a JSON string the model produced, and it is
    where a secret the agent passes to a tool actually lives — on such a message
    `content` is usually None, so anything reading only `content` sees an empty string.
    Assistant tool-call history replays on every subsequent turn, so a miss here repeats
    for the life of the session rather than happening once.
    """
    if not isinstance(message, dict):
        return ""
    calls = message.get("tool_calls")
    if not isinstance(calls, list):
        return ""
    parts = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function")
        if isinstance(fn, dict) and isinstance(fn.get("arguments"), str):
            parts.append(fn["arguments"])
    return "\n".join(parts)

# Roles carrying tool output. Both the modern OpenAI `tool` role and the legacy
# `function` role (still emitted by many OpenAI-compatible clients/SDKs) are
# attacker-influenced channels and MUST be scanned — otherwise an injection delivered
# as `{"role": "function", ...}` reaches the model on an unscanned channel.
TOOL_OUTPUT_ROLES = frozenset({"tool", "function"})


def trailing_tool_outputs(messages: list[dict]) -> list[dict]:
    """The contiguous run of tool-output messages at the tail of the conversation —
    i.e. the current turn's tool results, which may be a *parallel/batched* set
    (`assistant(tool_calls=[a,b,c]) → tool(a), tool(b), tool(c)`).

    Scanning only the single last tool message misses every result but the last when a
    harness emits parallel tool calls. Scanning the
    whole trailing run closes that, and each batch is still scanned exactly once: on the
    next turn these results sit behind an assistant message, so the trailing run becomes
    the *new* results, not the accumulated history.
    """
    block: list[dict] = []
    for m in reversed(messages):
        if m.get("role") in TOOL_OUTPUT_ROLES:
            block.append(m)
        else:
            break
    block.reverse()
    return block


def tool_output_texts(messages: list[dict]) -> list[tuple[str, str]]:
    """The trailing tool-output batch as [(source, text)] — the narrow scan surface.

    Its own function because it is a surface in its own right: the LLM guard scans only
    this, while `extract_untrusted` below is this plus the newest user turn. Defining it
    once is what keeps "what counts as tool-output text" from having two answers.
    """
    return [("tool_output", coerce_content(m.get("content")))
            for m in trailing_tool_outputs(messages)]


def extract_untrusted(messages: list[dict]) -> list[tuple[str, str]]:
    """Return [(source, text)] for the untrusted content in a request.

    Untrusted = the current turn's tool-output messages (`tool`/`function` role, often
    attacker-influenced — the whole trailing batch, so parallel tool calls are covered)
    and the newest user message. Earlier turns were already scanned on prior requests.

    The LLM guard deliberately scans a *narrower* surface than this — only
    `tool_output_texts`, never the user turn. See `guards.scan`.
    """
    out = tool_output_texts(messages)
    last_user = next(
        (m for m in reversed(messages) if m.get("role") == "user"), None
    )
    if last_user is not None:
        out.append(("user", coerce_content(last_user.get("content"))))
    return out
