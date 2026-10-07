"""Local-route request adaptation — a client-compatibility adapter, not a security control.

Two unrelated jobs, both scoped to the local (`is_local`) upstream. The overrides apply
only when set; the cleaning always runs:

1. **Env-driven overrides** (`AGENTGATE_LOCAL_*`) so the local server's model, stop list,
   token cap and reasoning flag can be changed with a restart, without a code change.
2. **System-prompt cleaning.** Some agent frameworks staple a Gemini-style
   `<think>…</think>` + `<final>…</final>` wrapping instruction into the system prompt.
   That conflicts with a local model's own chat template, so the block is swapped for a
   short equivalent.

Nothing here inspects untrusted content or makes a trust decision. It rewrites the
operator's own outbound request for a specific server's benefit — if it stopped working
the result is a badly formatted reply, not a missed attack.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterator

log = logging.getLogger("agentgate")

# Matches the Gemini-specific thinking/final wrapping instructions such frameworks
# inject. Lazy matching, so small edits across prompt-template versions still match.
_SYSTEM_PROMPT_CLEAN_RE = re.compile(
    r"ALL internal reasoning MUST be inside <think>.*?</think>\..*?"
    r"Format every reply as <think>.*?</think> then <final>.*?</final>.*?"
    r"Example: <think>.*?</think>\s*<final>.*?</final>",
    re.DOTALL
)

_SYSTEM_PROMPT_REPLACEMENT = (
    "For final user-visible answers, wrap them in <final>...</final>. "
    "For tool calls, use the native tool calling schema."
)


def _iter_system_text_slots(
    messages: list[dict],
) -> Iterator[tuple[str, Callable[[str], None]]]:
    """Yield (text, setter) for each system/developer text slot — plain-string
    content and each text part of list content."""
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") not in ("system", "developer"):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            def set_content(new: str, msg=msg) -> None:
                msg["content"] = new
            yield content, set_content
        elif isinstance(content, list):
            for part in content:
                if (
                    isinstance(part, dict)
                    and part.get("type") == "text"
                    and isinstance(part.get("text"), str)
                ):
                    def set_text(new: str, part=part) -> None:
                        part["text"] = new
                    yield part["text"], set_text


def _clean_system_prompts(payload: dict) -> bool:
    """Strip the client's <think>/<final> wrapping rules from every system text slot."""
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return False
    mutated = False
    for text, set_text in _iter_system_text_slots(messages):
        has_instr = (
            "ALL internal reasoning MUST be inside" in text
            or ("<think>" in text and "<final>" in text)
        )
        if not has_instr:
            continue
        new_text, count = _SYSTEM_PROMPT_CLEAN_RE.subn(_SYSTEM_PROMPT_REPLACEMENT, text)
        if count > 0:
            set_text(new_text)
            mutated = True
        else:
            log.warning(
                "Detected <think>/<final> prompt instructions, "
                "but regex adapter failed to match."
            )
    return mutated


def adapt(payload: dict, settings) -> bool:
    """Apply the local-route overrides and prompt cleaning in place. Returns "mutated?".

    Called only when the resolved provider `is_local`, and after the provider's own model
    rewrite — so `local_model_override` wins.
    """
    mutated = False
    if settings.local_model_override:
        payload["model"] = settings.local_model_override
        mutated = True
    if settings.local_stop:
        payload["stop"] = [s for s in settings.local_stop.split(",") if s]
        mutated = True
    if settings.local_max_tokens is not None:
        payload["max_completion_tokens"] = settings.local_max_tokens
        payload["max_tokens"] = settings.local_max_tokens
        mutated = True
    if settings.local_enable_thinking is not None:
        ctk = payload.get("chat_template_kwargs")
        ctk = ctk if isinstance(ctk, dict) else {}
        ctk["enable_thinking"] = settings.local_enable_thinking
        payload["chat_template_kwargs"] = ctk
        mutated = True
    return _clean_system_prompts(payload) or mutated
