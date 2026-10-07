"""Unit tests for the local-route adapter.

Drives the adapter directly, including the warning branch (instructions detected,
regex didn't match).
"""

from __future__ import annotations

import logging

import pytest

from agentgate import local_route
from tests.support import make_settings

WRAPPING = (
    "ALL internal reasoning MUST be inside <think>...</think>. "
    "Do not output any analysis outside <think>. "
    "Format every reply as <think>...</think> then <final>...</final>, with no other text. "
    "Only the final user-visible reply may appear inside <final>. Only text inside <final> is shown to the user; "
    "everything else is discarded and never seen by the user. Example: <think>Short internal reasoning.</think> "
    "<final>Hey there! What would you like to do next?</final>"
)
REPLACEMENT = local_route._SYSTEM_PROMPT_REPLACEMENT


def _settings(**kw):
    return make_settings(**kw)


# ---- env-driven overrides ------------------------------------------------------------

def test_no_overrides_is_a_noop():
    payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    assert local_route.adapt(payload, _settings()) is False
    assert payload == {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


def test_every_override_applies():
    payload = {"model": "m", "messages": []}
    settings = _settings(
        local_model_override="override-model",
        local_stop="</final>,/>",
        local_max_tokens=600,
        local_enable_thinking=False,
    )
    assert local_route.adapt(payload, settings) is True
    assert payload["model"] == "override-model"
    assert payload["stop"] == ["</final>", "/>"]
    assert payload["max_completion_tokens"] == 600 and payload["max_tokens"] == 600
    assert payload["chat_template_kwargs"]["enable_thinking"] is False


def test_enable_thinking_preserves_other_template_kwargs():
    payload = {"chat_template_kwargs": {"other": 1}, "messages": []}
    local_route.adapt(payload, _settings(local_enable_thinking=True))
    assert payload["chat_template_kwargs"] == {"other": 1, "enable_thinking": True}


def test_enable_thinking_replaces_non_dict_template_kwargs():
    payload = {"chat_template_kwargs": "junk", "messages": []}
    local_route.adapt(payload, _settings(local_enable_thinking=True))
    assert payload["chat_template_kwargs"] == {"enable_thinking": True}


def test_empty_stop_segments_are_dropped():
    payload = {"messages": []}
    local_route.adapt(payload, _settings(local_stop="</final>,,"))
    assert payload["stop"] == ["</final>"]


# ---- system-prompt cleaning ----------------------------------------------------------

@pytest.mark.parametrize("role", ["system", "developer"])
def test_cleans_string_content(role):
    payload = {"messages": [{"role": role, "content": f"Prefix.\n{WRAPPING}\nSuffix."}]}
    assert local_route.adapt(payload, _settings()) is True
    cleaned = payload["messages"][0]["content"]
    assert "ALL internal reasoning MUST be inside" not in cleaned
    assert REPLACEMENT in cleaned
    assert cleaned.startswith("Prefix.\n") and cleaned.endswith("\nSuffix.")


def test_cleans_each_text_part_of_list_content():
    payload = {"messages": [{"role": "system", "content": [
        {"type": "text", "text": "Prefix."},
        {"type": "text", "text": WRAPPING},
        {"type": "image_url", "image_url": {"url": "x"}},   # non-text part untouched
    ]}]}
    assert local_route.adapt(payload, _settings()) is True
    parts = payload["messages"][0]["content"]
    assert parts[0]["text"] == "Prefix."
    assert REPLACEMENT in parts[1]["text"]
    assert parts[2] == {"type": "image_url", "image_url": {"url": "x"}}


def test_user_and_assistant_turns_are_never_touched():
    payload = {"messages": [
        {"role": "user", "content": WRAPPING},
        {"role": "assistant", "content": WRAPPING},
    ]}
    assert local_route.adapt(payload, _settings()) is False
    assert payload["messages"][0]["content"] == WRAPPING
    assert payload["messages"][1]["content"] == WRAPPING


def test_system_prompt_without_instructions_is_untouched():
    payload = {"messages": [{"role": "system", "content": "You are a helpful assistant."}]}
    assert local_route.adapt(payload, _settings()) is False
    assert payload["messages"][0]["content"] == "You are a helpful assistant."


def test_detected_but_regex_missed_warns_and_leaves_content_alone(caplog):
    """A client edits its template past what the regex matches: leave the prompt alone
    and say so, rather than mangling it."""
    partial = "Use <think> for reasoning and <final> for the answer."  # detected, not matched
    payload = {"messages": [{"role": "system", "content": partial}]}
    with caplog.at_level(logging.WARNING, logger="agentgate"):
        assert local_route.adapt(payload, _settings()) is False
    assert payload["messages"][0]["content"] == partial
    assert "regex adapter failed to match" in caplog.text


def test_missing_or_malformed_messages_is_survivable():
    for payload in ({}, {"messages": "not a list"}, {"messages": [None, "junk"]}):
        assert local_route.adapt(payload, _settings()) is False


def test_cleaning_alone_reports_mutation():
    """Cleaning must report "mutated" even with no env overrides set — otherwise the
    body is never re-serialized and the cleaned prompt is silently dropped."""
    payload = {"messages": [{"role": "system", "content": WRAPPING}]}
    assert local_route.adapt(payload, _settings()) is True
