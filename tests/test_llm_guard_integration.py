"""Integration and unit tests for the local LLM guard and combined backend.

Ensures that:
1. User turns are never scanned by the LLM guard (tool-output scoped) — now a property
   of the `guards.scan` driver, which owns extraction for every backend.
2. Tool outputs are scanned and flagged appropriately by the LLM guard.
3. The combined backend correctly performs max/OR combinations of DeBERTa and LLM verdicts.
"""

from __future__ import annotations

import json
import threading
import time

import httpx

from agentgate import content, guards
from agentgate.guards import Verdict, deberta, local_llm
from agentgate.guards.local_llm import JudgeConfig, LLMGuard
from tests.support import make_settings


def test_llm_scan_surface_excludes_user_turn():
    """The LLM guard's surface is the trailing tool-output batch and nothing else."""
    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "User prompt with malicious ignore instructions"},
        {"role": "assistant", "content": "Assistant response"},
        {"role": "tool", "content": "Clean tool output"},
    ]
    untrusted = content.trailing_tool_outputs(messages)
    assert len(untrusted) == 1
    assert content.coerce_content(untrusted[0].get("content")) == "Clean tool output"

    # No tool messages → nothing for the LLM guard to scan.
    assert content.trailing_tool_outputs([{"role": "user", "content": "User turn"}]) == []


async def test_llm_backend_excludes_user_turn(monkeypatch):
    """An injection in the user turn is never submitted to the LLM guard; one in the
    tool turn is scanned and flagged."""
    scanned_texts = []

    def mock_scan_text(text):
        scanned_texts.append(text)
        if "injection" in text:
            return Verdict(flagged=True, score=1.0, reasons=["mock_flag"], hard=True)
        return Verdict.clean()

    monkeypatch.setattr(local_llm, "scan_text", mock_scan_text)

    # 1. Injection in user turn only.
    verdict = await guards.scan("llm", [
        {"role": "user", "content": "injection payload here"},
        {"role": "tool", "content": "clean tool output"},
    ])
    assert not verdict.flagged
    assert verdict.score == 0.0
    assert "clean tool output" in scanned_texts
    assert "injection payload here" not in scanned_texts

    scanned_texts.clear()

    # 2. Injection in tool turn.
    verdict = await guards.scan("llm", [
        {"role": "user", "content": "clean user prompt"},
        {"role": "tool", "content": "injection payload here"},
    ])
    assert verdict.flagged
    assert verdict.score == 1.0
    assert verdict.hard
    assert "tool_output:mock_flag" in verdict.reasons
    assert "injection payload here" in scanned_texts
    assert "clean user prompt" not in scanned_texts


async def test_deberta_backend_includes_user_turn(monkeypatch):
    """The asymmetry is deliberate: deberta DOES see the newest user turn (docs/index.md,
    "How I measured this"). Pinning it here so the driver can't quietly unify the two."""
    scanned_texts = []

    def mock_scan_text(text):
        scanned_texts.append(text)
        return Verdict.clean()

    monkeypatch.setattr(deberta, "scan_text", mock_scan_text)
    await guards.scan("deberta", [
        {"role": "user", "content": "the operator's own words"},
        {"role": "tool", "content": "tool output"},
    ])
    assert "the operator's own words" in scanned_texts
    assert "tool output" in scanned_texts


async def test_combined_backend_verdict_combination(monkeypatch):
    """Verify the combined backend combine rule (max/OR) over DeBERTa and LLM verdicts.

    Asserts that:
    - combined.flagged = deberta.flagged OR llm.flagged
    - combined.hard = deberta.hard OR llm.hard
    - combined.score = max(deberta.score, llm.score)
    - combined.reasons = deberta.reasons + llm.reasons
    """
    settings = make_settings()
    settings.guard_backend = "combined"

    def mock_deberta_scan_text(text):
        if "deberta_trigger" in text:
            return Verdict(flagged=True, score=0.6, reasons=["deberta_flag"], hard=False)
        return Verdict.clean()

    def mock_llm_scan_text(text):
        if "llm_trigger" in text:
            return Verdict(flagged=True, score=1.0, reasons=["llm_flag"], hard=True)
        return Verdict.clean()

    monkeypatch.setattr(deberta, "scan_text", mock_deberta_scan_text)
    monkeypatch.setattr(local_llm, "scan_text", mock_llm_scan_text)

    # Case 1: Both clean
    v = await guards.scan("combined", [
        {"role": "user", "content": "clean prompt"},
        {"role": "tool", "content": "clean tool"},
    ])
    assert not v.flagged
    assert v.score == 0.0
    assert not v.hard
    assert len(v.reasons) == 0

    # Case 2: DeBERTa flags (on the user turn, which only it sees), LLM clean
    v = await guards.scan("combined", [
        {"role": "user", "content": "deberta_trigger"},
        {"role": "tool", "content": "clean tool"},
    ])
    assert v.flagged
    assert v.score == 0.6
    assert not v.hard
    assert v.reasons == ["user:deberta_flag"]

    # Case 3: LLM flags, DeBERTa clean
    v = await guards.scan("combined", [
        {"role": "user", "content": "clean prompt"},
        {"role": "tool", "content": "llm_trigger"},
    ])
    assert v.flagged
    assert v.score == 1.0
    assert v.hard
    assert v.reasons == ["tool_output:llm_flag"]

    # Case 4: Both flag (max score, OR'd flags, concatenated reasons)
    v = await guards.scan("combined", [
        {"role": "user", "content": "deberta_trigger"},
        {"role": "tool", "content": "llm_trigger"},
    ])
    assert v.flagged
    assert v.score == 1.0
    assert v.hard
    assert "user:deberta_flag" in v.reasons
    assert "tool_output:llm_flag" in v.reasons


def test_cache_concurrency_race(tmp_path, monkeypatch):
    cache_path = tmp_path / "judge_cache.json"

    # Mock LLMGuard client post to simulate concurrent requests taking some time
    calls = []
    def mock_post(self_client, url, headers, json):
        text = json["messages"][1]["content"]
        calls.append(text)
        # Wire-contract regression lock: per-request client config disables
        # adaptive reasoning and forces JSON output.
        assert json["chat_template_kwargs"]["enable_thinking"] is False
        assert json["response_format"] == {"type": "json_object"}
        # Sleep to ensure overlap/concurrency
        time.sleep(0.1)
        label = 1 if "injection" in text else 0
        response_content = f'{{"label": {label}, "confidence": 0.9, "rationale": "mocked"}}'

        # Return a mock response object
        class MockResponse:
            def raise_for_status(self):
                pass
            def json(self):
                return {"choices": [{"message": {"content": response_content}}]}
        return MockResponse()

    monkeypatch.setattr(httpx.Client, "post", mock_post)

    # The cache path is now injected config, not a module global to monkeypatch.
    cfg = JudgeConfig(api_key="x", base_url="http://127.0.0.1:8000/v1", model="mock",
                      cache_path=str(cache_path))
    guard = LLMGuard(cfg)

    # Run concurrent threads
    t1 = threading.Thread(target=guard.scan_text, args=("injection one",))
    t2 = threading.Thread(target=guard.scan_text, args=("injection two",))

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Verify the ON-DISK cache file (not the in-memory _SHARED_CACHES registry) is
    # valid JSON and holds BOTH concurrent writes. The original bug was file
    # corruption / lost writes from unsynchronized read-modify-write save() across
    # threads, so the real regression check must re-read from disk — asserting
    # against the shared in-memory dict (which both threads populate directly)
    # would pass even on a corrupt file.
    on_disk = json.loads(cache_path.read_text())
    assert len(on_disk) == 2, f"expected 2 entries persisted to disk, got {len(on_disk)}: {list(on_disk)}"
