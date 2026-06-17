"""Integration and unit tests for the local LLM guard and combined backend.

Ensures that:
1. User turns are never scanned by the LLM guard (tool-output scoped).
2. Tool outputs are scanned and flagged appropriately by the LLM guard.
3. The combined backend correctly performs max/OR combinations of DeBERTa and LLM verdicts.
"""

from __future__ import annotations

import pytest

from agentgate.security import llm_guard, model_guard
from agentgate.security.injection import Verdict


def test_extract_untrusted_llm_excludes_user_turn():
    """Verify the trust boundary design guarantee: extract_untrusted_llm extracts ONLY

    tool messages and never user messages.
    """
    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "User prompt with malicious ignore instructions"},
        {"role": "assistant", "content": "Assistant response"},
        {"role": "tool", "content": "Clean tool output"},
    ]
    untrusted = llm_guard.extract_untrusted_llm(messages)
    assert len(untrusted) == 1
    assert untrusted[0][0] == "tool_output"
    assert untrusted[0][1] == "Clean tool output"

    # Test with no tool messages
    messages_no_tool = [
        {"role": "user", "content": "User turn"},
    ]
    assert len(llm_guard.extract_untrusted_llm(messages_no_tool)) == 0


def test_llm_guard_scan_request_excludes_user_turn(monkeypatch):
    """Verify that an injection in the user turn is not scanned or flagged by LLMGuard,

    but an injection in the tool turn is scanned and flagged.
    """
    # Mock the actual LLM call to scan_text
    scanned_texts = []

    def mock_scan_text(text):
        scanned_texts.append(text)
        if "injection" in text:
            return Verdict(flagged=True, score=1.0, reasons=["mock_flag"], hard=True)
        return Verdict.clean()

    monkeypatch.setattr(llm_guard, "scan_text", mock_scan_text)

    # 1. Injection in user turn only
    messages_user_injection = [
        {"role": "user", "content": "injection payload here"},
        {"role": "tool", "content": "clean tool output"},
    ]
    verdict = llm_guard.scan_request(messages_user_injection)
    assert not verdict.flagged
    assert verdict.score == 0.0
    assert "clean tool output" in scanned_texts
    assert "injection payload here" not in scanned_texts

    scanned_texts.clear()

    # 2. Injection in tool turn
    messages_tool_injection = [
        {"role": "user", "content": "clean user prompt"},
        {"role": "tool", "content": "injection payload here"},
    ]
    verdict = llm_guard.scan_request(messages_tool_injection)
    assert verdict.flagged
    assert verdict.score == 1.0
    assert verdict.hard
    assert "tool_output:mock_flag" in verdict.reasons
    assert "injection payload here" in scanned_texts
    assert "clean user prompt" not in scanned_texts


@pytest.mark.asyncio
async def test_combined_backend_verdict_combination(monkeypatch):
    """Verify the combined backend combine rule (max/OR) over DeBERTa and LLM verdicts.

    Asserts that:
    - combined.flagged = deberta.flagged OR llm.flagged
    - combined.hard = deberta.hard OR llm.hard
    - combined.score = max(deberta.score, llm.score)
    - combined.reasons = deberta.reasons + llm.reasons
    """
    from agentgate.app import _scan_request
    from agentgate.config import Settings

    settings = Settings()
    settings.guard_backend = "combined"

    # Mock DeBERTa scan_request
    def mock_deberta_scan(messages):
        if any("deberta_trigger" in m.get("content", "") for m in messages):
            return Verdict(flagged=True, score=0.6, reasons=["deberta_flag"], hard=False)
        return Verdict.clean()

    # Mock LLM scan_request
    def mock_llm_scan(messages):
        # Locate the mock scan trigger in tool messages
        for msg in messages:
            if msg.get("role") == "tool" and "llm_trigger" in msg.get("content", ""):
                return Verdict(flagged=True, score=1.0, reasons=["llm_flag"], hard=True)
        return Verdict.clean()

    monkeypatch.setattr(model_guard, "scan_request", mock_deberta_scan)
    monkeypatch.setattr(llm_guard, "scan_request", mock_llm_scan)

    # Case 1: Both clean
    messages = [
        {"role": "user", "content": "clean prompt"},
        {"role": "tool", "content": "clean tool"},
    ]
    v = await _scan_request(settings, messages)
    assert not v.flagged
    assert v.score == 0.0
    assert not v.hard
    assert len(v.reasons) == 0

    # Case 2: DeBERTa flags, LLM clean
    messages = [
        {"role": "user", "content": "deberta_trigger"},
        {"role": "tool", "content": "clean tool"},
    ]
    v = await _scan_request(settings, messages)
    assert v.flagged
    assert v.score == 0.6
    assert not v.hard
    assert v.reasons == ["deberta_flag"]

    # Case 3: LLM flags, DeBERTa clean
    messages = [
        {"role": "user", "content": "clean prompt"},
        {"role": "tool", "content": "llm_trigger"},
    ]
    v = await _scan_request(settings, messages)
    assert v.flagged
    assert v.score == 1.0
    assert v.hard
    assert v.reasons == ["llm_flag"]

    # Case 4: Both flag (strongest verdict wins, combined reasons)
    messages = [
        {"role": "user", "content": "deberta_trigger"},
        {"role": "tool", "content": "llm_trigger"},
    ]
    v = await _scan_request(settings, messages)
    assert v.flagged
    assert v.score == 1.0
    assert v.hard
    assert "deberta_flag" in v.reasons
    assert "llm_flag" in v.reasons


def test_cache_concurrency_race(tmp_path, monkeypatch):
    import threading
    import time

    import httpx

    from agentgate.security.llm_guard import JudgeConfig, LLMGuard

    # Set a temp cache path to avoid modifying production cache
    monkeypatch.setattr("agentgate.security.llm_guard._CACHE_PATH", tmp_path / "judge_cache.json")
    
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
    
    # Instantiate guard
    cfg = JudgeConfig(api_key="x", base_url="http://127.0.0.1:8000/v1", model="mock")
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
    import json as _json
    on_disk = _json.loads((tmp_path / "judge_cache.json").read_text())
    assert len(on_disk) == 2, f"expected 2 entries persisted to disk, got {len(on_disk)}: {list(on_disk)}"

