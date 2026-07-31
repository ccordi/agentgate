"""Tests for per-key injection-guard backend selection.

Covers `guards.resolve_backend` / `guards.scan`'s per-key dispatch:
- guard_backend_overrides routes a key to its mapped backend; unmapped keys fall
  back to the global default.
- DeBERTa unavailability changes only the current key's resolved backend.
- "combined" still composes both scans, and degrades to "llm" when deberta is absent.
- Settings validation fails fast on an unrecognized backend in guard_backend or
  guard_backend_overrides.
"""

from __future__ import annotations

import pytest

from agentgate import guards
from agentgate.config import Settings
from agentgate.guards import Verdict, deberta, local_llm
from tests.support import make_settings


async def _scan(settings, messages, key_id, *, deberta_available):
    """What the pipeline does: resolve the backend for this key, then scan with it."""
    backend = guards.resolve_backend(settings, key_id, deberta_available)
    return await guards.scan(backend, messages)


# Benign, and carries BOTH untrusted channels: the deberta/heuristic backends scan the
# tool output and the user turn, the LLM backend only the tool output.
MESSAGES = [
    {"role": "user", "content": "hello"},
    {"role": "tool", "tool_call_id": "c1", "content": "tool result"},
]

KEY_DEBERTA = "key-deberta"
KEY_LLM = "key-llm"
KEY_COMBINED = "key-combined"


def _settings(**overrides) -> Settings:
    return make_settings(
        guard_backend="heuristic",
        guard_backend_overrides={
            KEY_DEBERTA: "deberta",
            KEY_LLM: "llm",
            KEY_COMBINED: "combined",
        },
        **overrides,
    )


# The driver owns extraction now, so mocks patch `scan_text` (per text), not a
# per-request scanner.
def _recorder(monkeypatch, module, calls: list) -> None:
    """Patch ``module.scan_text`` with a clean-verdict stub that records each call."""
    def fake(text):
        calls.append(text)
        return Verdict.clean()
    monkeypatch.setattr(module, "scan_text", fake)


def _mock_deberta(monkeypatch, flagged=False):
    def fake(text):
        return Verdict(flagged=flagged, score=0.9 if flagged else 0.0,
                        reasons=["deberta_flag"] if flagged else [], hard=flagged)
    monkeypatch.setattr(deberta, "scan_text", fake)
    return fake


def _mock_llm(monkeypatch, flagged=False):
    def fake(text):
        return Verdict(flagged=flagged, score=0.8 if flagged else 0.0,
                        reasons=["llm_flag"] if flagged else [], hard=flagged)
    monkeypatch.setattr(local_llm, "scan_text", fake)
    return fake


async def test_mapped_keys_route_to_their_backend(monkeypatch):
    """key A -> deberta, key B -> llm, unmapped -> default (heuristic)."""
    settings = _settings()
    deberta_calls = []
    llm_calls = []
    _recorder(monkeypatch, deberta, deberta_calls)
    _recorder(monkeypatch, local_llm, llm_calls)

    await _scan(settings, MESSAGES, KEY_DEBERTA, deberta_available=True)
    assert deberta_calls and not llm_calls

    deberta_calls.clear()
    await _scan(settings, MESSAGES, KEY_LLM, deberta_available=True)
    assert not deberta_calls and llm_calls

    # Unmapped key falls back to the global default ("heuristic") -> neither mock called.
    llm_calls.clear()
    verdict = await _scan(settings, MESSAGES, "unmapped-key", deberta_available=True)
    assert not deberta_calls and not llm_calls
    assert verdict == Verdict.clean()


async def test_deberta_absent_falls_back_per_request_without_clobbering_other_keys(monkeypatch):
    """A DeBERTa fallback for one key does not change another key's LLM route."""
    settings = _settings()
    llm_calls = []
    deberta_calls = []
    _recorder(monkeypatch, deberta, deberta_calls)
    _recorder(monkeypatch, local_llm, llm_calls)

    # deberta-mapped key, deberta unavailable -> falls back to heuristic (no deberta call).
    verdict = await _scan(settings, MESSAGES, KEY_DEBERTA, deberta_available=False)
    assert deberta_calls == []
    assert verdict == Verdict.clean()  # heuristic on a benign message

    # llm-mapped key is unaffected by the other key's fallback.
    await _scan(settings, MESSAGES, KEY_LLM, deberta_available=False)
    assert deberta_calls == []
    assert llm_calls


async def test_combined_composes_both_backends(monkeypatch):
    """A key -> "combined" runs both scans and OR/max-combines the verdicts."""
    settings = _settings()
    _mock_deberta(monkeypatch, flagged=True)
    _mock_llm(monkeypatch, flagged=False)

    verdict = await _scan(settings, MESSAGES, KEY_COMBINED, deberta_available=True)

    assert verdict.flagged
    assert verdict.score == 0.9
    assert verdict.reasons == ["tool_output:deberta_flag"]
    assert verdict.hard


async def test_combined_falls_back_to_llm_when_deberta_absent(monkeypatch):
    """combined -> llm when deberta is unavailable; the deberta backend is never called."""
    settings = _settings()
    deberta_calls = []
    _recorder(monkeypatch, deberta, deberta_calls)
    _mock_llm(monkeypatch, flagged=True)

    verdict = await _scan(settings, MESSAGES, KEY_COMBINED, deberta_available=False)

    assert deberta_calls == []
    assert verdict.flagged
    assert verdict.reasons == ["tool_output:llm_flag"]


def test_resolve_guard_backend_matches_dispatch():
    """_resolve_guard_backend (used for audit) mirrors _scan_request's resolution,
    including the deberta-availability fallback."""
    settings = _settings()

    assert guards.resolve_backend(settings, KEY_DEBERTA, deberta_available=True) == "deberta"
    assert guards.resolve_backend(settings, KEY_LLM, deberta_available=True) == "llm"
    assert guards.resolve_backend(settings, KEY_COMBINED, deberta_available=True) == "combined"
    assert guards.resolve_backend(settings, "unmapped-key", deberta_available=True) == "heuristic"

    # Fallbacks when deberta is absent.
    assert guards.resolve_backend(settings, KEY_DEBERTA, deberta_available=False) == "heuristic"
    assert guards.resolve_backend(settings, KEY_COMBINED, deberta_available=False) == "llm"
    # llm-mapped key is unaffected.
    assert guards.resolve_backend(settings, KEY_LLM, deberta_available=False) == "llm"


def test_unknown_backend_in_guard_backend_fails_fast():
    with pytest.raises(ValueError, match="guard_backend"):
        Settings(guard_backend="not-a-real-backend")


def test_unknown_backend_in_overrides_fails_fast():
    with pytest.raises(ValueError, match="guard_backend_overrides"):
        Settings(guard_backend="heuristic",
                 guard_backend_overrides={"some-key": "not-a-real-backend"})
