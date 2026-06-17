"""Tests for per-key injection-guard backend selection.

Covers `_resolve_guard_backend` / `_scan_request`'s per-key dispatch:
- guard_backend_overrides routes a key to its mapped backend; unmapped keys fall
  back to the global default.
- deberta-unavailable degrades a key resolving to "deberta"/"combined" for THAT
  REQUEST ONLY, without affecting a different key resolved to a different backend
  (the cross-key clobber the old global-mutation approach was prone to).
- "combined" still composes both scans, and degrades to "llm" when deberta is absent.
- Settings validation fails fast on an unrecognized backend in guard_backend or
  guard_backend_overrides.
"""

from __future__ import annotations

import pytest

from agentgate.app import _resolve_guard_backend, _scan_request
from agentgate.config import Settings
from agentgate.security import llm_guard, model_guard
from agentgate.security.injection import Verdict

KEY_DEBERTA = "key-deberta"
KEY_LLM = "key-llm"
KEY_COMBINED = "key-combined"


def _settings(**overrides) -> Settings:
    return Settings(
        guard_backend="heuristic",
        guard_backend_overrides={
            KEY_DEBERTA: "deberta",
            KEY_LLM: "llm",
            KEY_COMBINED: "combined",
        },
        **overrides,
    )


def _mock_deberta(monkeypatch, flagged=False):
    def fake(messages):
        return Verdict(flagged=flagged, score=0.9 if flagged else 0.0,
                        reasons=["deberta_flag"] if flagged else [], hard=flagged)
    monkeypatch.setattr(model_guard, "scan_request", fake)
    return fake


def _mock_llm(monkeypatch, flagged=False):
    def fake(messages):
        return Verdict(flagged=flagged, score=0.8 if flagged else 0.0,
                        reasons=["llm_flag"] if flagged else [], hard=flagged)
    monkeypatch.setattr(llm_guard, "scan_request", fake)
    return fake


@pytest.mark.asyncio
async def test_mapped_keys_route_to_their_backend(monkeypatch):
    """key A -> deberta, key B -> llm, unmapped -> default (heuristic)."""
    settings = _settings()
    deberta_calls = []
    llm_calls = []
    monkeypatch.setattr(model_guard, "scan_request",
                         lambda m: (deberta_calls.append(1), Verdict.clean())[1])
    monkeypatch.setattr(llm_guard, "scan_request",
                         lambda m: (llm_calls.append(1), Verdict.clean())[1])

    messages = [{"role": "user", "content": "hello"}]

    await _scan_request(settings, messages, key_id=KEY_DEBERTA, deberta_available=True)
    assert deberta_calls == [1]
    assert llm_calls == []

    await _scan_request(settings, messages, key_id=KEY_LLM, deberta_available=True)
    assert deberta_calls == [1]
    assert llm_calls == [1]

    # Unmapped key falls back to the global default ("heuristic") -> neither mock called.
    verdict = await _scan_request(settings, messages, key_id="unmapped-key",
                                   deberta_available=True)
    assert deberta_calls == [1]
    assert llm_calls == [1]
    assert verdict == Verdict.clean()


@pytest.mark.asyncio
async def test_deberta_absent_falls_back_per_request_without_clobbering_other_keys(monkeypatch):
    """A key resolving to "deberta" degrades to "heuristic" for that request only;
    a different key resolving to "llm" is unaffected (no cross-key clobber)."""
    settings = _settings()
    llm_calls = []
    deberta_calls = []
    monkeypatch.setattr(model_guard, "scan_request",
                         lambda m: (deberta_calls.append(1), Verdict.clean())[1])
    monkeypatch.setattr(llm_guard, "scan_request",
                         lambda m: (llm_calls.append(1), Verdict.clean())[1])

    messages = [{"role": "user", "content": "hello"}]

    # deberta-mapped key, deberta unavailable -> falls back to heuristic (no model_guard call).
    verdict = await _scan_request(settings, messages, key_id=KEY_DEBERTA,
                                   deberta_available=False)
    assert deberta_calls == []
    assert verdict == Verdict.clean()  # heuristic on a benign message

    # llm-mapped key is unaffected by the other key's fallback.
    await _scan_request(settings, messages, key_id=KEY_LLM, deberta_available=False)
    assert deberta_calls == []
    assert llm_calls == [1]


@pytest.mark.asyncio
async def test_combined_composes_both_backends(monkeypatch):
    """A key -> "combined" runs both scans and OR/max-combines the verdicts."""
    settings = _settings()
    _mock_deberta(monkeypatch, flagged=True)
    _mock_llm(monkeypatch, flagged=False)

    messages = [{"role": "user", "content": "hello"}]
    verdict = await _scan_request(settings, messages, key_id=KEY_COMBINED,
                                   deberta_available=True)

    assert verdict.flagged
    assert verdict.score == 0.9
    assert verdict.reasons == ["deberta_flag"]
    assert verdict.hard


@pytest.mark.asyncio
async def test_combined_falls_back_to_llm_when_deberta_absent(monkeypatch):
    """combined -> llm when deberta is unavailable; model_guard is never called."""
    settings = _settings()
    deberta_calls = []
    monkeypatch.setattr(model_guard, "scan_request",
                         lambda m: (deberta_calls.append(1), Verdict.clean())[1])
    _mock_llm(monkeypatch, flagged=True)

    messages = [{"role": "user", "content": "hello"}]
    verdict = await _scan_request(settings, messages, key_id=KEY_COMBINED,
                                   deberta_available=False)

    assert deberta_calls == []
    assert verdict.flagged
    assert verdict.reasons == ["llm_flag"]


def test_resolve_guard_backend_matches_dispatch():
    """_resolve_guard_backend (used for audit) mirrors _scan_request's resolution,
    including the deberta-availability fallback."""
    settings = _settings()

    assert _resolve_guard_backend(settings, KEY_DEBERTA, deberta_available=True) == "deberta"
    assert _resolve_guard_backend(settings, KEY_LLM, deberta_available=True) == "llm"
    assert _resolve_guard_backend(settings, KEY_COMBINED, deberta_available=True) == "combined"
    assert _resolve_guard_backend(settings, "unmapped-key", deberta_available=True) == "heuristic"

    # Fallbacks when deberta is absent.
    assert _resolve_guard_backend(settings, KEY_DEBERTA, deberta_available=False) == "heuristic"
    assert _resolve_guard_backend(settings, KEY_COMBINED, deberta_available=False) == "llm"
    # llm-mapped key is unaffected.
    assert _resolve_guard_backend(settings, KEY_LLM, deberta_available=False) == "llm"


def test_unknown_backend_in_guard_backend_fails_fast():
    with pytest.raises(ValueError, match="guard_backend"):
        Settings(guard_backend="not-a-real-backend")


def test_unknown_backend_in_overrides_fails_fast():
    with pytest.raises(ValueError, match="guard_backend_overrides"):
        Settings(guard_backend="heuristic",
                 guard_backend_overrides={"some-key": "not-a-real-backend"})
