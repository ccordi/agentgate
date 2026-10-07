"""Tests for per-key injection-guard backend selection.

Covers `guards.resolve_backend` / `guards.scan`'s per-key dispatch:
- guard_backend_overrides routes a key to its mapped backend; unmapped keys fall
  back to the global default.
- the resolved backend is always the configured one. Resolution has no availability
  input to degrade on: a process whose configured backend needs a model it cannot
  load refuses to start (`app.lifespan`), and a model that dies *after* boot is a 503
  `guard_unavailable`, never a quietly weaker scan.
- "combined" composes both scans, and fails closed when either half cannot run.
- Settings validation fails fast on an unrecognized backend in guard_backend or
  guard_backend_overrides.
"""

from __future__ import annotations

import pytest

from agentgate import guards
from agentgate.config import Settings
from agentgate.guards import Verdict, deberta, local_llm
from tests.support import make_settings


async def _scan(settings, messages, key_id):
    """What the pipeline does: resolve the backend for this key, then scan with it."""
    backend = guards.resolve_backend(settings, key_id)
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


# The driver owns extraction, so mocks patch `scan_text` (per text), not a
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

    await _scan(settings, MESSAGES, KEY_DEBERTA)
    assert deberta_calls and not llm_calls

    deberta_calls.clear()
    await _scan(settings, MESSAGES, KEY_LLM)
    assert not deberta_calls and llm_calls

    # Unmapped key falls back to the global default ("heuristic") -> neither mock called.
    llm_calls.clear()
    verdict = await _scan(settings, MESSAGES, "unmapped-key")
    assert not deberta_calls and not llm_calls
    # Clean, but over a surface that was actually examined. Not `== Verdict.clean()`:
    # that would also assert scanned_items == 0 — i.e. pass for a request the guard
    # never looked at, the exact distinction the field exists to make. Assert both
    # halves separately.
    assert (verdict.flagged, verdict.score, verdict.hard) == (False, 0.0, False)
    assert verdict.scanned_items == 2  # the heuristic scanned the tool batch + user turn


async def test_a_model_that_cannot_run_refuses_rather_than_degrades(monkeypatch):
    """A deberta-mapped key whose model fails at scan time gets a refusal, not a
    heuristic verdict wearing deberta's name in the audit row.

    Degrading that key to `heuristic` for the request would hide the failure. A missing
    model is a refused start, and a model that dies after boot is `GuardUnavailable` → 503 —
    either way the operator finds out, and it stays that one key's problem."""
    settings = _settings()
    llm_calls = []
    _recorder(monkeypatch, local_llm, llm_calls)

    def boom(text):
        raise RuntimeError("onnxruntime session gone")

    monkeypatch.setattr(deberta, "scan_text", boom)

    with pytest.raises(guards.GuardUnavailable) as exc:
        await _scan(settings, MESSAGES, KEY_DEBERTA)
    assert exc.value.backend == "deberta"

    # The llm-mapped key scans normally through the same process.
    await _scan(settings, MESSAGES, KEY_LLM)
    assert llm_calls


async def test_combined_composes_both_backends(monkeypatch):
    """A key -> "combined" runs both scans and OR/max-combines the verdicts."""
    settings = _settings()
    _mock_deberta(monkeypatch, flagged=True)
    _mock_llm(monkeypatch, flagged=False)

    verdict = await _scan(settings, MESSAGES, KEY_COMBINED)

    assert verdict.flagged
    assert verdict.score == 0.9
    assert verdict.reasons == ["tool_output:deberta_flag"]
    assert verdict.hard


async def test_combined_fails_closed_when_a_half_cannot_run(monkeypatch):
    """`combined` does not drop to `llm` alone when deberta is unavailable: half a
    composition is a different control, and running it under the name `combined` is the
    silent downgrade this posture exists to prevent."""
    settings = _settings()
    _mock_llm(monkeypatch, flagged=False)

    def boom(text):
        raise RuntimeError("onnxruntime session gone")

    monkeypatch.setattr(deberta, "scan_text", boom)

    with pytest.raises(guards.GuardUnavailable) as exc:
        await _scan(settings, MESSAGES, KEY_COMBINED)
    assert exc.value.backend == "combined"


def test_resolved_backend_is_always_the_configured_one():
    """Resolution is override-else-default and nothing more, so the backend the audit row
    records is the one the operator configured — for every key, mapped or not."""
    settings = _settings()

    assert guards.resolve_backend(settings, KEY_DEBERTA) == "deberta"
    assert guards.resolve_backend(settings, KEY_LLM) == "llm"
    assert guards.resolve_backend(settings, KEY_COMBINED) == "combined"
    assert guards.resolve_backend(settings, "unmapped-key") == "heuristic"
    assert guards.resolve_backend(settings, None) == "heuristic"

    # A deberta default resolves to deberta for every key without an override of its own.
    # There is no availability argument for a caller to weaken that with: startup
    # guarantees the model is loaded, or the process is not running.
    deberta_default = make_settings(guard_backend="deberta",
                                    guard_backend_overrides={KEY_LLM: "llm"})
    assert guards.resolve_backend(deberta_default, "unmapped-key") == "deberta"
    assert guards.resolve_backend(deberta_default, KEY_LLM) == "llm"


def test_unknown_backend_in_guard_backend_fails_fast():
    with pytest.raises(ValueError, match="guard_backend"):
        Settings(guard_backend="not-a-real-backend")


def test_unknown_backend_in_overrides_fails_fast():
    with pytest.raises(ValueError, match="guard_backend_overrides"):
        Settings(guard_backend="heuristic",
                 guard_backend_overrides={"some-key": "not-a-real-backend"})
