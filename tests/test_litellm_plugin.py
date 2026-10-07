"""Tests for the LiteLLM guardrail plugin.

Skipped when the `litellm-plugin` extra is not installed. Every test uses the built-in
heuristic scanner, so no model files are needed; the classifier appears only in
the test that checks the plugin refuses to start when the classifier cannot load.
"""

from __future__ import annotations

import pytest

litellm = pytest.importorskip("litellm")

from litellm.proxy._types import ProxyException  # noqa: E402

from agentgate.integrations import litellm_guard  # noqa: E402
from tests.support import HARD_INJECTION  # noqa: E402

BENIGN = [{"role": "user", "content": "summarize this article"}]
ATTACK = [{"role": "user", "content": HARD_INJECTION}]


def _guard(**kwargs) -> litellm_guard.AgentgateGuard:
    kwargs.setdefault("backend", "heuristic")
    return litellm_guard.AgentgateGuard(guardrail_name="agentgate-test", **kwargs)


async def test_benign_passes_through_unchanged():
    guard = _guard()
    data = {"model": "m", "messages": BENIGN}
    assert await guard.async_pre_call_hook(None, None, data, "completion") is data


async def test_hard_injection_blocks_with_gateway_wire_convention():
    guard = _guard()
    with pytest.raises(ProxyException) as exc:
        await guard.async_pre_call_hook(None, None, {"messages": ATTACK}, "completion")
    envelope = exc.value.to_dict()
    assert exc.value.code == "400"
    assert envelope["type"] == "injection_blocked"
    # No message content in the refusal — same discipline as the gateway's 400.
    assert HARD_INJECTION not in str(envelope)


async def test_messageless_calls_pass_unscanned():
    """Embeddings-style payloads have no scan surface; a pass, not a block."""
    guard = _guard()
    data = {"model": "m", "input": HARD_INJECTION}
    assert await guard.async_pre_call_hook(None, None, data, "embeddings") is data


def test_unknown_backend_fails_fast():
    with pytest.raises(ValueError, match="unknown guard backend"):
        _guard(backend="typo")


@pytest.mark.parametrize("requested", ["deberta", "combined"])
def test_deberta_unavailable_refuses_to_arm(monkeypatch, caplog, requested):
    """The gateway's startup rule, applied inside the LiteLLM plugin: a configured scanner
    that cannot load refuses, so the proxy fails to boot rather than serving every request
    with a weaker scanner than its configuration asked for. `combined` refuses too —
    running the half that loaded, under the name of the composition, is the same silent
    downgrade.
    """
    from agentgate.guards import deberta

    def boom():
        raise RuntimeError("model not found")

    monkeypatch.setattr(deberta, "warmup", boom)
    with caplog.at_level("CRITICAL", logger="agentgate.integrations.litellm"):
        with pytest.raises(RuntimeError, match="could not load") as exc:
            _guard(backend=requested)
    assert "AGENTGATE_GUARD_MODEL_DIR" in str(exc.value)
    assert any("DEBERTA UNAVAILABLE" in r.message for r in caplog.records)


async def test_scan_failure_fails_closed_with_503(monkeypatch):
    """A control that cannot run must refuse, not pass — and in the gateway's
    convention rather than an unhandled 500."""
    async def boom(backend, messages):
        raise RuntimeError("judge unreachable")

    monkeypatch.setattr(litellm_guard.guards, "scan", boom)
    guard = _guard()
    with pytest.raises(ProxyException) as exc:
        await guard.async_pre_call_hook(None, None, {"messages": BENIGN}, "completion")
    assert exc.value.code == "503"
    assert exc.value.to_dict()["type"] == "guard_unavailable"


# --- observe mode (`mode: logging_only` -> async_logging_hook) ---------------
# LiteLLM's logging path calls the hook with keyword args
# (kwargs=model_call_details, result=response, call_type=...) and expects the
# (kwargs, result) pair back; the tests call it the same way.


async def test_observe_hook_logs_injection_and_never_raises(caplog):
    guard = _guard()
    kwargs = {"model": "m", "messages": ATTACK}
    result = {"choices": []}
    with caplog.at_level("INFO", logger="agentgate.integrations.litellm"):
        out = await guard.async_logging_hook(kwargs=kwargs, result=result, call_type="acompletion")
    assert out[0] is kwargs and out[1] is result
    observed = [r for r in caplog.records if "injection observed" in r.message]
    assert observed and observed[0].levelname == "WARNING"
    assert "verdict=hard" in observed[0].message
    # Same discipline as the block path: no message content in the log line.
    assert HARD_INJECTION not in observed[0].message


async def test_observe_hook_benign_passes_quietly(caplog):
    guard = _guard()
    kwargs = {"model": "m", "messages": BENIGN}
    result = {"choices": []}
    with caplog.at_level("INFO", logger="agentgate.integrations.litellm"):
        caplog.clear()  # drop the init-time "guardrail armed" line
        out = await guard.async_logging_hook(kwargs=kwargs, result=result, call_type="acompletion")
    assert out[0] is kwargs and out[1] is result
    assert not [r for r in caplog.records if r.name == "agentgate.integrations.litellm"]


async def test_observe_hook_swallows_scan_failure(monkeypatch, caplog):
    """The response is already sent when this hook runs; a scan failure is
    logged and passed, never raised — the opposite posture of pre_call's 503."""
    async def boom(backend, messages):
        raise RuntimeError("judge unreachable")

    monkeypatch.setattr(litellm_guard.guards, "scan", boom)
    guard = _guard()
    kwargs = {"messages": BENIGN}
    result = {"choices": []}
    with caplog.at_level("ERROR", logger="agentgate.integrations.litellm"):
        out = await guard.async_logging_hook(kwargs=kwargs, result=result, call_type="acompletion")
    assert out[0] is kwargs and out[1] is result
    assert any("observe mode" in r.message for r in caplog.records)


async def test_observe_hook_messageless_call_passes(caplog):
    """No scan surface (embeddings-style payload): pass, no log line."""
    guard = _guard()
    kwargs = {"model": "m", "input": HARD_INJECTION}
    result = {"data": []}
    with caplog.at_level("INFO", logger="agentgate.integrations.litellm"):
        caplog.clear()
        out = await guard.async_logging_hook(kwargs=kwargs, result=result, call_type="aembedding")
    assert out[0] is kwargs and out[1] is result
    assert not [r for r in caplog.records if r.name == "agentgate.integrations.litellm"]
