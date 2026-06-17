"""Sensitivity classifier + rules-table router — pure-logic tests (no I/O)."""

from __future__ import annotations

from agentgate.config import RoutingConfig, Settings
from agentgate.routing import router as rr
from agentgate.routing.providers import resolve
from agentgate.security.classifier import Sensitivity, classify, classify_request

# ---- classifier ------------------------------------------------------------

def test_classify_secret_beats_pii():
    # Secret + email present → secret wins (most-sensitive precedence).
    r = classify("contact me@x.com — key sk-abcdefghijklmnopqrstuvwx")
    assert r.sensitivity is Sensitivity.SECRET


def test_classify_private_key_and_aws():
    assert classify("-----BEGIN OPENSSH PRIVATE KEY-----").sensitivity is Sensitivity.SECRET
    assert classify("AKIAIOSFODNN7EXAMPLE").sensitivity is Sensitivity.SECRET


def test_classify_pii_email_phone():
    assert classify("reach me at jane@example.com").sensitivity is Sensitivity.PII
    assert classify("call 415-555-0123").sensitivity is Sensitivity.PII


def test_classify_none_for_benign():
    assert classify("what's the capital of France?").sensitivity is Sensitivity.NONE


def test_classify_private_repo_marker():
    r = classify("see internal/secret-project/readme", markers=["internal/secret-project"])
    assert r.sensitivity is Sensitivity.PRIVATE_REPO


def test_classify_request_scans_all_messages():
    msgs = [{"role": "user", "content": "hello"},
            {"role": "tool", "content": "token AKIAIOSFODNN7EXAMPLE here"}]
    assert classify_request(msgs).sensitivity is Sensitivity.SECRET


# ---- router ----------------------------------------------------------------

CFG = RoutingConfig()  # default rules table


def test_sensitive_routes_local_even_for_cloud_pinned_agent():
    d = rr.decide(rr.RouteContext(sensitivity="secret", agent_id="capture"), CFG)
    assert d.is_local and d.rule == "sensitive-stays-local"


def test_cloud_failure_routes_local():
    d = rr.decide(rr.RouteContext(sensitivity="none", cloud_unavailable=True), CFG)
    assert d.is_local and d.rule == "fallback-on-failure"


def test_over_spend_cap_routes_local():
    d = rr.decide(rr.RouteContext(sensitivity="none", over_spend_cap=True), CFG)
    assert d.is_local and d.rule == "fallback-on-failure"


def test_agent_pin_prefers_cloud():
    d = rr.decide(rr.RouteContext(sensitivity="none", agent_id="web-research"), CFG)
    assert not d.is_local and d.rule == "agent-pins"


def test_default_is_cloud():
    d = rr.decide(rr.RouteContext(sensitivity="none", agent_id="other"), CFG)
    assert not d.is_local and d.rule == "default"


def test_secrets_to_cloud_knob():
    # Default: secret stays local (unchanged behavior).
    cfg_off = RoutingConfig()
    d = rr.decide(rr.RouteContext(sensitivity="secret"), cfg_off)
    assert d.is_local and d.rule == "sensitive-stays-local"

    # Knob on: secret falls through to default→cloud; pii is untouched.
    cfg_on = RoutingConfig(secrets_to_cloud=True)
    d_secret = rr.decide(rr.RouteContext(sensitivity="secret"), cfg_on)
    assert not d_secret.is_local and d_secret.rule == "default"

    d_pii = rr.decide(rr.RouteContext(sensitivity="pii"), cfg_on)
    assert d_pii.is_local and d_pii.rule == "sensitive-stays-local"


def test_resolve_maps_decision_to_provider():
    s = Settings()
    local = resolve(s, rr.decide(rr.RouteContext(sensitivity="pii"), CFG))
    cloud = resolve(s, rr.decide(rr.RouteContext(sensitivity="none"), CFG))
    assert local.is_local and local.name == "local"
    assert not cloud.is_local and cloud.name == "gemini"
