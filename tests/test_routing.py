"""Sensitivity classifier + rules-table router — pure-logic tests (no I/O)."""

from __future__ import annotations

from agentgate import redaction, sensitivity
from agentgate.config import RoutingConfig, Settings
from agentgate.routing import RouteContext, decide, resolve
from agentgate.sensitivity import Sensitivity, classify, classify_request

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
    d = decide(RouteContext(sensitivity="secret", agent_id="capture"), CFG)
    assert d.is_local and d.rule == "sensitive-stays-local"


def test_cloud_failure_routes_local():
    d = decide(RouteContext(sensitivity="none", cloud_unavailable=True), CFG)
    assert d.is_local and d.rule == "fallback-on-failure"


def test_over_spend_cap_routes_local():
    d = decide(RouteContext(sensitivity="none", over_spend_cap=True), CFG)
    assert d.is_local and d.rule == "fallback-on-failure"


def test_agent_pin_prefers_cloud():
    d = decide(RouteContext(sensitivity="none", agent_id="web-research"), CFG)
    assert not d.is_local and d.rule == "agent-pins"


def test_default_is_cloud():
    d = decide(RouteContext(sensitivity="none", agent_id="other"), CFG)
    assert not d.is_local and d.rule == "default"


def test_secrets_to_cloud_knob():
    # Default: secret stays local (unchanged behavior).
    cfg_off = RoutingConfig()
    d = decide(RouteContext(sensitivity="secret"), cfg_off)
    assert d.is_local and d.rule == "sensitive-stays-local"

    # Knob on: secret falls through to default→cloud; pii is untouched.
    cfg_on = RoutingConfig(secrets_to_cloud=True)
    d_secret = decide(RouteContext(sensitivity="secret"), cfg_on)
    assert not d_secret.is_local and d_secret.rule == "default"

    d_pii = decide(RouteContext(sensitivity="pii"), cfg_on)
    assert d_pii.is_local and d_pii.rule == "sensitive-stays-local"


def test_resolve_maps_decision_to_provider():
    s = Settings()
    local = resolve(s, decide(RouteContext(sensitivity="pii"), CFG))
    cloud = resolve(s, decide(RouteContext(sensitivity="none"), CFG))
    assert local.is_local and local.name == "local"
    assert not cloud.is_local and cloud.name == "gemini"


def test_secret_types_match_redaction_pattern_names():
    """`sensitivity._SECRET_TYPES` decides secret-vs-PII precedence from the hit-type
    names `redaction.detect()` emits. If a pattern is renamed on one side only, the
    classifier silently downgrades that secret to PII."""
    assert (
        {name for _, name in redaction.SECRET_PATTERNS} | {"high_entropy_token"}
        == sensitivity._SECRET_TYPES
    )


def test_classify_request_reads_tool_call_arguments():
    """A secret passed as a tool-call argument must still route as sensitive.

    On an assistant tool-call message `content` is None, so coerce_content returned ""
    and the message classified as empty — the secret routed to cloud instead of the
    local model, and assistant tool-call history replays every turn.
    """
    msgs = [
        {"role": "user", "content": "ship it"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {
                "name": "deploy",
                "arguments": '{"token": "sk-abcdefghijklmnopqrstuvwx"}',
            }},
        ]},
    ]
    assert classify_request(msgs).sensitivity is Sensitivity.SECRET


def test_coerce_tool_call_args_tolerates_malformed_shapes():
    """Screening runs on attacker-influenced structure, so bad shapes must not raise."""
    from agentgate.content import coerce_tool_call_args

    assert coerce_tool_call_args({"tool_calls": "not-a-list"}) == ""
    assert coerce_tool_call_args({"tool_calls": [None, 3, "x"]}) == ""
    assert coerce_tool_call_args({"tool_calls": [{"function": None}]}) == ""
    assert coerce_tool_call_args({"tool_calls": [{"function": {"arguments": 5}}]}) == ""
    assert coerce_tool_call_args({}) == ""
    assert coerce_tool_call_args(None) == ""
    assert coerce_tool_call_args(
        {"tool_calls": [{"function": {"arguments": "a"}}, {"function": {"arguments": "b"}}]}
    ) == "a\nb"


def test_tool_call_args_never_displace_content_from_the_window():
    """A big benign tool call must not push a later secret out of the classify window.

    Reading tool-call arguments widened what the classifier sees, but the 20 000-char
    bound did not move — so interleaving them meant one ordinary `write_file` call
    truncated every message after it, and a secret that used to classify stopped being
    seen. A widening that produces misses is worse than the gap it closed.

    The marker case is the one with no backstop: redaction never touches a private-repo
    marker, so a miss here sends proprietary content to cloud with nothing to catch it.
    """
    def transcript(pad: int, tail: dict) -> list[dict]:
        return [
            {"role": "user", "content": "have a look"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {
                    "name": "write_file",
                    "arguments": '{"text": "%s"}' % ("x" * pad),
                }},
            ]},
            tail,
        ]

    secret = {"role": "user", "content": "key: AKIAIOSFODNN7EXAMPLE"}
    marker = {"role": "user", "content": "from the ACME-CONFIDENTIAL repo"}

    for pad in (100, 19_000, 25_000, 200_000):
        assert classify_request(transcript(pad, secret)).sensitivity is Sensitivity.SECRET, (
            f"a {pad}-char tool call hid a secret in a later message"
        )
        assert classify_request(
            transcript(pad, marker), ["ACME-CONFIDENTIAL"]
        ).sensitivity is Sensitivity.PRIVATE_REPO, (
            f"a {pad}-char tool call hid a private-repo marker — nothing else catches it"
        )

    # Non-vacuity in the other direction: arguments really are read. Every assertion
    # above would hold just as well if arguments were never scanned at all, so pin a
    # secret that exists ONLY in a tool call's arguments.
    args_only = [
        {"role": "user", "content": "ship it"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {
                "name": "deploy", "arguments": '{"token": "AKIAIOSFODNN7EXAMPLE"}'}},
        ]},
    ]
    assert classify_request(args_only).sensitivity is Sensitivity.SECRET

    # And benign arguments still classify NONE.
    assert classify_request(transcript(100, {"role": "user", "content": "ok"})).sensitivity \
        is Sensitivity.NONE


def test_tool_call_arguments_get_the_whole_remaining_window(monkeypatch):
    """The joining newline is not charged to the arguments' share of the window.

    `remaining` names how many characters are left for arguments, but slicing the joined
    string spent one of them on the separator, so an argument that exactly filled the
    remainder lost its last character. A 20-char AWS key landing on that boundary
    classified NONE and routed to cloud — the same silent widening this bound exists to
    stop.
    """
    key = "AKIA" + "Q" * 16
    assert len(key) == 20
    msgs = [{"role": "assistant", "content": "x" * 80, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "f", "arguments": key}},
    ]}]

    # Record the actual regex inputs, not just the verdict. The content and arguments are
    # classified separately so the synthetic separator neither consumes payload budget
    # nor pushes the work past the hard bound.
    import agentgate.sensitivity as sensitivity_mod

    original_classify = sensitivity_mod.classify
    scanned: list[str] = []

    def recording_classify(text, markers=()):
        scanned.append(text)
        return original_classify(text, markers)

    monkeypatch.setattr(sensitivity_mod, "classify", recording_classify)

    # 80 characters of content plus a 20-character argument is exactly the window.
    assert classify_request(msgs, (), max_chars=100).sensitivity is Sensitivity.SECRET
    assert [len(text) for text in scanned] == [80, 20]
    assert sum(map(len, scanned)) == 100

    # One character less of budget genuinely truncates it — the bound still bounds.
    scanned.clear()
    assert classify_request(msgs, (), max_chars=99).sensitivity is Sensitivity.NONE
    assert [len(text) for text in scanned] == [80, 19]
    assert sum(map(len, scanned)) == 99
