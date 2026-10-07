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


def test_sensitive_routes_local_whatever_the_agent():
    for agent_id in (None, "other"):
        d = decide(RouteContext(sensitivity="secret", agent_id=agent_id), CFG)
        assert d.is_local and d.rule == "sensitive-stays-local", agent_id


def test_default_is_cloud():
    d = decide(RouteContext(sensitivity="none", agent_id="other"), CFG)
    assert not d.is_local and d.rule == "default"


def test_resolve_default_cloud_branch_stays_local_by_default():
    """Pins that with shipped settings the `default` rule's prefer_cloud branch resolves to
    the `local` provider: the decision's is_local names the branch, the provider's is_local
    says where the request goes."""
    s = Settings()
    d = decide(RouteContext(sensitivity="none"), s.routing)
    assert d.rule == "default" and not d.is_local
    provider = resolve(s, d)
    assert provider.name == "local" and provider.is_local


def test_resolve_named_cloud_provider_reaches_cloud():
    """Pins that naming a cloud provider in routing.default_cloud sends the prefer_cloud
    branch there while the sensitive branch still resolves local."""
    s = Settings(routing={"default_cloud": "gemini"})
    local = resolve(s, decide(RouteContext(sensitivity="pii"), s.routing))
    cloud = resolve(s, decide(RouteContext(sensitivity="none"), s.routing))
    assert local.is_local and local.name == "local"
    assert not cloud.is_local and cloud.name == "gemini"


def test_routing_off_default_provider_is_local_until_named():
    """Pins that with routing disabled the shipped default_provider is `local` and an
    explicitly named cloud provider is honored."""
    assert Settings().provider().name == "local"
    assert Settings(default_provider="gemini").provider().name == "gemini"


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

    On an assistant tool-call message `content` is None, so coerce_content returns "",
    and read from `content` alone the message would classify as empty — the secret would
    route to cloud instead of the local model, and assistant tool-call history replays
    every turn.
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


def test_legacy_function_call_arguments_classify_and_route_like_tool_calls():
    """The pre-`tool_calls` spelling has to reach the classifier too.

    `function_call` is one object where `tool_calls` is a list, so an extraction point
    that reads only the list would classify a credential an agent passed to a tool in the
    legacy shape as `none` and send it down the default cloud route, while the identical
    bytes spelled `tool_calls` classify `secret` and stay local. Not malformed
    input: the request parser already accepts the matching legacy `functions[]` catalog,
    and OpenAI-compatible clients still emit this. Asserted through the real route
    decision, because the miss matters where it changes the destination.
    """
    import json
    import uuid

    from agentgate.pipeline import ChatCall, classify_and_route

    function = {"name": "deploy", "arguments": '{"token": "sk-abcdefghijklmnopqrstuvwx"}'}
    settings = Settings(_env_file=None)
    for spelling in ("function_call", "tool_calls"):
        msgs = [{"role": "user", "content": "ship it"},
                {"role": "assistant", "content": None, spelling: (
                    function if spelling == "function_call"
                    else [{"id": "c1", "type": "function", "function": function}])}]
        body = json.dumps({"model": "m", "messages": msgs}).encode()
        call = ChatCall(request_id=uuid.uuid4(), t0=0.0, body=body, headers={}, key_id="k",
                        agent_id=None, model_requested="m", messages=msgs, payload=None)
        classify_and_route(call, settings)
        assert call.sensitivity_class == "secret", spelling
        assert call.provider is not None and call.provider.is_local, spelling


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
    # The legacy spelling gets the same tolerance, and joins after the modern one.
    assert coerce_tool_call_args({"function_call": "not-an-object"}) == ""
    assert coerce_tool_call_args({"function_call": {"arguments": 5}}) == ""
    assert coerce_tool_call_args({"function_call": {"name": "f"}}) == ""
    assert coerce_tool_call_args({"function_call": {"arguments": "a"}}) == "a"
    assert coerce_tool_call_args(
        {"tool_calls": [{"function": {"arguments": "a"}}],
         "function_call": {"arguments": "b"}}
    ) == "a\nb"


def test_tool_call_args_never_displace_content_from_the_window():
    """A big benign tool call must not push a later secret out of the classify window.

    Tool-call arguments are read after all message content, so a large benign tool call
    cannot push a later message's secret out of the 20,000-character window.

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
    string would spend one of them on the separator, so an argument that exactly fills
    the remainder would lose its last character. A 20-character AWS key on that boundary
    would classify NONE and route to cloud.
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


# ---- classification window: the truncation caveat ---------------

def test_classify_truncation_is_recorded_past_the_window_and_not_at_it():
    """`truncated_at` is set exactly when input fell outside the window.

    Both directions, at the small bound the boundary pins above use and at the real
    default: a body that fills the window exactly is fully read and carries nothing;
    one character more is head-truncated and says so. The verdict itself is unchanged
    either way — this is visibility, not a change in what gets classified.
    """
    at = [{"role": "user", "content": "x" * 100}]
    over = [{"role": "user", "content": "x" * 101}]
    assert classify_request(at, (), max_chars=100).truncated_at is None
    assert classify_request(over, (), max_chars=100).truncated_at == 100

    # 10,000 + one join separator + 9,999 = 20,000 exactly; one more overruns.
    default_at = [{"role": "user", "content": "a" * 10_000},
                  {"role": "user", "content": "b" * 9_999}]
    default_over = [{"role": "user", "content": "a" * 10_000},
                    {"role": "user", "content": "b" * 10_000}]
    assert classify_request(default_at).truncated_at is None
    assert classify_request(default_over).truncated_at == 20_000

    # A secret in the head is still classified; the caveat rides alongside the verdict.
    head = [{"role": "user", "content": "AKIAIOSFODNN7EXAMPLE " + "x" * 100}]
    r = classify_request(head, (), max_chars=50)
    assert r.sensitivity is Sensitivity.SECRET and r.truncated_at == 50
    # A secret past the window is still missed — that is the bound — but marked.
    tail = [{"role": "user", "content": "x" * 50 + "AKIAIOSFODNN7EXAMPLE"}]
    r = classify_request(tail, (), max_chars=50)
    assert r.sensitivity is Sensitivity.NONE and r.truncated_at == 50


def test_classify_truncation_covers_arguments_past_the_remainder_and_a_spent_window():
    """The two argument sub-cases.

    Arguments longer than what the window has left, and arguments present when content
    had already spent the window — the latter a presence check on the two tool-call
    spellings, not a coercion, so nothing is assembled only to be discarded. Arguments
    that exactly fit the remainder are fully read and carry nothing (echoing the
    exactness pin above); an empty `tool_calls` list is not unread arguments.
    """
    def call(args: str) -> dict:
        return {"id": "c1", "type": "function", "function": {"name": "f", "arguments": args}}

    fits = [{"role": "assistant", "content": "x" * 80, "tool_calls": [call("y" * 20)]}]
    over = [{"role": "assistant", "content": "x" * 80, "tool_calls": [call("y" * 21)]}]
    assert classify_request(fits, (), max_chars=100).truncated_at is None
    assert classify_request(over, (), max_chars=100).truncated_at == 100

    # 99 chars of content + the join separator before an empty assistant content = 100:
    # the window is spent to the character, and the arguments on the next message are
    # never read.
    spent = [{"role": "user", "content": "x" * 99},
             {"role": "assistant", "content": None, "tool_calls": [call("y")]}]
    assert classify_request(spent, (), max_chars=100).truncated_at == 100
    spent_no_args = [{"role": "user", "content": "x" * 99},
                     {"role": "assistant", "content": None}]
    assert classify_request(spent_no_args, (), max_chars=100).truncated_at is None
    spent_empty_calls = [{"role": "user", "content": "x" * 99},
                         {"role": "assistant", "content": None, "tool_calls": []}]
    assert classify_request(spent_empty_calls, (), max_chars=100).truncated_at is None
    # Legacy arguments are unread arguments too, or a spent window would report a clean
    # read of a message the classifier never looked at.
    spent_legacy = [{"role": "user", "content": "x" * 99},
                    {"role": "assistant", "content": None,
                     "function_call": {"name": "f", "arguments": "y"}}]
    assert classify_request(spent_legacy, (), max_chars=100).truncated_at == 100
