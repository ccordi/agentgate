"""Benchmark-harness unit tests — the pure stats math + the bench-only mock provider.

The end-to-end load run (k6 + live processes) isn't a unit test; these lock the pieces that
can silently corrupt the artifact's numbers.
"""

from __future__ import annotations

from agentgate.config import Settings
from agentgate.routing import providers as routing_providers
from agentgate.routing import router as routing_router
from bench import stats


def _resolve_like_app(settings: Settings, *, sensitivity: str = "none", agent_id=None):
    """Replicate app.py's provider-selection branch so the test exercises the SAME path
    the live gateway takes (router when enabled, else settings.provider())."""
    if settings.routing.enabled:
        decision = routing_router.decide(
            routing_router.RouteContext(sensitivity=sensitivity, agent_id=agent_id),
            settings.routing,
        )
        return routing_providers.resolve(settings, decision)
    return settings.provider()


def test_percentile_linear_interpolation():
    vals = list(range(1, 11))  # 1..10
    assert stats.percentile(vals, 50) == 5.5
    assert round(stats.percentile(vals, 90), 2) == 9.1
    assert round(stats.percentile(vals, 99), 2) == 9.91
    assert stats.percentile([42], 99) == 42.0
    assert stats.percentile([], 99) == 0.0


def test_summarize_skips_none_and_rounds():
    s = stats.summarize([1.0, 2.0, None, 3.0, 4.0])
    assert s["count"] == 4
    assert s["min"] == 1.0 and s["max"] == 4.0
    assert s["mean"] == 2.5
    assert stats.summarize([])["count"] == 0
    assert stats.summarize([None, None])["count"] == 0


def test_gateway_overhead_clamps_negative():
    # total − upstream, paired; jitter that goes negative clamps to 0.
    oh = stats.gateway_overhead_ms([10.0, 5.0, None], [3.0, 6.0, 1.0])
    assert oh == [7.0, 0.0]  # third pair skipped (total None)


def test_mock_provider_is_bench_only_and_cloud_typed():
    s = Settings()
    mock = s.provider("mock")
    assert mock.base_url == "http://127.0.0.1:4200"
    # is_local=False so only the USD cap applies under load (never the local req-count guard).
    assert mock.is_local is False
    # Selected only via env; not the default.
    assert s.default_provider != "mock"
    assert Settings(default_provider="mock").provider().name == "mock"


def test_bench_gateway_config_actually_routes_benign_traffic_to_mock():
    """Regression: AGENTGATE_DEFAULT_PROVIDER=mock alone is NOT enough.

    The rules-table router is enabled by default, and when enabled it — not
    default_provider — chooses the upstream. A benign bench request then falls through
    to the `default` rule (prefer_cloud → default_cloud="gemini"), so the gateway
    forwards to the real Google endpoint instead of the mock. That silently routed the
    whole load benchmark at a live cloud API and made every gateway phase report
    fail% 100%. The bench must disable routing so default_provider=mock wins.
    """
    # The original (buggy) bench config: default_provider=mock, routing left at default.
    buggy = Settings(default_provider="mock")
    assert buggy.routing.enabled is True
    assert _resolve_like_app(buggy).name == "gemini"  # NOT the mock — this is the bug.

    # The fixed bench config: routing disabled (as bench/run.py now sets via
    # AGENTGATE_ROUTING__ENABLED=false) → benign traffic resolves to the mock.
    fixed = Settings(default_provider="mock", routing={"enabled": False})
    assert _resolve_like_app(fixed).name == "mock"
    assert _resolve_like_app(fixed).base_url == "http://127.0.0.1:4200"
