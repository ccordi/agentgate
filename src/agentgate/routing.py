"""Rules-table router — evaluates the declarative routing config.

Pure decision logic: given the sensitivity class and agent id, return which logical target
(local vs cloud) to use and which rule fired. Provider resolution is the one impure lookup
and is kept at the bottom, so the decision half stays testable without touching config.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentgate.config import Provider, RoutingConfig, RoutingRule, Settings


@dataclass
class RouteContext:
    sensitivity: str = "none"
    agent_id: str | None = None


@dataclass
class RouteDecision:
    provider: str   # config provider name
    # The branch the rule chose (route_local → True), not the resolved provider's
    # locality; the request path reads the provider's own is_local flag.
    is_local: bool
    rule: str       # name of the rule that fired
    action: str     # "route_local" | "prefer_cloud"


def _matches(rule: RoutingRule, ctx: RouteContext) -> bool:
    """A rule matches when every specified condition holds (unspecified = ignored)."""
    if rule.sensitivity_in is not None and ctx.sensitivity not in rule.sensitivity_in:
        return False
    if rule.agent_in is not None and (ctx.agent_id is None or ctx.agent_id not in rule.agent_in):
        return False
    return True


def decide(ctx: RouteContext, cfg: RoutingConfig) -> RouteDecision:
    """First matching rule wins. ``route_local`` → local provider; ``prefer_cloud`` → cloud."""
    for rule in cfg.rules:
        if _matches(rule, ctx):
            if rule.action == "route_local":
                return RouteDecision(cfg.default_local, True, rule.name, rule.action)
            return RouteDecision(cfg.default_cloud, False, rule.name, rule.action)
    # No rule matched (shouldn't happen — the default rule is unconditional) → safe default.
    return RouteDecision(cfg.default_cloud, False, "implicit-default", "prefer_cloud")


# --- provider resolution (the one impure lookup) ------------------------------------

def resolve(settings: Settings, decision: RouteDecision) -> Provider:
    """Map a RouteDecision's provider name to the registered Provider.

    Thin layer over the provider registry in ``config.Settings``, kept here so the
    local↔cloud indirection lives in one place.
    """
    return settings.provider(decision.provider)
