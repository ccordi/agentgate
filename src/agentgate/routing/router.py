"""Rules-table router — evaluates the declarative routing config.

Pure decision logic: given the sensitivity class, agent id, and failure flags, return which
logical target (local vs cloud) to use and which rule fired. Provider resolution lives in
``providers.py``; this module has no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentgate.config import RoutingConfig, RoutingRule


@dataclass
class RouteContext:
    sensitivity: str = "none"
    agent_id: str | None = None
    cloud_unavailable: bool = False
    over_spend_cap: bool = False

    def flags(self) -> set[str]:
        f = set()
        if self.cloud_unavailable:
            f.add("cloud_unavailable")
        if self.over_spend_cap:
            f.add("over_spend_cap")
        return f


@dataclass
class RouteDecision:
    provider: str   # config provider name
    is_local: bool
    rule: str       # name of the rule that fired
    action: str     # "route_local" | "prefer_cloud"


def _matches(rule: RoutingRule, ctx: RouteContext) -> bool:
    """A rule matches when every specified condition holds (unspecified = ignored)."""
    if rule.sensitivity_in is not None and ctx.sensitivity not in rule.sensitivity_in:
        return False
    if rule.agent_in is not None and (ctx.agent_id is None or ctx.agent_id not in rule.agent_in):
        return False
    if rule.any_flags is not None and not (set(rule.any_flags) & ctx.flags()):
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
