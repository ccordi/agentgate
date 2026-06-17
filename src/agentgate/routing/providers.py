"""Resolve a routing decision to a concrete upstream Provider.

Thin layer over the provider registry in ``config.Settings`` so the router stays pure
(no config/I/O). Keeps the local↔cloud indirection in one place.
"""

from __future__ import annotations

from agentgate.config import Provider, Settings

from .router import RouteDecision


def resolve(settings: Settings, decision: RouteDecision) -> Provider:
    """Map a RouteDecision's provider name to the registered Provider."""
    return settings.provider(decision.provider)
