"""The injection guard as a LiteLLM guardrail (`litellm-plugin` extra).

One class, two hooks, both scanning a request's messages through `guards.scan`
— the project's single scan entry point — so the plugin inherits the gateway's
channel-scoped extraction and verdict semantics rather than duplicating them.
`async_pre_call_hook` (config `mode: pre_call`) blocks: a hard verdict raises
400 with the gateway's wire convention (`error.type = "injection_blocked"`);
everything else passes through untouched. `async_logging_hook` (config
`mode: logging_only`) observes: it logs the verdict from LiteLLM's logging path
and never blocks or raises — the measurement posture before turning blocking on.

`ProxyException` is a private import path: it is the type LiteLLM's exception
handler serializes verbatim, so it is the only way a guardrail controls the wire
envelope's `error.type`. The exact version pin is what makes that acceptable.

Deliberately NOT part of the plugin: audit rows, metrics, per-key backend
overrides, spend/kill, admission. Those are gateway features; what their
absence costs inside LiteLLM is documented in docs/litellm-plugin.md §Scope.

LiteLLM proxy config (guardrail runs on every request):

    guardrails:
      - guardrail_name: agentgate-injection-guard
        litellm_params:
          guardrail: agentgate.integrations.litellm_guard.AgentgateGuard
          mode: pre_call          # or logging_only: observe-only, log but never block
          default_on: true
          backend: deberta        # or heuristic / llm / combined

The `deberta` backend needs the `guard` extra and model files
(`AGENTGATE_GUARD_MODEL_DIR`); when they are missing the guardrail raises at
construction and the proxy fails to boot — the gateway's own posture (a
configured scanner that cannot run is refused at startup, never swapped for a
weaker one behind a log line).
"""

from __future__ import annotations

import logging
from typing import Any

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import ProxyException  # private path; safe under the exact pin

from agentgate import guards

log = logging.getLogger("agentgate.integrations.litellm")


def _resolve_backend(requested: str) -> str:
    """The backend the hook will run — validated, and its model warmed, or no guardrail.

    No availability fallback: a classifier model that will not load raises here, as at the
    gateway's own startup (`app.lifespan`).
    """
    if requested not in guards.BACKENDS:
        raise ValueError(
            f"unknown guard backend {requested!r}; expected one of {sorted(guards.BACKENDS)}"
        )
    if requested in ("deberta", "combined"):
        try:
            from agentgate.guards import deberta

            deberta.warmup()
        except Exception as exc:  # noqa: BLE001 — any load failure refuses the guardrail
            log.critical(
                "DEBERTA UNAVAILABLE (%s) — refusing to arm the %s guardrail",
                exc, requested,
            )
            raise RuntimeError(
                f"guard backend {requested!r} needs the classifier's model and it could not "
                f"load ({exc}); install the `guard` extra and set "
                "AGENTGATE_GUARD_MODEL_DIR, or configure a backend that does not need it"
            ) from exc
    return requested


class AgentgateGuard(CustomGuardrail):
    """Injection scan for LiteLLM's proxy: blocking (`pre_call`) or observe-only
    (`logging_only`)."""

    def __init__(self, backend: str = "deberta", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.backend = _resolve_backend(backend)
        log.info("agentgate guardrail armed: backend=%s", self.backend)

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict,
        call_type: str,
    ) -> dict:
        messages = data.get("messages")
        if not messages:
            # Nothing to scan (embeddings, image calls, …). Same convention as the
            # gateway: an empty scan surface is a pass, not a block.
            return data
        try:
            verdict = await guards.scan(self.backend, messages)
        except Exception as exc:  # noqa: BLE001 — a control that cannot run must not pass
            # Fail closed, and say so in the gateway's own convention rather than
            # letting the proxy render an unhandled 500. Unreachable for the
            # in-process backends (warmed at init); real for `llm`, whose judge is
            # a network dependency that can be down at scan time.
            log.error("guard scan failed (%s) — refusing the request", exc)
            raise ProxyException(
                message="request refused: the prompt-injection guard could not run",
                type="guard_unavailable",
                param=None,
                code=503,
            ) from exc
        if verdict.hard:
            log.warning(
                "injection blocked by LiteLLM guardrail: backend=%s score=%.4f reasons=%d",
                self.backend, verdict.score, len(verdict.reasons),
            )
            raise ProxyException(
                message="request blocked: prompt-injection guard verdict "
                        f"(score {verdict.score:.4f})",
                type="injection_blocked",
                param=None,
                code=400,
            )
        return data

    async def async_logging_hook(
        self,
        kwargs: dict,
        result: Any,
        call_type: str,
    ) -> tuple[dict, Any]:
        """Observe-only scan (config `mode: logging_only`): log the verdict, never block.

        LiteLLM's success-logging path calls this with the call's
        `model_call_details` as `kwargs` and expects `(kwargs, result)` back.
        The response has already been sent by the time it runs, so the only
        correct failure behavior is to log and pass — including when the scan
        itself fails.
        """
        messages = kwargs.get("messages")
        if not messages:
            return kwargs, result
        try:
            verdict = await guards.scan(self.backend, messages)
        except Exception as exc:  # noqa: BLE001 — observe mode must never break the request
            log.error("guard scan failed in observe mode (%s) — passing", exc)
            return kwargs, result
        if verdict.hard or verdict.flagged:
            # Mirror the block-log line's fields so the two postures grep the same.
            (log.warning if verdict.hard else log.info)(
                "injection observed by LiteLLM guardrail (no block): backend=%s "
                "score=%.4f verdict=%s reasons=%d",
                self.backend, verdict.score,
                "hard" if verdict.hard else "soft", len(verdict.reasons),
            )
        return kwargs, result
