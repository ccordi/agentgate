---
title: agentgate
---

# Threat model

This page states what the gateway defends and what remains out of scope. It assumes a
**single-user deployment**: one operator, one host, and the gateway on
loopback. This models what the *gateway* defends, not the host OS or the agent harness.

## Assets

- **Operator secrets and PII** in content the agent can reach — config files, emails,
  repos, anything a tool call can pull into context.
- **Provider API keys in transit.** The gateway stores no provider keys; the inbound
  auth header passes through it on every cloud call.
- **The agent's ability to act** — its tool privileges (HTTP, shell, files). An
  injection doesn't steal a credential so much as borrow the agent that holds one.
- **Integrity of the operator's instructions** — the agent acting on what the operator
  asked, not on what a document told it.

## Adversary

**Capability: authors content the agent will read.** Web pages, emails, retrieved
documents, tool results, tool/MCP catalog descriptions. Nothing more is needed —
EchoLeak was exactly this capability, exercised zero-click.

Explicit **non**-capabilities:

- **Cannot modify the client or harness.** An agent running a maliciously rewritten
  harness is out of scope (last section) — the PEP is code the harness chooses to run.
- **Cannot intercept loopback traffic** between agent, gateway, and local model. A
  separate adversary already on the host is past what a gateway can do. Two notes on
  what this exclusion does *not* cover. First, the agent: it is on the host by
  construction, and loopback destinations are always allowed by the egress policy — so
  reaching the gateway's own ports is not something the egress policy prevents. The
  admin plane (`/admin/kill/*`) and the egress PDP therefore require **dedicated bearer
  tokens** (`AGENTGATE_ADMIN_TOKEN`, `AGENTGATE_PDP_TOKEN`) — the gateway refuses to
  start without them, because an unarmed kill switch is clearable *by* the agent it
  exists to halt. Second, the operator's browser: loopback excludes remote *sockets*,
  not remote *code*. A web page can fire cross-origin requests at 127.0.0.1, and DNS
  rebinding would make them same-origin — so the gateway also rejects any request whose
  `Host` header is not a loopback name. The proxy routes remain unauthenticated by
  design: a transparent proxy does not own the `Authorization` slot, so their inbound
  auth is the loopback bind itself — which is why the bind is **enforced at startup**
  (a non-loopback `AGENTGATE_HOST` refuses to serve, with deliberately no override
  flag), not assumed.
- **Is not the operator.** The local-LLM backend trusts the operator's turn by design and
  scans only tool output. The heuristic and DeBERTa backends also scan the newest user
  turn. [The essay](index.md) reports the local-LLM backend's operator-channel check and
  explains why its result is format-sensitive.

## Trust boundaries

1. **Operator turn vs. tool/retrieval channel, on the model wire.** By the time text
   reaches the model the two are indistinguishable; the channel it arrived on is the
   durable signal, and it's visible on the wire. Every backend scans the untrusted
   channel; the heuristic and DeBERTa backends also scan the newest user turn.
2. **Model wire vs. client-side tool execution.** The proxy sees everything the model
   reads and says; it cannot see what the agent *does* — tools execute in the harness,
   off the wire. So the egress decision (PDP, on the gateway) is split from its
   enforcement (PEP, inside the harness).
3. **The host boundary.** Content classified sensitive is routed to the on-device
   model and never leaves the host — which is why redaction is a cloud-route concern
   only.

## Controls → threats

Paths are relative to `src/agentgate/`.

| Threat | Example | Control | Where | Residual risk |
|---|---|---|---|---|
| Indirect injection in tool/retrieval output | Instruction buried in a fetched email or web page (EchoLeak-style) | Inbound scan of the untrusted channel — heuristic patterns, DeBERTa classifier (the default), opt-in local-LLM guard; a hard verdict is a 400 before the model ever sees the content | `guards/heuristic.py`, `guards/deberta.py`, `guards/local_llm.py`, dispatched from `pipeline.py` | Payloads scoring below the block threshold get through. DeBERTa scans at most 16 overlapping windows per item; the LLM backend is bounded by the local server's context limit and may reject or truncate oversized input. Content the model restates lands in assistant history, which the gateway trusts and does not re-scan |
| Injection via tool-catalog descriptions | A `tools[]` entry whose description says "ignore previous instructions and…" | Static inspection of every `tools[]` definition; the hard tier (instruction injection in a description field) blocks with a 400 | `tool_inspector.py` | The soft tier — suspicious names, overly-broad or empty descriptions — is record-only: logged and audited, never blocked, until false-positive data justifies more |
| Operator secrets/PII egressing to a cloud model | An API key in a config file the agent read, about to be forwarded upstream | Sensitivity classification routes sensitive content to the on-device model (zero egress); secrets/PII are redacted from anything that does go to cloud | `sensitivity.py`, `routing.py`, `redaction.py`, applied on cloud routes in `pipeline.py` | Regex + entropy detection over a bounded slice of the conversation — a secret with a novel shape, or one that falls outside the classified window in a long session, can slip the patterns |
| Exfiltration via agent *actions* | Agent POSTs a secret-bearing file to a non-allowlisted host | Egress PDP decides on destination allowlist × payload sensitivity; the PEP — the harness's only network tool — consults it before every request and fails closed if the PDP is unreachable | `egress/policy.py`, `egress/api.py` (PDP); `egress/pep.py`, `egress/mcp_server.py` (PEP) | Cooperative only: it gates the sanctioned path, and ungated paths — shellout above all — walk around it (next section) |
| Runaway spend loops | An agent stuck re-calling a cloud model | Per-key rolling-window USD cap on cloud routes; breaching it trips a **sticky** kill switch (halt, not throttle); local routes get a request-count cap | `limits/spend.py` | Loops under the cap run to completion, and spend is estimated from token counts, not billed truth; a request the client cuts off mid-stream records **zero** cost, because usage arrives in the stream's final event, so hanging up early accrues nothing against the cap; and the key is whatever the client says it is — this is accounting for one cooperative operator, not a tenancy boundary |
| Sensitive data in the gateway's own audit trail | A captured content sample containing user text | Samples are redacted **then** encrypted at rest, carry a TTL, and are swept; with no encryption key configured, capture is skipped entirely — fail-closed, never plaintext; local-route and sensitive requests are never content-captured | `audit/crypto.py`, `audit/models.py`; capture decision in `pipeline.py` | The metadata tier (no content) is retained indefinitely; whoever holds the host's key can read the samples; and the optional capture tap used to build the false-positive corpus writes raw text, by design, to a local file that is neither encrypted nor expired |
| Any control's dependency being down | The on-device guard model isn't running | Startup fallback to the always-available heuristic scanner | `app.py` (startup probe), `guards/__init__.py` (`resolve_backend`), dispatched from `pipeline.py` | Only the startup case is handled; a mid-session failure of the local model surfaces as a request error, not as an unscanned forward |
| The loopback assumption silently failing | `AGENTGATE_HOST=0.0.0.0` (the ordinary way a container ships), or a drive-by web page reaching 127.0.0.1 | Startup refuses a non-loopback bind (no override flag — the tokenless proxy has no credential to arm, so no config state makes a wide bind safe); mandatory bearer tokens on the admin plane and PDP; non-loopback `Host` headers rejected (kills DNS rebinding) | `config.py` (`validate_runtime_settings`), `app.py` (`LoopbackHostGuard`, `_check_admin_auth`), `egress/api.py` | A direct `uvicorn agentgate.app:app --host …` never consults settings and bypasses the bind check (the token checks still hold); blind cross-origin POSTs — unreadable responses, no credentials — can still burn local-model compute through the proxy |

## Out of scope / unmitigated

These are design boundaries, not claims of coverage:

- **A malicious or modified client/harness.** This is cooperative enforcement, not
  containment — a client that doesn't ask the PDP isn't governed by it.
- **Shell and subprocess network paths.** `sh -c 'curl …'` clears a first-token `curl`
  block; `python3` isn't gated at all; neither are `git push`, `npx`, `ssh`, or anything
  written to a shell profile to run later. Closing these means gating those surfaces
  too — or OS-level network sandboxing, a different problem.
- **OS-level containment.** No sandbox, no network-namespace isolation; the gate holds
  for an agent that asks.
- **Multi-tenant isolation.** One operator; caps and kill switches are per-key
  accounting, not tenant boundaries.
- **Attacks that steer behavior without a plantable instruction.** If nothing
  instruction-shaped ever arrives on the wire, there is nothing for a scanner to catch —
  the gap the essay names right next to its recall numbers.
- **Guaranteed audit delivery.** Audit writes are off the hot path and best-effort by
  design — the trail is for post-hoc characterization, not for non-repudiation.
- **Model-provider compromise.** The upstream provider is trusted with whatever the
  gateway sends it; routing and redaction bound *what that is*, not what the provider
  does with it.

## Evidence

The measurements behind these controls — indirect-injection recall, the false-positive
rate on the scanned channel, the operator-channel over-fire check, the latency tiers —
are stated in [the design essay](index.md) rather than repeated here. Reproduction
commands, corpus provenance, and the expected-miss taxonomy
live in `eval/redteam/README.md`.

---

← Back to the [agentgate overview](index.md).
