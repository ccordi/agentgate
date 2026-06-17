---
title: agentgate
---

# Integration guide

agentgate is an OpenAI/Anthropic-compatible reverse proxy. You integrate it by pointing
your tool's **model base URL** at the gateway (`http://127.0.0.1:4100` by default) — no
change to the model, and usually a one-line change to the tool.

Two setups are covered here:

1. **Agent framework** (example: [OpenClaw](https://github.com/openclaw/openclaw)) — route
   the agent's model calls through the gateway.
2. **Coding harness** (example: [Continue](https://continue.dev)) — same model-wire
   integration, plus the outbound-egress layer (the PDP/PEP).

Both assume the gateway is running:

```bash
uv sync
uv run --extra guard agentgate      # serves on http://127.0.0.1:4100
```

(`--extra guard` enables the DeBERTa scanner; plain `uv run agentgate` falls back to the
heuristic guard.)

## agentgate env vars you'll touch

Set these in the gateway's environment or a `.env` file (prefix `AGENTGATE_`; nested keys
use `__`). Full list with inline docs in [`src/agentgate/config.py`](https://github.com/ccordi/agentgate/blob/main/src/agentgate/config.py).

| Variable | Purpose |
|---|---|
| `AGENTGATE_DEFAULT_PROVIDER` | Upstream the gateway forwards to: `gemini`, `openai`, `ollama`, `local`. |
| `AGENTGATE_GUARD_BACKEND` | Injection-guard backend: `deberta` (default; needs `--extra guard`) or `heuristic`. |
| `AGENTGATE_ROUTING__ENABLED` | Sensitivity-aware routing on/off (default on). |
| `AGENTGATE_REDACTION_ENABLED` | Outbound secret/PII redaction on cloud egress (default on). |
| `AGENTGATE_LOCAL_MODEL_OVERRIDE` | Model name forced on the local route (sensitive content). |
| `AGENTGATE_EGRESS__ALLOWLIST` | JSON array of destinations the egress PDP treats as safe, e.g. `["api.github.com"]`. |
| `AGENTGATE_LOCAL_API_KEY` | Bearer the egress PEP presents to the PDP (see §2). |

Auth to the real upstream is **passed through** from the inbound request — the gateway
doesn't store provider keys. Put your real provider key in the *client's* config (below);
the gateway forwards it to whatever `AGENTGATE_DEFAULT_PROVIDER` points at.

---

## 1. Agent framework — OpenClaw

**File:** `~/.openclaw/openclaw.json`. Point a provider's `baseUrl` at the gateway. Every
model call is then scanned, routed, redacted, and audited.

```json5
{
  models: {
    providers: {
      // Keep the provider id matching the real upstream the gateway forwards to
      // (here: Gemini, so set AGENTGATE_DEFAULT_PROVIDER=gemini). The key you put on
      // this provider is passed through the gateway to that upstream.
      google: {
        baseUrl: "http://127.0.0.1:4100/a/my-agent",   // tagged route → per-agent audit + router pins
        models: [
          { id: "gemini-3.1-flash-lite", name: "Gemini 3.1 Flash Lite",
            contextWindow: 50000, maxTokens: 8192 }
        ]
      }
    }
  },
  agents: {
    defaults: { model: { primary: "google/gemini-3.1-flash-lite" } }
  }
}
```

- **Base URL is the integration.** `http://127.0.0.1:4100` routes all calls through the
  gateway; the `/a/<agent_id>` suffix (optional) tags the traffic so the audit log and the
  router's agent-pin rules can attribute it. Setting a custom `baseUrl` is also OpenClaw's
  network-trust decision for that origin (loopback is allowed by default).
- **CLI equivalent:** `openclaw config set models.providers.google.baseUrl "http://127.0.0.1:4100/a/my-agent"`.
- On the gateway, set `AGENTGATE_DEFAULT_PROVIDER` to the matching upstream (`gemini`,
  `openai`, …) so it knows where to forward.

---

## 2. Coding harness — Continue

Two parts: the model wire (same as above) and the egress layer.

### a) Model wire — `~/.continue/config.yaml`

```yaml
name: agentgate-local
version: 1.0.0
schema: v1

models:
  - name: gateway
    provider: openai
    model: gemma-4-26b-a4b-it          # whatever model your upstream expects
    apiBase: http://127.0.0.1:4100/v1  # ← point at the gateway
    apiKey: sk-...                     # passed through to the upstream (any non-empty token for a local server)
    roles: [chat, edit, apply, autocomplete]
    contextLength: 32000
```

### b) Egress layer — gate every outbound request

The gateway sits on the *model* wire; it can't see HTTP the agent makes through its own
tools. To bring those under policy, run the **gated egress MCP server** as the harness's
**only** network-capable tool, and close the other network paths.

Register the server in `~/.continue/config.yaml`:

```yaml
mcpServers:
  - name: egress-gateway
    command: uv
    args: [run, "--extra", egress-mcp, python, "-m", agentgate.pep.mcp_server]
    cwd: /path/to/agentgate          # where you cloned this repo
    env:
      AGENTGATE_LOCAL_API_KEY: ${AGENTGATE_LOCAL_API_KEY}   # presented to the PDP as bearer
```

Exclude the built-in network paths in `~/.continue/permissions.yaml` (or at launch with
`cn --exclude Fetch --exclude "Bash(curl*)" …`):

```yaml
allow: []
ask: []
exclude:
  - Bash(curl*)
  - Bash(wget*)
  - Bash(fetch*)
  - Fetch
```

With those excluded, the agent's only way out is the `safe_http_request` tool, which
consults the gateway's PDP (`POST http://127.0.0.1:4100/a/egress/decision`) before each
call. The PDP allows or denies by **destination allowlist** (`AGENTGATE_EGRESS__ALLOWLIST`)
and **payload sensitivity**, and the PEP fails closed if the PDP is unreachable.

> **This is cooperative enforcement, not a sandbox.** The `Bash(curl*)` rules match the
> command's first token, so a shell wrapper (`sh -c 'curl …'`) starts with `sh` and slips
> past — a known evasion surface. The gate holds for an agent that stays on its normal
> tool path (including one misled by an injection); it does not contain an agent
> determined to reach an ungated path.

---

← Back to the [agentgate overview](index.md).

See also the [README](https://github.com/ccordi/agentgate/blob/main/README.md) quickstart and
[`src/agentgate/config.py`](https://github.com/ccordi/agentgate/blob/main/src/agentgate/config.py)
for the full settings reference.
