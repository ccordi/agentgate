# agentgate

[![CI](https://github.com/ccordi/agentgate/actions/workflows/ci.yml/badge.svg)](https://github.com/ccordi/agentgate/actions/workflows/ci.yml)

A transparent reverse-proxy **safety gateway** for LLM agents. Point an agent's
OpenAI or Anthropic-compatible `base_url` at it to enforce that every model call is
inspected and governed at one choke point:

- **Prompt-injection / tool-output scanning** of the untrusted channel (tool and
  retrieval output), so an injected instruction is caught before it reaches the model.
  Default scanner is a fine-tuned DeBERTa classifier; an optional local-LLM guard reads
  the whole untrusted item for higher recall.
- **Sensitivity-aware routing** — content the classifier marks sensitive is routed to an
  on-device model, so it never leaves the host.
- **Outbound secret redaction** on cloud egress.
- **Egress policy (PDP/PEP)** — a decision endpoint the agent's gated egress tool consults
  before any outbound request, so policy is enforced off the model wire.
- **Queryable audit log** of every prompt / tool-call / response (metadata always;
  content sampled and redacted-at-rest).

Responses stream straight through, so the gateway adds little latency.

This is a public extract of a larger private project. The write-up explains the design and
what it can and can't guarantee: **[ccordi.github.io/agentgate](https://ccordi.github.io/agentgate/)**.

> The eval corpus is full of secret-shaped strings (`sk-…`, `AKIA…`, base64 blobs) **on
> purpose** — synthetic fixtures for the redaction tests, never real credentials. One file
> is different: `eval/redteam/corpus/fp_capture.frozen.jsonl` is real benign traffic captured
> from single-user use (agent prompts + public web/tool content), kept real so the
> false-positive rate is measured against a realistic distribution; personal identifiers in
> it (handles, emails) have been anonymized.

## Quickstart (dev)

```bash
uv sync                   # install dependencies
uv run agentgate          # serves on http://127.0.0.1:4100
```

Point your agent or client at `http://127.0.0.1:4100` as its OpenAI/Anthropic `base_url`.

## Using it with an agent or coding harness

agentgate is OpenAI/Anthropic compatible, so you integrate it by pointing your tool's
model base URL at the gateway. The two modes differ only in how much of the gateway
they exercise.

### 1. Behind an agent framework (e.g. [OpenClaw](https://github.com/openclaw/openclaw))

Set the agent's model-provider base URL to the gateway. Every model call is then scanned
(untrusted-channel injection), routed (sensitive content stays local), redacted (cloud
egress), and audited.

- In your provider config, set the base URL to `http://127.0.0.1:4100`.
- Optional: use the tagged route `http://127.0.0.1:4100/a/<agent_id>` so the gateway
  attributes each agent's traffic (per-agent audit, plus the router's agent-pin rules).

### 2. Inside a coding harness (e.g. [Continue](https://continue.dev))

Same model-wire integration — set your model's `apiBase` to `http://127.0.0.1:4100` —
plus the outbound-egress layer (the PDP/PEP tier):

- Run the gated egress tool as the harness's **only** network path:

  ```bash
  uv run --extra egress-mcp python -m agentgate.pep.mcp_server
  ```

- Register it as an MCP tool, and exclude the harness's built-in fetch and shell
  `curl`/`wget` in its permissions config. Every outbound HTTP request then consults the
  policy endpoint (`POST /a/egress/decision`) before it runs, and is allowed or denied
  by destination allowlist + payload sensitivity.

For concrete config examples (`openclaw.json`, Continue's `config.yaml` / `permissions.yaml`)
and the relevant `AGENTGATE_*` env vars, see the **[integration guide](docs/integration.md)**.

## Configuration

Configured via environment variables (prefix `AGENTGATE_`) or a `.env` file; nested
settings use `__` (e.g. `AGENTGATE_ROUTING__ENABLED=false`). The knobs you're most
likely to touch:

- `AGENTGATE_DEFAULT_PROVIDER` — upstream to forward to (`gemini`, `openai`, `ollama`, `local`).
- `AGENTGATE_GUARD_BACKEND` — injection-guard backend: `deberta` (default; needs the
  `guard` extra and a pulled model — launch with `uv run --extra guard agentgate`) or
  `heuristic` (regex, always available). Falls back to `heuristic` if the model isn't present.
- `AGENTGATE_REDACTION_ENABLED` — outbound secret redaction (default on).

**Choosing a model.** For cloud providers the model is whatever the client sends in each
request — the gateway passes it through. The local route is different: it runs a fixed
on-device model (sensitivity-aware routing sends sensitive content here), and local servers
may require an exact model name — so set the model with `AGENTGATE_LOCAL_MODEL_OVERRIDE`.

The full set, with inline docs, lives in [`src/agentgate/config.py`](src/agentgate/config.py).

## Layout

- [`src/agentgate/`](src/agentgate) — the gateway (proxy, scanning, routing, redaction, egress PDP, audit)
- [`eval/redteam/`](eval/redteam) — the adversarial-evaluation harness and corpus
- [`bench/`](bench) — a canned-SSE mock upstream and latency benchmark for measuring gateway overhead
- [`tests/`](tests) — the test suite
