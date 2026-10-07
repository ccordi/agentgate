# agentgate

agentgate is an example LLM gateway. It is a reverse proxy between your agent application and an OpenAI Chat Completions endpoint. Point your application's `base_url` at agentgate to apply controls and monitoring to your agent application.

## Features

- detection of prompt injections and sensitive content, to reject, redact or keep on a local model
- an HTTP tool with destination and sensitivity controls
- limits on concurrent requests from each client
- a queryable audit log

The gateway has three prompt-injection scanners. The heuristic scanner matches known attack phrases; it is fast but misses many attacks. The classifier and the LLM judge catch more but add latency.

## Demo

No provider API keys or model downloads are needed.

From the repository directory, with `uv` and `curl` installed:

```bash
bash scripts/demo.sh
```

The demo starts a mock model server and a separate gateway with a temporary
database. It shows:

- A normal request completing.
- Prompt injection in tool output blocked with HTTP 400 (`injection_blocked`).
- A policy check denying a proposed HTTP request carrying a fake AWS key to a
  destination outside the allowlist.
- Audit records for those requests.

## Docs

- [Getting started](docs/getting-started.md) — Run the gateway with a mock model server or connect your own model.
- [Docker](docs/docker.md) — Run the gateway in a container.
- [OpenCode](docs/opencode.md) — Connect OpenCode with optional HTTP tool checks.
- [Configuration](docs/configuration.md) — Set providers, scanners, credentials and limits.
- [Audit log](docs/audit.md) — Inspect recorded requests, cost estimates and saved content.
- [Write-up](docs/index.md) — Why the gateway was built, how it works and what the tests showed.
- [Threat model](docs/threat-model.md) — Protections, assumptions and limitations.
- [Results](docs/results.md) — Scanner accuracy and latency measurements.
- [Concurrency](docs/scale-design.md) — Admission control, client keys and what was measured under load.
- [LiteLLM](docs/litellm-plugin.md) — Use the injection scanners as a LiteLLM guardrail.
