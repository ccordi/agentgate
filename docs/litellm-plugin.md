# Injection scanning in LiteLLM

agentgate's injection scanners can run as a guardrail inside LiteLLM, an open-source
proxy that puts one OpenAI-style API in front of many model providers, without the rest
of the gateway. The guardrail can block a request before LiteLLM calls a model,
or log the result and let the request continue.

## Setup

Install from a clone of this repository:

```bash
uv sync --extra guard --extra litellm-plugin
```

Add the guardrail to your LiteLLM configuration:

```yaml
guardrails:
  - guardrail_name: agentgate-injection-guard
    litellm_params:
      guardrail: agentgate.integrations.litellm_guard.AgentgateGuard
      mode: pre_call           # or logging_only
      default_on: true
      backend: deberta         # deberta | heuristic | llm | combined
```

Start the proxy from the clone so it can import the plugin:

```bash
uv run --extra guard --extra litellm-plugin litellm --config your-config.yaml
```

The classifier needs the optional `guard` dependencies and its
[local model files](configuration.md#set-up-the-classifier-on-the-host); set
`AGENTGATE_GUARD_MODEL_DIR` to the model directory. If the files cannot be
loaded, the LiteLLM proxy does not start. This applies to `deberta` and `combined`: the
plugin does not fall back to a weaker scanner.

In `pre_call` mode, a request the scanner blocks gets HTTP 400 with
`error.type = "injection_blocked"`, and a scan that fails gets HTTP 503 with
`error.type = "guard_unavailable"`. Every other request goes through.

In `logging_only` mode, nothing is blocked. The plugin logs flagged results with the
scanner, score, verdict and number of reasons, without the reason text, and logs scan
errors.

Requests with no messages pass without scanning.

## What the plugin scans

The `backend` setting selects the scanner; see
[scanner backends](configuration.md#scanner-backends) for the choices. Each scanner
checks the same messages as in the gateway, as shown in the
[threat model](threat-model.md#prompt-injection-in-tool-results).

For `llm` and `combined`, set `AGENTGATE_JUDGE_BASE_URL`, `AGENTGATE_JUDGE_MODEL` and
`AGENTGATE_JUDGE_API_KEY` for your local model server. The URL must use `localhost`,
`127.0.0.1` or `::1`.

## LiteLLM version

The optional `litellm-plugin` dependencies pin LiteLLM to `1.95.1`. The plugin relies on
LiteLLM internals that can change between releases, so other versions may need changes.

## What was tested

The plugin ran inside a LiteLLM 1.95.1 proxy with the classifier, in front of a
mock model server that always returns the same response. Ordinary requests and garak
injection attacks were sent through it. None of the 10 ordinary requests was blocked. The
classifier blocked 256 of 256 attacks that tell the model outright to ignore its
instructions, and 306 of 512 injections hidden in documents. The
[results page](results.md#scanner-accuracy) measures the classifier's accuracy.

The plugin's [unit tests](https://github.com/ccordi/agentgate/blob/main/tests/test_litellm_plugin.py) need no model files and are
skipped when LiteLLM is not installed.

## Scope

The plugin scans request messages only. It has none of the gateway's other controls:

- The audit log and metrics.
- Client keys, spending caps, kill switches and limits on concurrent requests.
- Sensitive-content classification, redaction and routing, and the destination and
  payload checks of the HTTP tool.
