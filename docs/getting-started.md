# Getting started

Start the gateway with the built-in heuristic scanner and a mock model server that
returns a fixed response.

For container setup, see [Running with Docker](docker.md).

Run these commands from the directory where you cloned agentgate. The gateway uses
port 4100 and the mock model server uses port 4200.

## Running directly

You need [uv](https://docs.astral.sh/uv/getting-started/installation/) and `curl`.
uv installs the Python version and dependencies required by the project.

```bash
uv sync --frozen
uv run agentgate init # initialize .env file with demo settings
```

Start the mock model server in one terminal:

```bash
uv run python -m bench.mock_upstream
```

In a second terminal, from the same checkout, start the gateway:

```bash
uv run agentgate
```

Leave both running and [send a request](#send-a-request) from another terminal.

## Send a request

Check readiness:

```bash
curl --fail http://127.0.0.1:4100/readyz
```

Then send a streaming chat request:

```bash
curl --fail --no-buffer http://127.0.0.1:4100/v1/chat/completions \
  -H 'Authorization: Bearer getting-started' \
  -H 'Content-Type: application/json' \
  -d '{"model":"mock-model","stream":true,"messages":[{"role":"user","content":"Hello"}]}'
```

The stream spells out “Hello from the mock upstream.” and ends with `[DONE]`.
The mock model server ignores the client key; `getting-started` is a placeholder.

Inspect the recorded request:

```bash
uv run agentgate audit tail
```

The [audit guide](audit.md) explains the fields and other queries.

## Stop and clean up

When you're finished, press Ctrl-C in each server terminal. Settings remain in
`.env`, and audit records remain in `data/agentgate.db`.

## Connect a real model server

Use an existing server that implements OpenAI Chat Completions. In `.env`, change
`AGENTGATE_DEFAULT_PROVIDER` to `local` and add the provider definition and model
name below. Keep routing disabled for this setup with one model server.

```dotenv
AGENTGATE_DEFAULT_PROVIDER=local
AGENTGATE_ROUTING__ENABLED=false
AGENTGATE_PROVIDERS='{"local":{"name":"local","base_url":"http://127.0.0.1:8000","is_local":true}}'
AGENTGATE_LOCAL_MODEL_OVERRIDE=your-model-name
# Set this only if your model server requires a credential:
# AGENTGATE_LOCAL_API_KEY=your-server-key
```

Replace the address and model name with your server's values. Give the server's base
address only; the gateway appends `/v1/chat/completions` to it.

Start or restart the gateway, then send the earlier chat request again to check
that your model server responds.

This setup still uses the heuristic scanner, which misses many attacks and
is meant only for trying the gateway out. [Scanner backends](configuration.md#scanner-backends)
explains how to set up the classifier or the LLM judge instead.

## Using it with an agent application

Set your application's OpenAI-compatible base URL to
`http://127.0.0.1:4100/v1`. The client key is the credential your
model server expects, unless you configured a replacement key on the gateway.
See [credentials](configuration.md#credentials).

To label a client in audit records, see [client labels](configuration.md#client-labels).

See [Using with OpenCode](opencode.md) for client configuration and HTTP tool setup.

## Further reading

- [Configuration](configuration.md) — settings, scanner backends, credentials, and limits.
- [Audit log](audit.md) — query recorded requests and inspect saved content.

← Back to the [write-up](index.md).
