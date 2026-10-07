# Using with OpenCode

Connect OpenCode to agentgate so the gateway checks its model requests and records
them in the audit log. First, [connect a real model server](getting-started.md#connect-a-real-model-server)
and leave the gateway running.

## Configure OpenCode

Add this to your OpenCode project's `opencode.json`. Replace each
`your-model-name` with the model name you configured in Getting Started:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "agentgate/your-model-name",
  "provider": {
    "agentgate": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "agentgate (local gateway)",
      "options": {
        "baseURL": "http://127.0.0.1:4100/v1",
        "apiKey": "{env:AGENTGATE_OPENCODE_API_KEY}"
      },
      "models": {
        "your-model-name": {
          "name": "Local model via agentgate"
        }
      }
    }
  }
}
```

## Send a prompt

Start OpenCode from your project directory, supplying your model server's API key:

```bash
AGENTGATE_OPENCODE_API_KEY=your-api-key opencode
```

Use `local` instead of `your-api-key` if the server needs no key or the gateway
already supplies it. See [credentials](configuration.md#credentials) for other
authentication setups.

Send a simple prompt, such as “Reply with hello.” After the response finishes,
open another terminal in your agentgate checkout and check the audit log:

```bash
uv run agentgate audit tail
```

Look for a new record with your model name and status `200`. This confirms that
the request passed through agentgate. For Docker or a different settings file,
use the command in the [audit guide](audit.md).

## HTTP requests made by tools

The configuration above sends OpenCode's model requests through agentgate. HTTP
requests made by its tools require a separate setup.

This optional setup gives OpenCode a `safe_http_request_tool` that asks the gateway
to check each destination and request payload before sending it. If the gateway
denies the request or cannot be reached, the tool does not send it.

Add these top-level entries to the same `opencode.json`. They register the tool as
an MCP server and disable OpenCode's built-in fetch and search tools, along with
direct `curl`, `wget`, and `fetch` commands:

```json
{
  "mcp": {
    "agentgate-egress": {
      "type": "local",
      "command": [
        "uv", "run", "--extra", "egress-mcp",
        "python", "-m", "agentgate.egress.mcp_server"
      ],
      "cwd": "/absolute/path/to/agentgate"
    }
  },
  "permission": {
    "webfetch": "deny",
    "websearch": "deny",
    "bash": {
      "curl *": "deny",
      "wget *": "deny",
      "fetch *": "deny"
    }
  }
}
```

Replace `cwd` with your agentgate checkout. The MCP server reads
`AGENTGATE_PDP_TOKEN` from that directory's `.env` to authenticate its checks with
the gateway. `agentgate init` has already saved this token there.

Restart OpenCode to load the tool configuration. To permit detected sensitive data
to be sent to a trusted destination, add its host to `AGENTGATE_EGRESS__ALLOWLIST`
on the gateway, for example `'["api.github.com"]'`, and restart the gateway.

These restrictions rely on OpenCode using the supplied tool; the
[threat model](threat-model.md#out-of-scope) explains what they don't cover.

These entries were checked against [OpenCode 1.18.31](https://github.com/anomalyco/opencode/releases/tag/v1.18.31);
see its [MCP configuration](https://opencode.ai/docs/mcp-servers/) and
[permissions reference](https://opencode.ai/docs/permissions/).

← Back to [Getting started](getting-started.md).
