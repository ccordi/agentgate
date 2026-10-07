# Configuration

This page lists the gateway's settings and their defaults.

`agentgate init` creates a `.env` file with demo settings: the built-in heuristic
scanner, a mock model server, and disabled sensitivity routing. The
[getting started guide](getting-started.md) walks through that setup.

The gateway reads `.env` from its working directory. To select another file, run
`uv run agentgate --env-file /path/to/settings.env`. Environment variables override
values in the file. Relative paths in settings use the working directory. Restart the
gateway after changing settings.

Setting names start with `AGENTGATE_`; nested settings use `__`, as in
`AGENTGATE_ROUTING__ENABLED`. [`.env.example`](https://github.com/ccordi/agentgate/blob/main/.env.example) contains examples
and comments for additional options.

For [Docker](docker.md), Compose reads the file selected by `--env-file .env`.
The `environment` block in `deploy/docker-compose.yml` passes the demo and model
server settings into the container. Add other settings to that block when you need them. Bind addresses
are configured in Compose.

## Providers and models

| Setting | Purpose |
| --- | --- |
| `AGENTGATE_DEFAULT_PROVIDER` | Provider used when sensitivity routing is disabled: `gemini`, `openai`, `ollama`, `local`, or the fixed-response `mock`. Default: `local`. |
| `AGENTGATE_ROUTING__ENABLED` | Route requests according to the sensitivity of their content. On by default. |
| `AGENTGATE_ROUTING__DEFAULT_LOCAL` | Provider for the local route. Default: `local`. |
| `AGENTGATE_ROUTING__DEFAULT_CLOUD` | Provider for the cloud route. Also defaults to `local`; choose a cloud provider explicitly to send requests off the machine. |
| `AGENTGATE_PROVIDERS` | JSON object defining providers by name. Each entry needs `name` and `base_url`; use `is_local: true` for a local model server. Optional fields: `api_key`, sent instead of the client's key, and `model_name`, which replaces the requested model. Replaces the default registry. See the [setup example](getting-started.md#connect-a-real-model-server). |
| `AGENTGATE_LOCAL_MODEL_OVERRIDE` | Model name to use for local requests, as expected by your model server. Set it when using the built-in `local` provider: its default model name is unlikely to exist on your server. |

With sensitivity routing enabled, the default rules send detected sensitive
content to the local provider and other requests to `AGENTGATE_ROUTING__DEFAULT_CLOUD`.
For example, set that value to `openai` to use OpenAI for the cloud route.

Cloud requests use the model named by the agent application unless the provider entry
sets `model_name`.

## Scanner backends

`AGENTGATE_GUARD_BACKEND` selects the prompt-injection scanner:

- **`deberta` (default):** The classifier, a model trained to score text for prompt
  injection. It blocks a request when a message it scans scores 0.995 or higher. It
  needs local model files and the optional `guard` dependencies;
  [setup below](#set-up-the-classifier-on-the-host). The gateway refuses to start if
  the model cannot load.
- **`heuristic`:** The built-in heuristic scanner: regular-expression checks with
  no model files.
- **`llm`:** The LLM judge, a local language model. It scans tool results only.
  Configure it with the `AGENTGATE_JUDGE_*` settings in
  [`.env.example`](https://github.com/ccordi/agentgate/blob/main/.env.example); its endpoint must be loopback.
- **`combined`:** The classifier and the LLM judge run concurrently; the
  stricter result wins. Both need their model setup, and the classifier must load
  at startup.

The [threat model](threat-model.md#prompt-injection-in-tool-results) explains
which messages each scanner checks and why.

`AGENTGATE_GUARD_OBSERVE_MODE=true` lets through requests the prompt-injection
scanner would block, and records in the audit log that they would have been
blocked. It doesn't cover
[tool descriptions](threat-model.md#prompt-injection-in-tool-descriptions):
a request whose tool descriptions contain instructions is still rejected.
Off by default.

The [Results](results.md#latency) page gives each scanner's measured latency.

### Set up the classifier on the host

The classifier is [PIGuard](https://huggingface.co/leolee99/PIGuard), a DeBERTa-v3 model;
the backend value `deberta` is named after that model family. The model is published
without the ONNX files the gateway loads, so a script in this repository downloads a fixed
version of it and converts it:

```bash
uv run --script scripts/convert_piguard_onnx.py
```

The script checks that the converted model gives the expected scores for a set of test
texts, then writes `model.onnx` and `tokenizer.json` to `models/piguard-onnx/` under the
checkout, where the gateway looks by default. To write them elsewhere, add `--out` and a
directory. The 0.995 threshold is set in the code for this model and stays the same if
`AGENTGATE_GUARD_MODEL_DIR` points at a different one.

The script downloads 746 MB and writes 745 MB. While it runs, it needs about 1.5 GB of free
disk and about 3.5 GB of memory, and it takes a few minutes. uv keeps the script's
dependencies, including PyTorch, in its cache, about 1 to 2 GB; the gateway does not use them.

Select the scanner in `.env`, with the model directory if it is elsewhere:

```dotenv
AGENTGATE_GUARD_BACKEND=deberta
AGENTGATE_GUARD_MODEL_DIR=/absolute/path/to/piguard-onnx
```

Start the gateway with the optional `guard` dependencies, which `uv run` installs
the first time:

```bash
uv run --extra guard agentgate
```

The classifier loads into the gateway process. Your model server continues to
run separately.

### Scanner models in Docker

The supplied image supports the heuristic scanner and contains no model weights
or ONNX runtime. For the classifier or `combined`, build an image with
`--extra guard` added to both `uv sync` lines in the Dockerfile. Convert the model on the
host [as above](#set-up-the-classifier-on-the-host), mount its directory read-only, for
example `/absolute/path/to/piguard-onnx:/models/piguard-onnx:ro`, and pass
`AGENTGATE_GUARD_MODEL_DIR=/models/piguard-onnx` into the container. The image's user
must be able to read those files.

The LLM judge's model server must be reachable at a loopback address from the
gateway process. A server on the Docker host is not on the container's loopback
interface. To use the LLM judge with a model server on the host, run the gateway on
the host rather than in Docker.

## Credentials

| Setting | Purpose |
| --- | --- |
| `AGENTGATE_ADMIN_TOKEN` | Required token for the gateway's admin API. |
| `AGENTGATE_PDP_TOKEN` | Required token for the HTTP tool's policy checks. |
| `AGENTGATE_LOCAL_API_KEY` | Optional model server key. Replaces the client's credential on requests to the `local` provider. |
| `AGENTGATE_REQUIRE_ISSUED_KEYS` | Require client keys issued by the gateway. Off by default. Create and revoke keys through the admin API. |

`agentgate init` generates and saves the admin and HTTP tool tokens. Both are
required at startup and must differ from each other and any local model server key.

The admin API has two groups of endpoints on the gateway: `/admin/kill/{key_id}`
for the [spending kill switch](#spending) and `/admin/keys` for issued keys. Each
call carries the admin token as a bearer token, `Authorization: Bearer <admin token>`.

Normally, the gateway forwards the client's key to the selected provider. A provider's
`api_key`, when set, replaces it.

With `AGENTGATE_REQUIRE_ISSUED_KEYS` enabled, clients send a gateway-issued key,
and a request with a missing, unknown or revoked key is rejected with HTTP 401.
`POST /admin/keys` issues a key, with an optional JSON body such as
`{"label": "laptop"}`; the response holds the key once, and the gateway keeps
only a hash. Give it to the agent application as its API key. `GET /admin/keys`
lists keys and `DELETE /admin/keys/{key_id}` revokes one. Set each cloud
provider's `api_key`; requests to a cloud provider without one are rejected with
HTTP 503. This includes the demo's `mock` provider, which is not marked `is_local`.

## Client labels

To identify a client in audit records, set its base URL to
`http://127.0.0.1:4100/a/<agent_id>/v1`, replacing `<agent_id>` with a name you choose.

## Controls and limits

| Setting | Purpose |
| --- | --- |
| `AGENTGATE_REDACTION_ENABLED` | Mask detected secrets and personal data in requests sent to cloud providers. On by default. |
| `AGENTGATE_EGRESS__ALLOWLIST` | JSON array of trusted destinations for the HTTP tool, for example `["api.github.com"]`. |
| `AGENTGATE_ADMISSION_ENABLED` | Limit active and queued requests. Off by default. |
| `AGENTGATE_REQUIRE_SHARED_LIMITS` | Require Redis for shared spend counters. Off by default. When enabled, the gateway refuses to start if Redis is unavailable. |
| `AGENTGATE_REDIS_URL` | Redis server for spend counters and kill switches. Default: `redis://127.0.0.1:6379/0`. If it is unreachable at startup, they are kept in the gateway process instead. |
| `AGENTGATE_PRIVATE_REPO_MARKERS` | JSON array of strings that appear only in your private material, such as an internal package name or a confidentiality notice: `["acme-internal"]`. A request with one in the first 20,000 characters of its text, matched exactly and case-sensitively, is classified `private_repo` and kept on the local route. Needs sensitivity routing. Empty by default. |

The [Concurrency](scale-design.md#admission-control) page lists the admission limits,
their settings and defaults, and how they are enforced.

### Spending

| Setting | Purpose |
| --- | --- |
| `AGENTGATE_SPEND_CLOUD_USD_CAP` | Estimated spending allowed per key per window on cloud providers, in USD. Default: `5.0`. |
| `AGENTGATE_SPEND_LOCAL_REQUEST_CAP` | Requests allowed per key per window on local providers. Default: `10000`. |
| `AGENTGATE_SPEND_WINDOW_S` | Window length in seconds. Default: `3600`. A key's window starts at its first request and resets when it expires. |
| `AGENTGATE_SPEND_KILL_TTL_S` | Seconds after which a tripped kill switch clears itself. Unset by default, so it stays tripped until cleared. |

Each cloud request's estimated cost is added to a per-key total; the
[audit guide](audit.md#cost-estimates) explains the estimate. A key that reaches
the cloud cap trips its kill switch, and every further request on that key is
rejected with HTTP 429. Requests over the local cap are rejected with HTTP 429
until the window resets, without tripping the switch. Without Redis, counters and
kill switches are kept in the gateway process, so a restart resets spending to
zero and clears any tripped kill switch.

To clear a key's kill switch, call the admin API with the key's identifier. Every
audit record carries it as `key_id`. With issued keys, it is also the `key_id`
returned when the key was issued:

```bash
curl -X DELETE -H 'Authorization: Bearer <admin token>' http://127.0.0.1:4100/admin/kill/<key_id>
```

A `POST` to the same address trips it. Clearing the switch before the key's window
has reset does not last: the next cloud request finds the total still over the cap
and trips it again.

## Audit storage

| Setting | Purpose |
| --- | --- |
| `AGENTGATE_DATABASE_URL` | Database for audit records. Default: `sqlite+aiosqlite:///./data/agentgate.db`, a SQLite file under the working directory. |
| `AGENTGATE_CONTENT_ENC_KEY` | Key that encrypts saved message content. Unset by default, and no content is saved without it. |
| `AGENTGATE_CONTENT_CAPTURE_ENABLED` | Save message content when the key is set. On by default. |
| `AGENTGATE_CONTENT_SAMPLE_RATE` | Fraction of unflagged requests whose content is saved, picked at random. Default: `0.05`, one in twenty. |
| `AGENTGATE_CONTENT_RETENTION_DAYS` | Days saved content is kept before an hourly sweep deletes it. Default: `30`. |

Content is saved only for requests sent to a cloud provider and not classified
as sensitive: every such request the scanner flagged, plus a random sample of
the rest at the sample rate. It is redacted before it is encrypted. Generate a
key with:

```bash
uv run python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

The [audit guide](audit.md) covers querying records and reading saved content.
Most gateway settings are defined in [`config.py`](https://github.com/ccordi/agentgate/blob/main/src/agentgate/config.py)
`Settings`.
