# Concurrency

The gateway is designed for a single operator, who may run several agents at once,
such as a background batch job and an editor session. This page explains how one
gateway process limits requests from different clients and what happened when a
client shared a model server with a heavy batch job.

## Admission control

Concurrent requests compete for the gateway's scanner threads and memory, and for the
model server's resources. Without limits, any single client can starve the others. The
gateway's admission control mitigates this by capping the number of active requests per
client key (the credential each client sends; see [Client keys](#client-keys)) and for
the process as a whole. It is off by default;
`AGENTGATE_ADMISSION_ENABLED=true` turns it on with these limits:

| Limit | Setting | Default |
|---|---|---:|
| Active requests per key | `AGENTGATE_ADMISSION_PER_KEY` | 8 |
| Active requests per process | `AGENTGATE_ADMISSION_GLOBAL` | 64 |
| Waiting requests per key | `AGENTGATE_ADMISSION_QUEUE_DEPTH` | 32 |
| Maximum wait for a key slot | `AGENTGATE_ADMISSION_WAIT_DEADLINE_S` | 5 seconds |

A request that arrives when its key already has 8 active requests
waits, in arrival order, for up to 5 seconds. If the key's queue is already full, or the
wait runs out, the request is rejected with 429. A request that gets a key slot but
finds all 64 process slots taken is rejected with 503. The response carries no
`Retry-After` header; when to retry is the client's decision. A request counts as active
until its streamed response has finished or the client has closed the stream, so a long
generation holds its slot for its whole duration.

Rejections are counted by reason in the Prometheus metric
`agentgate_admission_shed_total`, served at `/metrics`, and logged. They are not
written to the audit log, which keeps database work out of an overload, so the audit
log shows only requests that were admitted. Admission is decided before the request
body is read or scanned; with issued keys enabled, the key is verified first.

Requests admitted to the same model server still share its compute and can slow each
other down.

### What one client experienced beside a batch job

In a test:

- The model server (llama.cpp running Gemma 4 E2B) was configured to serve 4 concurrent
  requests.
- Admission control, when on, limited each key to 8 active requests, the default.
- A client sent one request at a time, while a batch job on another key kept 32 requests
  going at once, sending a new one as each finished.

Alone, the client's requests took about 2 seconds on average. With the batch job running
and admission control off, they took about 35 seconds, because all 32 batch requests
reached the model server and competed with the client's. With admission control on, they
took about 10 seconds, because the gateway allowed only 8 of the batch job's requests to
reach the model server at a time, resulting in less congestion at the server.

## Client keys

The gateway uses client keys to track each client's active requests, waiting requests
and spending. By default the key is whatever credential the client sends, taken as it
is and never verified. That keeps well-behaved clients from crowding each other out,
but a client that presents a different credential gets a fresh set of limits.
With `AGENTGATE_REQUIRE_ISSUED_KEYS=true`, the gateway accepts only keys it issued;
[Credentials](configuration.md#credentials) covers issuing and
revoking them. The key is checked before admission and before the request body is
read. Missing, unknown and revoked keys all receive the same 401 response. Limits
and audit records then use the key's stored identifier.

## Audit records and traces under load

The audit log is written in the background after each response has finished, so a
failed write never delays or fails the response. Writes go through a connection pool in
front of SQLite's single writer; when many requests finish at once, a write can fail to
get a connection in time, and the log misses that request.
[Missing records](audit.md#missing-records) lists what the log omits and the metric
that counts failed writes.

A load test against a mock model server compared audit records with exported traces over
four runs. All 59,342 saved audit records had exactly one matching `chat` span, and the
pairs agreed on status, scan time and total duration. Another 2,986 requests had a span
but no audit record. In each run, that number matched the audit-write failure counter.

For live observation, Prometheus metrics cover request counts, latency, token usage,
estimated cost, rejections and failures; client key identifiers stay out of metric
labels. Traces are optional: add
`--extra tracing` to the gateway's `uv run` command and set `AGENTGATE_OTLP_ENDPOINT`
to the collector's traces URL, such as `http://127.0.0.1:4318/v1/traces`. A `chat`
span carries request metadata, scan verdicts, timings and token usage, never message
text. It is emitted when a forwarded response ends or a check rejects the request before
forwarding. Requests refused by the `Host` header check or rejected at authentication or
admission, or whose connection to the model server failed, produce no `chat` span. The
Compose `otel` profile runs a collector and Tempo, a trace store.

## Scanning under load

Requests wait for scanning to finish before the gateway forwards them.

The classifier runs on the
gateway's CPU through ONNX Runtime. It uses worker threads so the gateway can handle other
requests while a scan runs.
`AGENTGATE_SCAN_EXECUTOR_THREADS` sets the worker pool size, and
`AGENTGATE_GUARD_INTRA_OP_THREADS` sets the threads used within each inference.

The LLM judge evaluates tool results using a local model server. It caches verdicts in
memory and a local file. When a result is not cached, the request waits for a model
call. If the LLM judge shares a model server with the agent, those calls compete with the
agent's generation requests.

## Beyond one process

The measurements on this page used a single gateway process. Multiple processes would
need a shared audit database, with Redis sharing spending counters and kill switches
(see [Controls and limits](configuration.md#controls-and-limits)). Admission limits
would still apply separately in each process. Running multiple processes or hosts has
not been tested.
