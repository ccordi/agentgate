# Results

This page gives the measurements behind the [write-up](index.md): how well each scanner
detects prompt injection, and how much time the gateway and its scanners add.

## Scanner accuracy

These tests compare the LLM judge with the classifier. The LLM judge ran with two
local models, 4-bit quantized versions of Gemma 4 26B A4B IT and Qwen3.6-35B-A3B.
[Configuration](configuration.md#scanner-backends) describes how to select each scanner.

| Measure | LLM judge (Gemma) | LLM judge (Qwen) | Classifier |
| --- | --- | --- | --- |
| Attacks caught, garak | 72 of 72 (100%) | 64 of 72 (88.9%) | 31 of 72 (43.1%) |
| Attacks caught, generated variants | 28 of 28 (100%) | 28 of 28 (100%) | 5 of 28 (17.9%) |
| Benign tool results flagged | 0 of 109 (0%) | 0 of 109 (0%) | 12 of 109 (11.0%) |
| Benign user messages flagged | 8 of 29 (27.6%) | 3 of 29 (10.3%) | 2 of 29 (6.9%) |
| All benign items flagged | 8 of 138 (5.8%) | 3 of 138 (2.2%) | 14 of 138 (10.1%) |

The garak attacks are its `latentinjection` attacks, which hide instructions inside
ordinary-looking documents. A Gemma model wrote the generated variants of ten attacks.
The benign items are anonymized traffic from one automated agent, with few direct human
commands.

Classifier scores of 0.995 or higher are marked as flagged; the gateway blocks requests at
the same score. Both language models ran at temperature 0 with thinking
disabled.

With Gemma, the LLM judge flagged more than a quarter of the benign user messages. In the
gateway it [scans only new tool results](threat-model.md#prompt-injection-in-tool-results),
so user messages never reach it.

### Robustness probes

The LLM judge with Gemma and the classifier also scored two small sets of 15 tool
results each: attacks encoded in base64, ROT13 or zero-width characters, with no cleartext
instruction to decode them, and benign text that discusses or quotes attacks.

| Set | LLM judge (Gemma) | Classifier | Either flags |
| --- | --- | --- | --- |
| Encoded attacks caught | 14 of 15 | 7 of 15 | 15 of 15 |
| Benign text about attacks flagged | 0 of 15 | 5 of 15 | 5 of 15 |

The classifier caught the one ROT13 attack the LLM judge missed, so flagging an
item when either scanner flags it caught every encoded attack in this small set. That
rule also kept the classifier's five false alarms. The gateway's `combined` backend flags
by the same rule.

## Latency

### Gateway processing time

The load-testing tool k6 sent requests through the gateway, at 50 to 200 requests per
second, to a local server that returns the same fixed streaming response each time. No
model ran. The gateway ran as one process with the heuristic scanner, with routing rules
disabled. The timings come from
the gateway's 5,404 audit records for the run and are in milliseconds.

| Timing | p50 | p95 | p99 | Max |
|---|---:|---:|---:|---:|
| Injection scan | 0.02 | 0.29 | 0.31 | 0.58 |
| Before forwarding | 0.12 | 1.03 | 1.08 | 1.29 |
| Mock server request and response | 1.34 | 2.33 | 2.66 | 7.34 |
| Total | 1.60 | 2.65 | 2.95 | 7.81 |

Before forwarding is the time from receiving a request to starting the call to the
mock server, and includes the injection scan. The mock server row starts at that
call and includes forwarding the response to the client.

### Live-traffic latency

OpenCode answered questions about a small Python project through the gateway, with no
other client, using a quantized version of Gemma 4 26B on a local model server and the
classifier scanning each request. The scan row covers all 96 requests; the other
rows cover the 72 larger requests, those carrying the project's files; the other
24 were short requests. Times are in milliseconds.

| Stage | N | p50 | p99 | Max |
| --- | --- | --- | --- | --- |
| Injection scan | 96 | 31.9 | 133.3 | 161.0 |
| Model generation | 72 | 4,332.8 | 24,167.3 | 32,057.4 |
| Total | 72 | 4,362.3 | 24,269.0 | 32,104.4 |

The scan runs in the gateway process on the CPU.

### LLM judge

With Gemma, the LLM judge took about 1.46 s at the median and 1.78 s at the 95th
percentile per tool result, over 30 scans with the model already loaded. The LLM judge
scans only new tool results, so turns without them do not wait for it.
