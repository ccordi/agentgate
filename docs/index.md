---
title: agentgate
---

# Mind Your Prompts and Tools

*A proxy can guard one, not the other.*

---

In June 2025, researchers at Aim Security disclosed [EchoLeak](https://www.catonetworks.com/blog/breaking-down-echoleak/) (CVE-2025-32711). A malicious instruction embedded in an email reached Copilot through its retrieval pipeline. Without any action from the recipient, the agent read the email, read the embedded instruction, and exfiltrated internal data to an external server — the first documented zero-click exfiltration via prompt injection in a production AI system. Microsoft's own classifier didn't catch it on the way in.

I built a reverse proxy that sits between an agent and the model it calls to scan for injected instructions like EchoLeak's — instructions buried in the content an agent reads — before they reach the model. On the kind of buried, indirect injection EchoLeak used, a standard injection classifier caught just 31%; the guard I built caught 98.6% with the local model I ran it on (89% even on a weaker alternative). But catching what the agent *reads* doesn't touch what it then *does* — and an agent's actions happen off the model wire, where a proxy can't see them. Closing that gap meant moving enforcement off the wire entirely.

---

## The problem: an agent can't tell your instructions from the ones it reads

To get anything done, an agent reads from places its operator doesn't control: web pages, emails, retrieved documents, the output of the tools it calls. Any of that untrusted input can be written to look like a command.

Models do receive role and message-boundary cues, but they do not reliably treat those cues as authority boundaries. An instruction you typed and one buried in a fetched document can both steer the agent. That's prompt injection, and it follows from how agents mix trusted and untrusted input.

**The useful enforcement signal is the channel the text arrived on** — your turn versus a tool result. The gateway can enforce that boundary before the model sees the text.

---

## Where the gateway sits

The gateway is a transparent proxy for the OpenAI Chat Completions API. Point an agent's base URL at it and it intercepts every model call: no code change in the agent, no change to the model. Transparent describes the integration, not the payload — what gets forwarded is edited, by the routing and redaction below and by a client-compatibility prompt rewrite on the local route. I ran it in front of a general assistant I use daily: email, web, messaging — the classic injection surface.

![Where the gateway sits: your trusted turn and untrusted tool and retrieval output converge at the agent; every model call then passes through the gateway, which routes sensitive content to a local model and redacted content to the cloud provider.](diagrams/gateway-architecture.svg)

*Responses stream back along the same path. Not drawn: the audit log, spend caps, block responses, and the egress gate — the sections below cover them.*

The local-LLM guard scans tool and retrieval output, not the operator's own turn. The default DeBERTa and heuristic backends also scan the newest user message. Choosing a backend therefore changes both the detector and its scan surface.

It also does two things with the content it routes: it sends material the sensitivity classifier marks sensitive to an on-device model instead of the cloud, so detected sensitive data never leaves the host, and it redacts secrets from anything that does go out. The egress gate reuses that classifier when it evaluates an outbound tool request.

---

## The recall win

The default inbound scanner is a DeBERTa classifier fine-tuned for injection detection. It did well on *overt* attacks — payloads that say "ignore previous instructions" outright — and much worse on indirect ones. The garak `latentinjection` probes hide their instruction inside an ordinary-looking document the model is processing, and the classifier caught only 31% of those. The result suggests a blind spot around indirect framing, not a proven account of the model's internal cause.

Context length does not explain this result. The scanner splits long inputs into overlapping 512-token windows, and the missed probes were short enough to fit one window. The measured gap is associated with indirect framing, though the evaluation does not establish the classifier's internal cause.

So I added a guard backed by a local LLM and evaluated the item as a single prompt.

**On these indirect attacks, recall went from 31% to 98.6%** — the local-LLM guard on the same garak `latentinjection` probes. A weaker alternative model reached 89% on the same set. That is evidence for this detector and corpus, not a general claim that larger models always outperform classifiers.

Recall is the fraction of real attacks the scanner caught. Computing it requires knowing how many attacks were missed. With garak's published `latentinjection` probes, every item is a known attack by construction — so every miss is counted, and recall follows directly.

These figures are recall on injected *instructions* — and specifically the indirect, buried kind these probes test. An attack that instead steers what the agent *does*, without ever planting an instruction to detect, is a separate and harder problem this number doesn't measure. A scanner on the wire can narrow the injection gap; closing the action gap means governing what the agent does, not just what it reads.

---

## Beyond the model wire

A proxy on the model wire sees everything the model reads and says. Pointing a real coding agent at the gateway showed me the limit.

The gateway hard-blocked a poisoned tool result on the way in — a 400, before the model ever saw it. That worked as designed. But a wire proxy can only act on what the model reads, and **what the agent *does* is a different surface.** Tool execution is client-side: when an agent runs a shell command or calls an HTTP tool, it happens in its own harness, off the model wire. A proxy there can police what the model *reads* and *says*. It cannot police what the agent *does*.

A control can act only on what it can see, and the action is not visible from the model wire.

So I stopped trying to enforce actions from the wire and **split the decision from the enforcement** — the PDP/PEP pattern from access control:

![PDP/PEP split: the agent's outbound request goes to the gated egress tool — the PEP, inside its harness — which asks the gateway — the PDP — for a verdict against the destination allowlist and payload sensitivity; on allow the request reaches the destination host, on deny nothing leaves the machine.](diagrams/pdp-pep-flow.svg)

*A missing or malformed verdict counts as deny — the tool fails closed.*

The gateway became the **Policy Decision Point**: an endpoint that evaluates an outbound request against destination allowlists and the sensitivity of its payload. Enforcement moved to a **Policy Enforcement Point** inside the agent's harness — a gated egress tool, set up as the agent's only network path. It asks the PDP before every call.

I drove a real agent to POST a secret-bearing config file to a host that wasn't allowlisted. The tool consulted the PDP, and the PDP **denied it** — `decision: deny`, payload classed `sensitivity=secret`, nothing left the machine.

No scanner could have stood in for the gate here. There was no injected instruction to detect: nothing malicious arrived in what the agent *read*. The secret left in what the agent *did* — an outbound request the scanner never inspects.

---

## What it costs, and where I'd put it

Three tiers of added latency:

- **Proxy overhead — 1–2 ms** (always paid). The gateway's own work — parse, inject-scan, spend check, SSE tee, fire-and-forget audit — adds a millisecond or two at the tail (server-side, against a deterministic mock upstream, with the regex guard as the scanner). The stream is not buffered, and the proxy did not dominate this benchmark. This tier is not backend-invariant: a model-backed scan costs more than the proxy work itself. ([full benchmark](benchmarks.md))
- **DeBERTa scan — ~40 ms p50** per scanned item, but the tail is much longer: p99 ran into the hundreds of milliseconds when the box was also serving a local model. The default guard scans each trailing tool result and the newest user message. Budget for the tail if the same machine is generating.
- **Local-LLM guard — ~1.4 s p50, ~1.6 s p95** per untrusted item (~3.6 s on a cold model load). This is the opt-in recall win.

Scoping limits where the 1.4 s cost lands: the LLM guard runs only on requests with trailing tool output, so ordinary user↔assistant turns do not pay it. Whether inline blocking is acceptable depends on the workload.

---

## Limitations

- **This is cooperative enforcement, not containment.** The PEP gates the egress tool and hard-blocks it — for an agent that stays on that path, including one misled by an injection but still running its normal harness. It doesn't take a maliciously rewritten client to get around it: denied its first request, a persistent agent keeps problem-solving and reaches an ungated path on its own. The gate holds for an agent that asks; it isn't a sandbox.
- **Shellout is the easy ungated path.** The gate reads the first token of a command, so `sh -c 'curl …'` clears a `curl` block; and the block list is the wrong shape for the problem anyway — `python3`, `git push`, `npx`, `ssh` and anything written into a shell profile to run later are all ungated network paths. Closing that class means gating those surfaces too — or, for true containment, OS-level network sandboxing, a different problem and out of scope here.
- **It's a single-user deployment, not production.** "Zero false positives on the traffic it scans" means one person's real traffic, not a fleet blocking multi-tenant production.

The full map — assets, adversary capability, trust boundaries, and which control covers which threat (plus what nothing covers) — is the [threat model](threat-model.md).

---

## How I measured this

The gateway is built in Python/FastAPI; the local models run on-device via [oMLX](https://github.com/jundot/omlx). Recall is an offline garak run on the independent `latentinjection` probes (n=72): 98.6% (71/72) with the stronger local model, 89% (64/72) with a weaker one, versus 31% for DeBERTa. The local-LLM guard is opt-in; the always-on default stays the fast DeBERTa scanner — so this recall is what the guard achieves when enabled, not what blocks every request by default. And the LLM backend ships no default model of its own: you point it at a local endpoint, which it requires to be loopback, since a guard that reads untrusted content must not ship it to a cloud API.

The live-traffic false-positive measurement is pinned to the channel the local-LLM guard actually reads: 0/109 on the scanned tool/retrieval channel. Running the same guard over the user channel as a check, it over-fired on ~16% of benign turns (7/44; model-dependent, small sample). Scoping that backend to the untrusted channel avoided those false positives in this sample.

That 16% is not a stable guard-wide rate. The captured user turns carry a timestamp prefix the agent prepends. With that prefix removed — the form committed in this repository after anonymization — the same check flagged 52% (23/44). The result is format-sensitive, and the sample comes from one agent with few direct human commands. At either end of the range the design conclusion is the same: the user's own turn is where this guard is least trustworthy, which is exactly why it never sees it. A purpose-built benign-imperative corpus is still needed.

**What reproduces from this repository?** Re-running the evaluation against the same local model reproduced every figure unaffected by the scrubbed text to the exact item count: all five DeBERTa cells, both tool-channel cells, and both robustness probes. Two changed: the guard's garak recall by one item out of 72, showing run-to-run variation, and the user-channel false-positive rate, from 7/44 to 23/44, because anonymization removed the timestamps. The cell-by-cell comparison is in `eval/redteam/sections/local_llm_guard.md`, which the red-team report includes verbatim.

**Reproducing these numbers.** Every claim above maps to a command, and about half of them need nothing but a clone:

- The recall figures and the false-positive rate on the scanned channel come out of `python -m eval.redteam llm-eval` — the guard columns need a local OpenAI-compatible model server, the DeBERTa column needs the `guard` extra and a pulled model, and the benign split of that same run is where the tool-channel FP rate comes from. The robustness probes are `probes-eval`.
- Corpus-wide scores for the always-on backends are `python -m eval.redteam score [--detector deberta]`, and the sensitivity-classifier over-fire report is `classifier-eval`. Both offline, no keys.
- The overhead tiers are [the benchmark](benchmarks.md), regenerated by `bench/run.py` (needs k6).
- The controls firing at all — injection block, egress deny, audit trail — is `bash scripts/demo.sh`: no keys, no model, about ninety seconds.

The corpus files are frozen snapshots rather than live captures, so a clone reproduces the offline numbers exactly; `eval/redteam/README.md` records which of them are re-fetchable byte-for-byte and which aren't, and `eval/redteam/sections/local_llm_guard.md` carries the snapshot date and the cell-by-cell re-run above.

---

## Conclusion

The two results expose different limits. The local-LLM guard raised indirect-injection recall from 31% to 98.6% on this corpus (89% with the alternative model). The model-wire proxy still could not govern client-side tool execution, so the HTTP control moved to a policy check inside the harness.

The egress gate covers its HTTP tool today. The same PDP/PEP split could be applied to shell and subprocess paths, but those paths are not gated here.

---

*The code is a curated public extract: the gateway, the adversarial-eval harness, and the test suite. The eval corpus is full of secret-shaped strings (`sk-…` and friends) on purpose — they're synthetic fixtures for the redaction tests, marked as such, no real credential ever in version control.*

*Setup: [Integration guide](integration.md)*
