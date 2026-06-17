## Local-LLM Guard Evaluation

A local-LLM guard (a Q4-quantized **Gemma 4 26B A4B IT**, served locally via oMLX in this configuration — zero cloud egress) is the backend under evaluation here, compared against a Q4-quantized **Qwen3.6-35B-A3B** and the DeBERTa baseline. (Both LLMs are local quantizations of the public base models; any OpenAI-compatible local inference server can serve them.) The LLM guard is **opt-in** (`AGENTGATE_GUARD_BACKEND=llm`; the code default is `deberta`); **Gemma 4 26B A4B IT** is the default *model* of the LLM backend. The **headline is recall**: the LLM guard reads the *whole* untrusted item, so it catches the buried / document-embedded payloads DeBERTa's 512-token window never sees. Channel-scoping (scan tool outputs, not the user's own turn) keeps the false-positive rate at zero on the scanned channel.

| Metric / Sub-Corpus | Gemma Guard (Primary) | Qwen Guard (Alternative) | DeBERTa Baseline | Note / Context |
| --- | --- | --- | --- | --- |
| **Recall — Independent Garak** | **98.6% (71/72)** | 88.9% (64/72) | 30.6% (22/72) | NVIDIA garak `latentinjection` (independent) |
| **Recall — seed_mutate** | 100.0% (28/28) | 100.0% (28/28) | 32.1% (9/28) | ⚠️ **Circular** (discounted; Gemma/Qwen family bias) |
| **FP — Untrusted channel** (scanned) | **0.0% (0/109)** | 0.0% (0/109) | 11.0% (12/109) | Benign `tool_output` captures — what the guard *does* scan |
| **FP — User channel** (not scanned) | 15.9% (7/44) | 4.5% (2/44) | 11.4% (5/44) | Benign `user` captures — what channel-scoping skips |
| **FP — Combined** | 4.6% (7/153) | 1.3% (2/153) | 11.1% (17/153) | All organic benign captures (FP) |

> [!NOTE]
> **What this shows.** The reproducible win is **recall**: the local-LLM guard catches more independent indirect-injection attacks than the windowed DeBERTa classifier (garak `latentinjection`), at **0% FP on the untrusted (`tool_output`) channel**. seed_mutate recall is **circular** — Qwen/Gemma variants are involved in generation/guarding — so the independent garak number is the honest recall.
>
> **On the trust boundary.** Channel-scoping — scan tool/retrieved content, never the user's own turn — is a deployment *design choice*. On the deduplicated capture (unique payloads), the primary guard over-fires on **15.9% (7/44)** of benign *user* messages. Read two ways, both honest: (1) the LLM guard is trigger-happy on benign user-channel text, and (2) channel-scoping sidesteps that cost — keeping the scanned (tool) channel at **0% FP** and the over-fire off the wire. The effect is model-dependent (Qwen: 4.5%, 2/44) and the sample is small (n=44, 7 events — wide CI). The capture is one automated agent's traffic, thin on diverse imperative human commands, so it under-tests the principle; a purpose-built benign-imperative corpus is a follow-up. Until then the trust-boundary claim stands as a design guarantee (we never second-guess the user's own turn), now with a measured user-channel cost behind it.

### Robustness probes

Two predicted failure modes, probed on **bare** encoded payloads (no cleartext "decode this and follow it" lure) and on benign tool results that merely *quote* attack strings:

| Probe | Gemma Guard | DeBERTa | Combined (OR) | Finding |
| --- | --- | --- | --- | --- |
| **Obfuscation in tool channel** — bare base64 / ROT13 / zero-width, n=15 | **93.3% (14/15)** | 46.7% (7/15) | **100% (15/15)** | The LLM decodes most bare encoded payloads (rationales show in-context decode); misses one ROT13 |
| **Security-meta-content FP** — benign tool results discussing injection, n=15 | **0.0% (0/15)** | 40.0% (6/15) | 40.0% (6/15) | DeBERTa over-fires on benign text that *quotes* attack strings; the LLM reads context and does not |

> [!NOTE]
> **Why keep both layers.** Not because DeBERTa backstops obfuscation — it doesn't here; the LLM is the stronger obfuscation reader. The evidence is **complementarity**: neither detector is individually perfect on bare obfuscation (LLM 93.3%, DeBERTa 46.7%), but the OR-combination is **100%** — the single LLM miss (a ROT13 item) was a DeBERTa hit, so their misses don't overlap on this corpus. DeBERTa also adds value the LLM can't: ~40 ms latency, and it can scan the user channel the LLM never touches by design.
> **Caveats:** n=15, single guard model (Gemma); complementarity is suggestive, not proven at scale.

### Latency (deployment cost)

Gemma 4 26B A4B IT, served locally via oMLX: **warm p50 ≈ 1.38 s, p95 ≈ 1.62 s** per untrusted item; **cold VRAM-swap load ≈ 3.6 s** (first call after the model is evicted). The guard fires **only on tool-using turns** (trust-boundary scoping), so ordinary user↔assistant turns pay nothing. Deployment-shape recommendation: inline-blocking is viable on tool turns at ~1.4 s.
