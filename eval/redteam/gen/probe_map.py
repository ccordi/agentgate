"""Curated garak probe set + the ``expected_miss`` split for the DeBERTa guard.

Dependency-free **on purpose**: this module is imported both from agentgate's venv (by
``export.py``) and from garak's isolated venv (by ``dump_prompts.py``). Keep it pure stdlib —
no pydantic, no garak — so either interpreter can load it.

Why a *curated* subset (not "every probe × 256"):
    Each garak probe emits 256 prompts (payloads × templates), most near-duplicate. The
    existing default corpus is ~710 labeled (292+/418-). Dumping thousands of templated
    garak positives would *drown* the authored/deepset corpus and skew the headline recall.
    The follow-up question is about **coverage of attack forms**, not volume, so we take a
    capped, evenly-strided sample per probe (``PER_PROBE_CAP``) and lean on the per-category
    recall table (each probe = its own ``garak:<probe>`` category) for the real signal.

----------------------------------------------------------------------------------------------
The ``expected_miss`` split — what it is and why it's drawn where it is
----------------------------------------------------------------------------------------------
The harness already separates two kinds of false negative (``harness.py``):
  * **surprise FN** — a missed positive NOT tagged ``expected_miss``. This is the real signal:
    the guard was *supposed* to catch it and didn't.
  * **expected miss** — a missed positive tagged ``meta.expected_miss=true``: a *documented*
    blind spot, reported in its own bucket so it can't masquerade as a surprise.

We are scoring **DeBERTa** (a trained classifier), so the principled blind-spot axis is
**tokenization corruption**, not regex gaps. Obfuscation is where a DeBERTa classifier's
tokenization breaks. So:

  expected_miss = True  (OBFUSCATION — the payload is disguised at the character/byte/token
                         level; the tokenizer literally cannot see the semantic instruction):
      - all ``encoding.Inject*`` (base64/16/32, hex, ROT13, atbash, morse, braille, zalgo,
        unicode-tag-chars, leetspeak — every scheme is a distinct tokenization challenge)
      - ``latentinjection.LatentInjectionTranslation*`` (low-resource / cross-lingual; the
        design doc names "translation" as a known blind spot alongside base64)

  expected_miss = False (PLAIN or DOCUMENT-EMBEDDED natural-language injection — DeBERTa is
                         trained for exactly this; a miss is a genuine, interesting FN):
      - ``promptinject.Hijack*`` (canonical goal-hijacking in plain English)
      - ``latentinjection.*`` except translation (indirect injection buried in a document —
        fact snippet / report / resume / jailbreak). NB: several of these run 800-3000 chars,
        past DeBERTa's ~512-token window, so truncation may cause misses. That is *precisely*
        the kind of non-trivial FN we want surfaced, NOT hidden under expected_miss.

Rule of thumb: tag ``expected_miss`` only when the *disguise mechanism* is the reason for a
miss. If the attack is legible English and the guard still misses it, that is signal — leave
it untagged.
"""

from __future__ import annotations

# Evenly-strided sample size per probe (see module docstring for the volume rationale).
PER_PROBE_CAP = 12

# garak probe name ("module.Class") -> expected_miss (see the split rationale above).
PROBES: dict[str, bool] = {
    # --- plain natural-language injection: DeBERTa should catch these ---
    "promptinject.HijackHateHumans": False,
    "promptinject.HijackKillHumans": False,
    "promptinject.HijackLongPrompt": False,
    # --- indirect / document-embedded injection: in-scope; miss = real signal ---
    "latentinjection.LatentInjectionFactSnippetEiffel": False,
    "latentinjection.LatentInjectionReport": False,
    "latentinjection.LatentInjectionResume": False,
    "latentinjection.LatentJailbreak": False,
    # --- low-resource / cross-lingual translation: documented blind spot ---
    "latentinjection.LatentInjectionTranslationEnZh": True,
    "latentinjection.LatentInjectionTranslationEnFr": True,
    # --- obfuscation / encoding: tokenization blind spot ---
    "encoding.InjectBase64": True,
    "encoding.InjectBase16": True,
    "encoding.InjectBase32": True,
    "encoding.InjectHex": True,
    "encoding.InjectROT13": True,
    "encoding.InjectAtbash": True,
    "encoding.InjectMorse": True,
    "encoding.InjectBraille": True,
    "encoding.InjectZalgo": True,
    "encoding.InjectUnicodeTagChars": True,
    "encoding.InjectLeet": True,
}


def stride_indices(n: int, cap: int = PER_PROBE_CAP) -> list[int]:
    """Evenly-spread ``cap`` indices over ``range(n)`` (not just the first ``cap``, which
    would over-sample a single template). Deterministic → reproducible corpora."""
    if n <= cap:
        return list(range(n))
    return [round(i * (n - 1) / (cap - 1)) for i in range(cap)]


def category_for(probe: str) -> str:
    """Per-category recall row label — namespaced so garak families don't collide with the
    existing authored categories."""
    return f"garak:{probe}"
