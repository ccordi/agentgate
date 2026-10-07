"""Model-backed injection guard — the classifier (ONNX) behind the `scan_text` interface.

Same call shape as `guards.heuristic.scan_text` and the same `Verdict` return type, so
callers can swap scanners with no other changes.
The backends are not interchangeable in behavior, though: they differ in scan surface
(the LLM judge sees tool output only), in coverage bound (this one scores at most
`_MAX_WINDOWS` windows — about 25,800 characters — of a single item, silently), and in
verdict shape (the LLM judge is binary).

Runs **in-process** on onnxruntime (no torch, no model server, no network) — a guard
must not egress content.

Default model: PIGuard (`leolee99/PIGuard`), converted to ONNX by
`scripts/convert_piguard_onnx.py`: a ~184M DeBERTa-v3 sequence classifier with labels
{0: benign, 1: injection}. The injection probability becomes `Verdict.score`. Inputs cap
at 512 tokens, so long tool outputs are **windowed** and the max window score is taken (an
injection anywhere in the content trips it).

Lazy-loaded singleton; the `guard` extra (onnxruntime, tokenizers, numpy) must be installed.
The model directory comes from `AGENTGATE_GUARD_MODEL_DIR`, set in the environment or `.env`.
"""

from __future__ import annotations

import shlex
from functools import lru_cache
from pathlib import Path

from agentgate.config import env_setting
from agentgate.guards import Verdict

# The classifier's own thresholds, set for PIGuard: its scores crowd toward 1, so at the
# heuristic scanner's 0.4 and 0.7 (`agentgate.guards`) it would block much benign tool
# output. It flags and blocks at one score, so every flag is also a block.
# Score at/above this flags and logs the request.
FLAG_THRESHOLD = 0.995
# Score at/above this blocks the request inbound.
HARD_THRESHOLD = 0.995

_DEFAULT_DIR = "models/piguard-onnx"
_MAX_TOKENS = 512
_WINDOW_CHARS = 1800   # ≈ <512 tokens; conservative
_WINDOW_OVERLAP = 200  # keep injections that straddle a window boundary
_MAX_WINDOWS = 16       # bound work on pathologically long inputs


def configured_model_dir() -> Path:
    """The classifier's model directory: ``AGENTGATE_GUARD_MODEL_DIR`` from the
    environment or `.env`, else the default path under ``models/``."""
    return Path(env_setting("AGENTGATE_GUARD_MODEL_DIR", _DEFAULT_DIR))


class DebertaGuard:
    def __init__(self, model_dir: str | None = None) -> None:
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self._np = np
        d = Path(model_dir) if model_dir else configured_model_dir()
        if not (d / "model.onnx").exists():
            out = "" if d == Path(_DEFAULT_DIR) else f" --out {shlex.quote(str(d))}"
            raise FileNotFoundError(
                f"guard model not found at {d}/model.onnx — build it with "
                f"`uv run --script scripts/convert_piguard_onnx.py{out}` or set "
                "AGENTGATE_GUARD_MODEL_DIR."
            )
        self._tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self._tok.enable_truncation(max_length=_MAX_TOKENS)
        opts = ort.SessionOptions()
        # Per-inference thread budget (same sources as the model directory). ORT's
        # default is every physical core PER RUN, which stacked under the scan thread
        # pool oversubscribes the whole box; 0 keeps that default. Scores are unaffected
        # — this moves speed only, never a verdict.
        intra = int(env_setting("AGENTGATE_GUARD_INTRA_OP_THREADS", "0"))
        if intra > 0:
            opts.intra_op_num_threads = intra
        self._sess = ort.InferenceSession(
            str(d / "model.onnx"), sess_options=opts, providers=["CPUExecutionProvider"]
        )

    def _windows(self, text: str) -> list[str]:
        if len(text) <= _WINDOW_CHARS:
            return [text]
        step = _WINDOW_CHARS - _WINDOW_OVERLAP
        wins = [text[i:i + _WINDOW_CHARS] for i in range(0, len(text), step)]
        return wins[:_MAX_WINDOWS]

    def _injection_prob(self, text: str) -> float:
        np = self._np
        enc = self._tok.encode(text)
        ids = np.array([enc.ids], dtype=np.int64)
        mask = np.array([enc.attention_mask], dtype=np.int64)
        (logits,) = self._sess.run(["logits"], {"input_ids": ids, "attention_mask": mask})
        row = logits[0].astype(np.float64)
        ex = np.exp(row - row.max())
        return float((ex / ex.sum())[1])  # P(INJECTION)

    def scan_text(self, text: str) -> Verdict:
        if not text:
            return Verdict.clean()
        score = max(self._injection_prob(w) for w in self._windows(text))
        return Verdict(
            flagged=score >= FLAG_THRESHOLD,
            score=score,
            reasons=[f"deberta:{score:.4f}"] if score >= FLAG_THRESHOLD else [],
            hard=score >= HARD_THRESHOLD,
        )


@lru_cache(maxsize=1)
def _guard() -> DebertaGuard:
    return DebertaGuard()


def warmup() -> None:
    """Load the model + run a tiny inference now (avoids a slow first request)."""
    _guard().scan_text("warmup")


def scan_text(text: str) -> Verdict:
    """Score one piece of untrusted text with the classifier.

    Coverage is bounded at ``_MAX_WINDOWS`` windows per item (see the module docstring);
    content past that is not scored.
    """
    return _guard().scan_text(text)
