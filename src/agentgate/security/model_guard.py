"""Model-backed injection guard — ONNX DeBERTa classifier behind the `scan_text` interface.

An interchangeable alternative to the heuristic `injection.scan_text`: same signature, same
`Verdict` return type, so the red-team harness and (eventually) the gateway hot path can
swap detectors with no other changes. Runs **in-process** on onnxruntime (no torch, no model
server, no network) — a guard must not egress content.

Default model: `protectai/deberta-v3-base-prompt-injection-v2` (ONNX export), a ~184M
sequence classifier with labels {0: SAFE, 1: INJECTION}. The injection probability becomes
`Verdict.score`. Inputs cap at 512 tokens, so long tool outputs are **windowed** and the max
window score is taken (an injection anywhere in the content trips it).

Lazy-loaded singleton; the `guard` extra (onnxruntime, tokenizers, numpy) must be installed.
Model dir is overridable via `AGENTGATE_GUARD_MODEL_DIR`.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from .injection import FLAG_THRESHOLD, HARD_THRESHOLD, Verdict, extract_untrusted

_DEFAULT_DIR = "models/protectai-deberta-v3-base-prompt-injection-v2/onnx"
_MAX_TOKENS = 512
_WINDOW_CHARS = 1800   # ≈ <512 tokens; conservative
_WINDOW_OVERLAP = 200  # keep injections that straddle a window boundary
_MAX_WINDOWS = 16       # bound work on pathologically long inputs


class DebertaGuard:
    def __init__(self, model_dir: str | None = None) -> None:
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self._np = np
        d = Path(model_dir or os.getenv("AGENTGATE_GUARD_MODEL_DIR", _DEFAULT_DIR))
        if not (d / "model.onnx").exists():
            raise FileNotFoundError(
                f"guard model not found at {d}/model.onnx — set AGENTGATE_GUARD_MODEL_DIR "
                "or pull protectai/deberta-v3-base-prompt-injection-v2 (onnx)."
            )
        self._tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self._tok.enable_truncation(max_length=_MAX_TOKENS)
        self._sess = ort.InferenceSession(str(d / "model.onnx"), providers=["CPUExecutionProvider"])

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
            reasons=[f"deberta:{score:.2f}"] if score >= FLAG_THRESHOLD else [],
            hard=score >= HARD_THRESHOLD,
        )


@lru_cache(maxsize=1)
def _guard() -> DebertaGuard:
    return DebertaGuard()


def warmup() -> None:
    """Load the model + run a tiny inference now (avoids a slow first request)."""
    _guard().scan_text("warmup")


def scan_text(text: str) -> Verdict:
    """Drop-in replacement for ``injection.scan_text`` backed by the DeBERTa classifier."""
    return _guard().scan_text(text)


def scan_request(messages: list[dict]) -> Verdict:
    """Mirror ``injection.scan_request``: strongest verdict over the untrusted content."""
    worst = Verdict.clean()
    for source, text in extract_untrusted(messages):
        v = scan_text(text)
        if v.score > worst.score:
            worst = Verdict(v.flagged, v.score, [f"{source}:{r}" for r in v.reasons], v.hard)
    return worst
