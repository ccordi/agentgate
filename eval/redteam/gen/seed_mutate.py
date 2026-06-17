"""Seed-and-mutate harness — generate adversarial items into the **same** corpus.

Generated attacks become ``corpus/public/seed_mutate.jsonl``, auto-globbed by
``loader.default_sources()`` (no loader edit), then scored by the **real** guard via
the unchanged ``score``/``report`` chain.

Two product lines, mirroring the garak ``expected_miss`` split so the report reads
consistently:

  * **Headline — indirect / long-context (``expected_miss=False``).** The attacker model
    (``attacker.py``) expands agent-specific seeds (``seeds.py``) into document-embedded and
    tool-result attacks. This is the axis the garak evaluation found the guard *misses* — a
    miss here is real signal, so it is left untagged.
  * **Control — obfuscation (``expected_miss=True``).** Canonical instructions run through the
    in-house converters (``converters.py``). The garak evaluation showed DeBERTa mostly catches
    these; tagged as a documented tokenization blind spot, like garak's ``encoding.*``.

Mutation-validity: converter items are lossless → valid by construction. Attacker items are
paraphrases that *could* be watered down, so (if a judge is configured via ``AGENTGATE_JUDGE_*``)
they are validity-checked and benign-judged ones dropped; otherwise kept with
``meta.validity="unchecked"`` and a caveat in the writeup.

The committed JSONL is the durable artifact (the attacker is non-deterministic — same stance as
the garak corpus). Reproduce:  ``uv run python -m eval.redteam.gen.seed_mutate``
"""

from __future__ import annotations

import argparse
import asyncio

import httpx

from ..loader import PUBLIC_DIR, load_jsonl, write_jsonl
from ..schema import CorpusItem, LabelOrigin
from . import attacker as attacker_mod
from .converters import CONVERTERS
from .seeds import CONTROL_INSTRUCTIONS, SEEDS, VARIANTS_PER_SEED, Seed

OUT = PUBLIC_DIR / "seed_mutate.jsonl"


def _is_control(item: CorpusItem) -> bool:
    """Control = obfuscation items (valid by construction, never re-judged)."""
    return item.meta.get("validity") == "by_construction" or "converter" in item.meta


def build_attacker_items(pairs: list[tuple[Seed, str]]) -> list[CorpusItem]:
    """(seed, attack-text) → headline CorpusItems (indirect/long-context; expected_miss=False)."""
    items: list[CorpusItem] = []
    for seed, text in pairs:
        items.append(CorpusItem(
            source="seed_mutate",
            text=text,
            label=1,
            label_origin=LabelOrigin.KNOWN,
            category=f"seedmutate:{seed.axis}",
            meta={"seed_id": seed.id, "axis": seed.axis, "expected_miss": False,
                  "validity": "unchecked"},
        ))
    return items


def build_control_items() -> list[CorpusItem]:
    """CONTROL_INSTRUCTIONS × CONVERTERS → obfuscation control items (expected_miss=True)."""
    items: list[CorpusItem] = []
    for name, fn in CONVERTERS.items():
        for instr in CONTROL_INSTRUCTIONS:
            items.append(CorpusItem(
                source="seed_mutate",
                text=fn(instr),
                label=1,
                label_origin=LabelOrigin.KNOWN,
                category=f"seedmutate:enc_{name}",
                meta={"converter": name, "expected_miss": True, "validity": "by_construction"},
            ))
    return items


async def _validity_filter(items: list[CorpusItem]) -> list[CorpusItem]:
    """Drop attacker items a configured judge labels benign (payload was neutralized).

    No-op (returns items unchanged, validity stays 'unchecked') if no judge is configured —
    the refusal heuristic in ``attacker.generate`` already removed empty/hedged outputs.
    """
    from .. import judge as judge_mod

    cfg = judge_mod.JudgeConfig()
    if not cfg.configured:
        print("  (judge not configured → skipping mutation-validity check; "
              "items kept with validity='unchecked')")
        return items
    print(f"  validity-checking {len(items)} attacker items with judge {cfg.model}…")
    # use_cache=False on purpose: this is a *generation-time* gate, not part of the labeling
    # chain. Writing these (all known-positive) into the shared judge cache would make
    # `report`'s judge-validation section render a degenerate all-positive κ.
    labels = await judge_mod.judge_corpus(items, cfg=cfg, use_cache=False)
    kept: list[CorpusItem] = []
    for it in items:
        lab = labels.get(it.id)
        verdict = "valid" if (lab and lab.label == 1) else "neutralized"
        it.meta["validity"] = verdict
        if verdict == "valid":
            kept.append(it)
    print(f"  validity: {len(kept)}/{len(items)} attacks survived (rest neutralized → dropped)")
    return kept


async def generate_corpus(variants_per_seed: int = VARIANTS_PER_SEED) -> list[CorpusItem]:
    cfg = attacker_mod.AttackerConfig()
    print(f"attacker: {cfg.model} @ {cfg.base_url} "
          f"(key {'set' if cfg.configured else 'MISSING'})")
    async with httpx.AsyncClient(timeout=cfg.timeout_s) as client:
        pairs = await attacker_mod.generate(SEEDS, variants_per_seed, cfg=cfg, client=client)
    print(f"generated {len(pairs)} attacker variants from {len(SEEDS)} seeds "
          f"(after refusal/dup filter)")
    attacker_items = await _validity_filter(build_attacker_items(pairs))
    control_items = build_control_items()
    print(f"control: {len(control_items)} obfuscation items "
          f"({len(CONVERTERS)} converters × {len(CONTROL_INSTRUCTIONS)} instructions)")
    return attacker_items + control_items


async def revalidate_corpus() -> list[CorpusItem]:
    """Re-judge the *existing* committed attacks in place — without regenerating them.

    The validity judge is otherwise coupled to generation (``generate_corpus``), so swapping the
    judge model would also draw a new, non-deterministic attack set and you couldn't isolate the
    judge's effect. This re-runs the (now reconfigured) ``AGENTGATE_JUDGE_*`` judge over the
    attacker items already in ``seed_mutate.jsonl``, refreshing ``meta.validity`` and dropping any
    the new judge calls neutralized. Control items (valid by construction) pass through untouched.
    """
    if not OUT.exists():
        raise SystemExit(f"{OUT} not found — run generation first.")
    existing = list(load_jsonl(OUT))
    control = [i for i in existing if _is_control(i)]
    attacker = [i for i in existing if not _is_control(i)]
    print(f"revalidating {len(attacker)} attacker items ({len(control)} control untouched)")
    kept = await _validity_filter(attacker)
    return kept + control


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m eval.redteam.gen.seed_mutate")
    ap.add_argument("-n", "--variants", type=int, default=VARIANTS_PER_SEED,
                    help=f"attacker variants per seed (default {VARIANTS_PER_SEED})")
    ap.add_argument("--revalidate", action="store_true",
                    help="re-judge the existing committed attacks with the current "
                         "AGENTGATE_JUDGE_* model (no regeneration); for swapping judge family")
    args = ap.parse_args()
    items = asyncio.run(revalidate_corpus() if args.revalidate
                        else generate_corpus(args.variants))
    n = write_jsonl(OUT, items)
    em = sum(1 for i in items if i.meta.get("expected_miss"))
    print(f"wrote {n} items → {OUT}  ({em} control/expected_miss / {n - em} headline)")


if __name__ == "__main__":
    main()
