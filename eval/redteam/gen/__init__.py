"""Adversarial *generation* into the red-team corpus (offline, same-machine).

This package turns external attack generators into corpus JSONL that the existing
``eval.redteam`` scoring chain (``score`` → ``report``) consumes **unchanged**. garak runs in
its own isolated venv (zero footprint on agentgate's lock); the handoff to agentgate is a
JSON file, never a shared import. See ``probe_map.py`` for the curated probe set and the
``expected_miss`` split, and ``run-garak.sh`` for the reproducer.

Two generators live here, both feeding the same corpus:
  * **garak** (``probe_map``/``dump_prompts``/``export``): breadth, the regression baseline.
    Runs garak probes and exports them into the corpus JSONL. See ``run-garak.sh``.
  * **seed-and-mutate** (``seeds``/``attacker``/``converters``/``seed_mutate``): a local
    attacker model rewrites agent-specific seeds into *indirect / document-embedded /
    tool-result* attacks (the blind spot garak's evaluation found), plus an obfuscation
    **control** axis. Reproduce with ``uv run python -m eval.redteam.gen.seed_mutate``
    (needs ``AGENTGATE_ATTACKER_*``).
"""
