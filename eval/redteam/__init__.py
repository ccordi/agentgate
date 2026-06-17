"""Red-team harness — output written to ``docs/redteam-results.md`` by the ``report`` command.

Measures the injection scanner's honest catch-rate / false-positive table via a defensible
labeling chain:

    known-source labels  →  independent LLM judge (different family than the scanner)
                         →  versioned human gold set validating the judge (agreement + κ)

Two traps this is built to avoid:
  * **Circularity** — never treat the scanner's own verdict as ground truth.
  * **False-negative trap** — measure recall against *labeled* positives, and validate on a
    sample of *all* content (flagged + unflagged), or the catch-rate is a lie.

The harness drives the stable entry point ``agentgate.security.injection.scan_text`` so the
same evaluation applies unchanged when the scanner is swapped for a different backend.
"""
