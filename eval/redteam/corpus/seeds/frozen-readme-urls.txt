# Seed sources for the adversarial-evaluation generation engine — NOT a live crawl list.
#
# These four repos are FROZEN historical corpora. They were formerly used as live capture
# targets; that approach is retired. They are kept here only as *seed material* the generation
# engine may rewrite/mutate — import-or-not is decided when the seed-and-mutate harness is built.
#
# To use: fetch out-of-band (a plain HTTP GET, not an agent), extract payload strings, feed as
# seeds. There is no live agent and no scheduled fetch.

https://raw.githubusercontent.com/greshake/llm-security/main/README.md
https://raw.githubusercontent.com/verazuo/jailbreak_llms/main/README.md
https://raw.githubusercontent.com/lakeraai/pint-benchmark/main/README.md
https://raw.githubusercontent.com/elder-plinius/L1B3RT4S/main/README.md
