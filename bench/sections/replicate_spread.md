## Is this one run representative?

The tables above are a single `bench.run`. One run can't tell you whether its numbers are
typical, so the gateway-overhead figure was re-measured at **N=6** on 2026-07-28 — six
full runs, each ~5,400 gateway requests, every replicate DB kept and re-percentiled with
`bench.stats` (the same code that renders the tables):

| replicate | n | gateway internal p95 | p99 |
| --- | --- | --- | --- |
| 1 | 5403 | 0.95 | 1.02 |
| 2 | 5403 | 1.00 | 1.05 |
| 3 | 5403 | 1.01 | 1.07 |
| 4 | 5402 | 1.04 | 1.09 |
| 5 | 5404 | 1.00 | 1.06 |
| 6 | 5401 | 0.95 | 1.00 |
| **median** | | **1.00** | **1.06** |

Spread is about ±4%. The tables above use replicate 5, which is at the median on both
percentiles. Every replicate ran the `heuristic` backend, confirmed per request from the
audit rows.

**This supersedes an earlier figure.** Through 2026-07-27 this document reported **1.48 ms
p95 / 1.70 ms p99** from a 2026-06-17 run. That pair does not reproduce: it sits outside
the range of all six replicates, 43% above the highest one. No artifact from the original
run survived, so the cause cannot be traced. The new replicates still support sub-2 ms
proxy overhead with the regex guard; the earlier pair should not be quoted.

The six-run spread is what shows that 1.48 ms is outside the reproduced distribution.
