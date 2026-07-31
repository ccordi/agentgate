"""Independent LLM-as-judge — labels corpus items for the methodology chain.

Runs **offline** (not in the gateway hot path) against already-captured/authored data. Uses
a *different model family than the scanner* so judge errors aren't correlated with scanner
errors (see the labeling-chain design notes). Default judge = Gemini via its
OpenAI-compatible endpoint; written against the plain OpenAI Chat Completions contract so
swapping to Claude/GPT is a config change.

Credentials: key/model/base_url come from ``AGENTGATE_JUDGE_*`` env vars (the harness
reads its own key, separate from any client credentials). The key is **never logged** —
only its presence is reported.
"""

from __future__ import annotations

import asyncio

import httpx

from agentgate.guards.local_llm import (
    JUDGE_SYSTEM_PROMPT,
    JudgeCache,
    JudgeConfig,
    JudgeLabel,
    LLMGuard,
    parse_judge_label,
)

# LLMGuard is re-exported, not used here: `__main__`'s llm-guard detector and the
# eval-side guard tests reach it through this module.
__all__ = ["CACHE_PATH", "JudgeConfig", "JudgeLabel", "LLMGuard", "cached_labels",
           "judge_corpus", "run_judge"]

from .common import chat_completion
from .loader import RUNS_DIR
from .schema import CorpusItem

# The eval cache is separate from the guard's runtime cache, so live traffic and
# measurement runs do not share an artifact.
CACHE_PATH = RUNS_DIR / "judge_cache.json"


async def _judge_one(client: httpx.AsyncClient, cfg: JudgeConfig, item: CorpusItem) -> JudgeLabel:
    content = await chat_completion(
        client, cfg.base_url, cfg.api_key, cfg.model,
        [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": item.text},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return parse_judge_label(content, cfg.model)


async def judge_corpus(
    items: list[CorpusItem],
    *,
    cfg: JudgeConfig | None = None,
    client: httpx.AsyncClient | None = None,
    use_cache: bool = True,
) -> dict[str, JudgeLabel]:
    """Label items with the judge. Cached by ``model:id``; concurrency-limited.

    ``client`` is injectable so tests can pass an ``httpx.MockTransport`` (no live calls).
    """
    cfg = cfg or JudgeConfig()
    if not cfg.configured and client is None:
        raise RuntimeError(
            "judge not configured: set AGENTGATE_JUDGE_API_KEY (and optionally "
            "AGENTGATE_JUDGE_MODEL / AGENTGATE_JUDGE_BASE_URL)."
        )

    cache = JudgeCache.load(CACHE_PATH) if use_cache else JudgeCache(data={}, path=CACHE_PATH)
    results: dict[str, JudgeLabel] = {}
    todo: list[CorpusItem] = []
    for it in items:
        hit = cache.data.get(JudgeCache.key(cfg.model, it.id))
        if use_cache and hit is not None:
            results[it.id] = JudgeLabel(**hit)
        else:
            todo.append(it)

    if todo:
        owns_client = client is None
        client = client or httpx.AsyncClient(timeout=cfg.timeout_s)
        sem = asyncio.Semaphore(cfg.max_concurrency)

        async def worker(it: CorpusItem) -> tuple[str, JudgeLabel]:
            async with sem:
                return it.id, await _judge_one(client, cfg, it)

        try:
            for coro in asyncio.as_completed([worker(it) for it in todo]):
                item_id, label = await coro
                results[item_id] = label
                cache.data[JudgeCache.key(cfg.model, item_id)] = label.as_dict()
        finally:
            if owns_client:
                await client.aclose()
        if use_cache:
            cache.save()

    return results


def run_judge(items: list[CorpusItem], **kw) -> dict[str, JudgeLabel]:
    """Sync entry point for the CLI."""
    return asyncio.run(judge_corpus(items, **kw))


def cached_labels(model: str) -> dict[str, JudgeLabel]:
    """Read previously-judged labels for ``model`` from the cache (no API calls).

    Used by the report step so rendering never hits the network.
    """
    cache = JudgeCache.load(CACHE_PATH)
    prefix = f"{model}:"
    out: dict[str, JudgeLabel] = {}
    for key, val in cache.data.items():
        if not key.startswith(prefix):
            continue
        item_id = key[len(prefix):]
        # Guard-written keys are `{model}:guard-<tag>:{id}` — same prefix, different shape.
        # Skip them so a shared cache file can't produce garbage ids here.
        if item_id.startswith("guard-"):
            continue
        out[item_id] = JudgeLabel(**val)
    return out
