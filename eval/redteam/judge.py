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

from agentgate.security.llm_guard import (
    _CACHE_PATH as _CACHE_PATH,
)
from agentgate.security.llm_guard import (
    _SYSTEM as _SYSTEM,
)
from agentgate.security.llm_guard import (
    JudgeConfig as JudgeConfig,
)
from agentgate.security.llm_guard import (
    JudgeLabel as JudgeLabel,
)
from agentgate.security.llm_guard import (
    LLMGuard as LLMGuard,
)
from agentgate.security.llm_guard import (
    _Cache as _Cache,
)
from agentgate.security.llm_guard import (
    _parse_label as _parse_label,
)
from agentgate.security.llm_guard import (
    scan_text as scan_text,
)

from .schema import CorpusItem


async def _judge_one(client: httpx.AsyncClient, cfg: JudgeConfig, item: CorpusItem) -> JudgeLabel:
    resp = await client.post(
        f"{cfg.base_url}/chat/completions",
        headers={"Authorization": f"Bearer {cfg.api_key}"},
        json={
            "model": cfg.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": item.text},
            ],
        },
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return _parse_label(content, cfg.model)


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

    cache = _Cache.load() if use_cache else _Cache()
    results: dict[str, JudgeLabel] = {}
    todo: list[CorpusItem] = []
    for it in items:
        hit = cache.data.get(_Cache.key(cfg.model, it.id))
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
                cache.data[_Cache.key(cfg.model, item_id)] = label.as_dict()
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
    cache = _Cache.load()
    prefix = f"{model}:"
    out: dict[str, JudgeLabel] = {}
    for key, val in cache.data.items():
        if key.startswith(prefix):
            out[key[len(prefix):]] = JudgeLabel(**val)
    return out
