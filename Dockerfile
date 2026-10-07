# Image for the gateway and, in the Docker setup, the mock model server (same image,
# different command). It sets no gateway settings: deploy/docker-compose.yml chooses the
# provider, routing, scanner and container bind.
FROM python:3.13-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependency layer first so source edits don't re-resolve. README.md is a build
# input: pyproject declares it, so the project wheel can't build without it.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --extra tracing --extra egress-mcp
COPY src/ src/
COPY bench/ bench/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra tracing --extra egress-mcp

FROM python:3.13-slim
# Non-root. /app/data backs the default SQLite audit store
# (sqlite+aiosqlite:///./data/agentgate.db, relative to the workdir).
RUN groupadd -r -g 10001 agentgate && useradd -r -u 10001 -g agentgate agentgate \
    && mkdir -p /app/data && chown -R agentgate:agentgate /app
WORKDIR /app
COPY --from=builder --chown=agentgate:agentgate /app /app
ENV PATH="/app/.venv/bin:$PATH"
USER agentgate
EXPOSE 4100
# stdlib probe — the slim image has no curl. Probes the gateway; the mock service
# disables this healthcheck (it shares the image but not the port).
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:4100/healthz', timeout=2).status == 200 else 1)"]
CMD ["agentgate"]
