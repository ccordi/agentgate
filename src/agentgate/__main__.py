"""Entry point: `agentgate` / `python -m agentgate`."""

from __future__ import annotations

import logging

import uvicorn

from agentgate.config import get_settings


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    settings = get_settings()
    uvicorn.run(
        "agentgate.app:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
