"""Entry point: `agentgate` / `python -m agentgate`.

    agentgate                 # serve (unchanged default)
    agentgate serve           # the same thing, said out loud
    agentgate audit …         # read the audit trail (see audit/cli.py)
"""

from __future__ import annotations

import argparse
import logging
import sys


def serve() -> None:
    import uvicorn  # imported lazily so `agentgate audit` doesn't pay for the server

    from agentgate.config import get_settings, validate_runtime_settings

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    settings = get_settings()
    # Also enforced in the app's lifespan (covers `uvicorn agentgate.app:app`); checked
    # here first for a clean one-line CLI error instead of a startup traceback.
    try:
        validate_runtime_settings(settings)
    except RuntimeError as exc:
        print(f"agentgate: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    uvicorn.run(
        "agentgate.app:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentgate", description=__doc__.split("\n")[0])
    parser.add_argument(
        "command", nargs="?", default="serve", choices=["serve", "audit"],
        help="serve (the default when no command is given) or audit",
    )
    parser.add_argument("args", nargs=argparse.REMAINDER, help="arguments for the subcommand")
    return parser


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        serve()  # bare `agentgate` starts the server — unchanged
        return

    args = build_parser().parse_args(argv)
    if args.command == "audit":
        from agentgate.audit import cli
        cli.main(args.args)
    else:
        serve()


if __name__ == "__main__":
    main()
