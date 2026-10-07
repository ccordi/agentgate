"""Entry point: `agentgate` / `python -m agentgate`.

    agentgate                 # serve (the default)
    agentgate serve           # the same thing, said out loud
    agentgate --env-file PATH # serve with an explicit settings file
    agentgate init            # create a model-free starter .env
    agentgate audit …         # read the audit trail (see audit/cli.py)
"""

from __future__ import annotations

import argparse
import logging
import os
import secrets
import sys
from pathlib import Path

_STARTER_ENV = """\
# Model-free starter configuration for agentgate.
# This uses the built-in heuristic scanner and the mock model server, so it needs no
# model files or provider credentials. Start the mock model server separately, before
# the gateway.
AGENTGATE_ADMIN_TOKEN={admin_token}
AGENTGATE_PDP_TOKEN={pdp_token}
AGENTGATE_GUARD_BACKEND=heuristic
AGENTGATE_DEFAULT_PROVIDER=mock
AGENTGATE_ROUTING__ENABLED=false
"""


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
    #
    # Deliberately inside `serve()`, not `main()`: what this validates is the posture for
    # *listening* — the loopback bind and the two gateway tokens. `agentgate audit` opens
    # the database and serves nothing, so it needs neither the bind check nor the tokens.
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


def initialize(directory: Path) -> None:
    """Create a minimal starter `.env` without loading the application runtime."""
    target_dir = directory.absolute()
    if not target_dir.exists():
        print(f"agentgate init: directory does not exist: {target_dir}", file=sys.stderr)
        raise SystemExit(2)
    if not target_dir.is_dir():
        print(f"agentgate init: not a directory: {target_dir}", file=sys.stderr)
        raise SystemExit(2)

    env_path = target_dir / ".env"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    previous_umask = os.umask(0o077)
    try:
        try:
            fd = os.open(env_path, flags, 0o600)
        except FileExistsError as exc:
            print(f"agentgate init: refusing to overwrite existing {env_path}", file=sys.stderr)
            raise SystemExit(2) from exc
        except OSError as exc:
            print(f"agentgate init: cannot create {env_path}: {exc.strerror}", file=sys.stderr)
            raise SystemExit(2) from exc
    finally:
        os.umask(previous_umask)

    contents = _STARTER_ENV.format(
        admin_token=secrets.token_urlsafe(32),
        pdp_token=secrets.token_urlsafe(32),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as env_file:
            env_file.write(contents)
    except OSError as exc:
        print(f"agentgate init: cannot write {env_path}: {exc.strerror}", file=sys.stderr)
        raise SystemExit(2) from exc

    print(f"Created {env_path}")
    print("Next: follow docs/getting-started.md to start the mock upstream and gateway.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentgate", description=__doc__.split("\n")[0])
    parser.add_argument(
        "--env-file", type=Path,
        help="settings file for serve or audit (place before the command; default: .env)",
    )
    parser.add_argument(
        "command", nargs="?", default="serve", choices=["serve", "init", "audit"],
        help="serve (the default), create starter configuration, or read the audit trail",
    )
    parser.add_argument("args", nargs=argparse.REMAINDER, help="arguments for the subcommand")
    return parser


def build_init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentgate init",
        description="Create a model-free starter .env without overwriting an existing file.",
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path.cwd(),
        help="existing directory in which to create .env (default: current directory)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        serve()  # bare `agentgate` starts the server
        return

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "init":
        if args.env_file is not None:
            parser.error("--env-file selects runtime settings; use init --directory for output")
        init_args = build_init_parser().parse_args(args.args)
        initialize(init_args.directory)
        return
    if args.command == "serve" and args.args:
        parser.error(
            f"unrecognized arguments: {' '.join(args.args)}; place --env-file before serve"
        )
    if args.env_file is not None:
        if not args.env_file.is_file():
            parser.error(f"env file is not a regular file: {args.env_file}")
        try:
            with args.env_file.open("rb"):
                pass
        except OSError as exc:
            parser.error(f"cannot read env file {args.env_file}: {exc.strerror}")

        from agentgate.config import Settings, get_settings
        from agentgate.guards.local_llm import JudgeConfig

        # Settings, judge settings and point-of-use readers use the same file.
        env_file = args.env_file.absolute()
        Settings.model_config["env_file"] = env_file
        JudgeConfig.model_config["env_file"] = env_file
        get_settings.cache_clear()

    if args.command == "audit":
        from agentgate.audit import cli
        cli.main(args.args)
    else:
        serve()


if __name__ == "__main__":
    main()
