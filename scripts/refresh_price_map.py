"""Refresh the vendored LiteLLM price map (src/agentgate/data/model_prices.json).

The default source is the copy bundled inside the *installed* litellm package
(`model_prices_and_context_window_backup.json`), located by path via
`importlib.util.find_spec` — never by importing litellm: a stock
`import litellm` fetches the live map from GitHub at import time, and no
default path of this script touches the network. The bundled file is only
present with the `litellm-plugin` extra (`uv sync --extra litellm-plugin`).

Sources:
    (default)          the installed litellm package's bundled copy
    --from-file PATH   a local JSON file (offline; also what the tests use)
    --from-url URL     fetch over the network — explicit opt-in, never default

The candidate map is validated (parses, top-level dict, enough model entries
with the expected price fields), then written verbatim — bytes, not
re-serialized, so diffs against upstream stay clean — and a summary of added /
removed / re-priced models is printed. The "Last refreshed" line of the
adjacent PROVENANCE file is updated when one exists next to the output path.

Usage:
    uv run python scripts/refresh_price_map.py
    uv run python scripts/refresh_price_map.py --dry-run
    uv run python scripts/refresh_price_map.py --from-file path/to/map.json
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import re
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENDORED = REPO_ROOT / "src" / "agentgate" / "data" / "model_prices.json"
BUNDLED_NAME = "model_prices_and_context_window_backup.json"

# Meta keys that are not model entries (mirrors src/agentgate/pricing.py).
RESERVED_KEYS = frozenset({"sample_spec", "fallback_generalizations"})

PRICE_FIELDS = ("input_cost_per_token", "output_cost_per_token")


def locate_bundled_map() -> Path:
    """Path of the installed litellm package's bundled price map — without importing it."""
    spec = importlib.util.find_spec("litellm")
    if spec is None or not spec.origin:
        sys.exit(
            "litellm is not installed in this environment. The refresh script reads the\n"
            "bundled price map from the installed package; install the extra first:\n"
            "    uv sync --extra litellm-plugin\n"
            "or pass an explicit source with --from-file / --from-url."
        )
    path = Path(spec.origin).parent / BUNDLED_NAME
    if not path.is_file():
        sys.exit(f"installed litellm package has no {BUNDLED_NAME} at {path}")
    return path


def validate(raw_bytes: bytes) -> dict:
    """Parse and sanity-check a candidate map; exits with a message on failure."""
    try:
        data = json.loads(raw_bytes)
    except ValueError as exc:
        sys.exit(f"candidate map is not valid JSON: {exc}")
    if not isinstance(data, dict):
        sys.exit(f"candidate map top level is {type(data).__name__}, expected an object")
    models = {k: v for k, v in data.items() if k not in RESERVED_KEYS and isinstance(v, dict)}
    priced = sum(
        1
        for v in models.values()
        if all(isinstance(v.get(f), int | float) for f in PRICE_FIELDS)
    )
    # Loose floors: catch a truncated or wrong file, not normal upstream drift.
    if len(models) < 1000 or priced < 500:
        sys.exit(
            f"candidate map looks wrong: {len(models)} model entries, only {priced} with "
            f"both price fields — refusing to overwrite the vendored copy"
        )
    return data


def prices(entry: dict) -> tuple:
    return tuple(entry.get(f) for f in PRICE_FIELDS)


def summarize(old: dict | None, new: dict) -> tuple[str, bool]:
    """Human summary of what changes; second value says whether anything did."""
    new_models = {k for k in new if k not in RESERVED_KEYS}
    if old is None:
        return f"no existing vendored map — vendoring {len(new_models)} model entries", True
    old_models = {k for k in old if k not in RESERVED_KEYS}
    added = sorted(new_models - old_models)
    removed = sorted(old_models - new_models)
    repriced = sorted(
        k
        for k in new_models & old_models
        if isinstance(new[k], dict)
        and isinstance(old[k], dict)
        and prices(new[k]) != prices(old[k])
    )
    lines = [f"added: {len(added)}   removed: {len(removed)}   price changed: {len(repriced)}"]
    for label, ids in (("added", added), ("removed", removed), ("price changed", repriced)):
        for m in ids[:10]:
            if label == "price changed":
                lines.append(f"  ~ {m}: {prices(old[m])} -> {prices(new[m])}")
            else:
                sign = "+" if label == "added" else "-"
                lines.append(f"  {sign} {m}")
        if len(ids) > 10:
            lines.append(f"  ... and {len(ids) - 10} more {label}")
    return "\n".join(lines), bool(added or removed or repriced)


def update_provenance(out_path: Path, source_desc: str) -> None:
    prov = out_path.with_name("model_prices.PROVENANCE.md")
    if not prov.is_file():
        return
    text = prov.read_text()
    line = f"Last refreshed: {date.today().isoformat()} from {source_desc}"
    new_text, n = re.subn(r"^Last refreshed:.*$", line, text, count=1, flags=re.MULTILINE)
    if n == 0:
        new_text = text.rstrip("\n") + "\n\n" + line + "\n"
    prov.write_text(new_text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--from-file", metavar="PATH", help="read the map from a local JSON file")
    src.add_argument(
        "--from-url", metavar="URL",
        help="fetch the map over the network (explicit opt-in; never the default)",
    )
    parser.add_argument(
        "--out", metavar="PATH", default=str(VENDORED),
        help=f"where to write the vendored map (default: {VENDORED})",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the summary but write nothing",
    )
    args = parser.parse_args()

    if args.from_file:
        source_path = Path(args.from_file)
        if not source_path.is_file():
            sys.exit(f"no such file: {source_path}")
        raw = source_path.read_bytes()
        source_desc = str(source_path)
    elif args.from_url:
        import httpx  # only the explicit network path needs it

        resp = httpx.get(args.from_url, timeout=60, follow_redirects=True)
        resp.raise_for_status()
        raw = resp.content
        source_desc = args.from_url
    else:
        source_path = locate_bundled_map()
        raw = source_path.read_bytes()
        try:
            version = importlib.metadata.version("litellm")
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
        source_desc = f"installed litellm=={version} bundled copy"

    new_data = validate(raw)

    out_path = Path(args.out)
    old_data: dict | None = None
    if out_path.is_file():
        try:
            old_data = json.loads(out_path.read_bytes())
        except ValueError:
            print(f"warning: existing {out_path} does not parse; treating as absent")

    summary, changed = summarize(old_data, new_data)
    print(f"source: {source_desc}")
    print(summary)

    if args.dry_run:
        print("dry run — nothing written")
        return
    if not changed and out_path.is_file() and out_path.read_bytes() == raw:
        print("no changes — vendored map already up to date")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(raw)  # verbatim: keeps diffs against upstream clean
    update_provenance(out_path, source_desc)
    print(f"wrote {out_path} ({len(raw):,} bytes)")


if __name__ == "__main__":
    main()
