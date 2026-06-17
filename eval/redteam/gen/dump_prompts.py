"""Dump garak probe prompts → raw JSON. **Runs inside garak's isolated venv** (via uvx).

garak is used purely as a *payload source*: we instantiate each curated probe and read its
``.prompts`` (the strings garak would send a target), never running garak's attack→response
loop or needing a target model. Output is a flat JSON mapping ``probe -> [prompt, ...]`` that
``export.py`` (in agentgate's venv) turns into corpus items. The only import shared across
the venv boundary is ``probe_map`` — loaded here by file path so we never trigger agentgate's
package ``__init__`` chain (pydantic isn't installed in garak's venv).

Usage (see run-garak.sh):
    uvx --from garak python eval/redteam/gen/dump_prompts.py <out.json>
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path


def _load_probe_map():
    """Import probe_map by path — avoids pulling in eval.redteam's pydantic-laden siblings."""
    pm_path = Path(__file__).with_name("probe_map.py")
    spec = importlib.util.spec_from_file_location("probe_map", pm_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: dump_prompts.py <out.json>")
    out_path = Path(sys.argv[1])

    import garak._config as cfg

    cfg.load_base_config()
    pm = _load_probe_map()

    dump: dict[str, list[str]] = {}
    for probe in pm.PROBES:
        module, cls_name = probe.split(".", 1)
        mod = importlib.import_module(f"garak.probes.{module}")
        instance = getattr(mod, cls_name)(config_root=cfg)
        prompts = [str(p) for p in getattr(instance, "prompts", [])]
        picked = [prompts[i] for i in pm.stride_indices(len(prompts))]
        dump[probe] = picked
        print(f"  {probe}: {len(prompts)} → {len(picked)}", file=sys.stderr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dump, ensure_ascii=False, indent=2))
    total = sum(len(v) for v in dump.values())
    print(f"wrote {total} prompts across {len(dump)} probes → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
