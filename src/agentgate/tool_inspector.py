"""Static analysis of tools[] definitions (tool-catalog injection surface).

Covers the prompt-injection-via-tool-catalog attack surface (distinct from the
injection guard, which scans untrusted *content* in messages).  MCP tools appear
as ordinary function entries in `tools[]`; this module covers them too.

The screened surface is everything the model's catalog renders: the tool's own
`description`, and the `description`/`enum` text nested anywhere in its
`parameters` schema.  The caller is responsible for handing over the legacy
`functions[]` array alongside `tools[]` — see `pipeline._parse_body`.

Two-tier enforcement:

  HARD (high precision) — description-level instruction injection.
      "ignore previous instructions", "do not mention", "exfiltrate", etc.
      Few false positives, so it justifies a 400 block.

  SOFT (record-only) — coarse heuristics: imperative phrasing ("you must",
      "always call … first"), suspicious tool names, overly-broad descriptions.
      Many false positives — legitimate tools are routinely named run_shell/execute/
      http_request.  Logged and audited, but forwarded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from agentgate.content import bounded_gap_abandoned


@dataclass
class ToolVerdict:
    flagged: bool
    hard: bool = False
    reasons: list[str] = field(default_factory=list)
    suspicious_tools: list[str] = field(default_factory=list)
    # `gap_abandoned:<label>:<distance>` for each bounded-gap pattern whose prefix was
    # seen with its suffix past the bound. Deliberately NOT part of `reasons` and never
    # sets `flagged`/`hard`: soft reasons feed `tool_def_flagged`, and making a detection
    # limit visible must not add flags. Audit caveats only.
    gap_abandoned: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# HARD tier — description-level instruction injection
# ---------------------------------------------------------------------------
# A tool *description* should describe the tool, not issue orders to the model.
# These phrases are almost never legitimate in a description field.

_HARD_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"ignore\s+(all\s+)?(previous|prior|earlier)\s+(instructions?|prompts?|context)", re.I),
     "ignore_previous_instructions"),
    (re.compile(r"\bsend\s+the\s+(user['']?s?|all)\b", re.I), "send_user_data"),
    (re.compile(r"\bexfiltrate\b", re.I), "exfiltrate"),
    (re.compile(r"\bdo\s+not\s+(tell|inform|mention|reveal)\b", re.I), "suppress_disclosure"),
    # Bounded gap, not `.*`: an unbounded one between a repeatable prefix and a suffix that
    # never arrives is quadratic, and this runs inline on the request path. 60 chars matches
    # the heuristic scanner's construction and covers the phrasings seen in testing.
    (re.compile(r"\bact\s+as\b.{0,60}\b(without|bypass|ignore)\b", re.I), "act_as_bypass"),
    (re.compile(r"\boverride\s+(your\s+)?(instructions?|rules?|guidelines?)\b", re.I),
     "override_instructions"),
    (re.compile(r"\bforget\s+(your\s+)?(previous|prior|all)\b", re.I), "forget_previous"),
]

# ---------------------------------------------------------------------------
# SOFT tier (description) — imperative phrasing that ALSO appears in legitimate
# tool descriptions ("You must provide an absolute path", "Always call list_*
# first"). Too many false positives to block; recorded only.
# ---------------------------------------------------------------------------
_SOFT_DESC_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\byou\s+must\b", re.I), "imperative_you_must"),
    (re.compile(r"\balways\s+call\b.{0,60}\bfirst\b", re.I), "always_call_first"),  # bounded, as above
]

# The two bounded-gap patterns above, decomposed for the abandon probe: (label, prefix,
# suffix). The bound is the `.{0,60}` in each pattern, and a description carrying the
# prefix with the suffix further away than that is exactly what the bound stopped
# seeing — recorded on the verdict (`gap_abandoned`), never a reason. Kept beside the
# patterns they mirror: a change to one is a change to both, pinned by
# `test_gap_probe_bound_matches_the_pattern`.
_GAP_BOUND = 60
_GAP_PROBES: list[tuple[str, re.Pattern, re.Pattern]] = [
    ("act_as_bypass",
     re.compile(r"\bact\s+as\b", re.I), re.compile(r"\b(without|bypass|ignore)\b", re.I)),
    ("always_call_first",
     re.compile(r"\balways\s+call\b", re.I), re.compile(r"\bfirst\b", re.I)),
]

# ---------------------------------------------------------------------------
# SOFT tier — suspicious tool names (anchored word matches)
# ---------------------------------------------------------------------------
# These names appear in legitimate tools too (run_shell, http_request).
# Recorded only.

_SUSPICIOUS_NAME_RE = re.compile(
    r"\b(exec|shell|eval|system|cmd|curl|wget|upload|send_file|read_env|"
    r"execute|subprocess|spawn|popen|run_cmd|invoke_shell)\b",
    re.I,
)

# Overly-broad description markers
_BROAD_DESC_RE = re.compile(
    r"\b(do\s+anything|arbitrary\s+(command|code|action|task)|no\s+restrictions?)\b",
    re.I,
)


def _tool_name(tool: dict) -> str:
    """Extract the tool name from the OpenAI function-tool schema."""
    fn = tool.get("function") or tool
    return str(fn.get("name") or "")


def _tool_description(tool: dict) -> str:
    fn = tool.get("function") or tool
    return str(fn.get("description") or "")


# Depth cap: `parameters` is attacker-supplied JSON, so the walk is bounded rather
# than trusting the schema to be shallow.
_MAX_SCHEMA_DEPTH = 12


def _iter_schema_strings(node: object, depth: int = 0):
    """Yield the catalog-visible strings in a JSON-Schema subtree.

    `description` (at every level, including `properties`, `items`, `$defs`) and
    `enum` values: the text a model sees rendered alongside the tool's own
    description. `title` is deliberately left out — it is rarely rendered and
    widening to it adds false positives for little coverage.
    """
    if depth > _MAX_SCHEMA_DEPTH:
        return
    if isinstance(node, dict):
        desc = node.get("description")
        if isinstance(desc, str):
            yield desc
        enum = node.get("enum")
        if isinstance(enum, list):
            yield from (v for v in enum if isinstance(v, str))
        for value in node.values():
            yield from _iter_schema_strings(value, depth + 1)
    elif isinstance(node, list):
        for value in node:
            yield from _iter_schema_strings(value, depth + 1)


def _tool_schema_text(tool: dict) -> str:
    """Screened text from the tool's parameter schema.

    Kept separate from `_tool_description` so `empty_description` keeps meaning
    "no top-level documentation" rather than silently passing on a nested one.
    """
    fn = tool.get("function") or tool
    return "\n".join(_iter_schema_strings(fn.get("parameters")))


def inspect_tools(tools: list[dict]) -> ToolVerdict:
    """Inspect a list of tool definitions from a request body.

    Returns a ToolVerdict indicating whether any checks fired and whether the
    verdict warrants a hard block (vs. record-only).
    """
    if not tools:
        return ToolVerdict(flagged=False)

    reasons: list[str] = []
    suspicious: list[str] = []
    gap_abandoned: list[str] = []
    hard = False

    for tool in tools:
        name = _tool_name(tool)
        desc = _tool_description(tool)
        # The catalog the model reads is the description *plus* the parameter schema's
        # own description/enum text, so both are screened with the same patterns.
        schema_text = _tool_schema_text(tool)
        screened = f"{desc}\n{schema_text}" if schema_text else desc
        matched: set[str] = set()

        # --- HARD tier: description-level instruction injection ---
        for pat, label in _HARD_PATTERNS:
            if pat.search(screened):
                hard = True
                reasons.append(f"hard:{label}:{name or '<unnamed>'}")
                matched.add(label)

        # --- SOFT tier: imperative description phrasing (many false positives) ---
        for pat, label in _SOFT_DESC_PATTERNS:
            if pat.search(screened):
                reasons.append(f"soft:{label}:{name or '<unnamed>'}")
                suspicious.append(name or "<unnamed>")
                matched.add(label)

        # --- Gap probe: only where a bounded pattern missed; visibility, not a verdict.
        # No tool name in the value — the caveat column carries no free text.
        for label, prefix, suffix in _GAP_PROBES:
            if label in matched:
                continue
            distance = bounded_gap_abandoned(screened, prefix, suffix, _GAP_BOUND)
            if distance is not None:
                gap_abandoned.append(f"gap_abandoned:{label}:{distance}")

        # --- SOFT tier: suspicious name ---
        if name and _SUSPICIOUS_NAME_RE.search(name):
            reasons.append(f"soft:suspicious_name:{name}")
            suspicious.append(name)

        # --- SOFT tier: overly-broad description ---
        if _BROAD_DESC_RE.search(screened):
            reasons.append(f"soft:broad_description:{name or '<unnamed>'}")
            suspicious.append(name or "<unnamed>")

        # --- SOFT tier: empty description (no documentation) ---
        if not desc.strip():
            reasons.append(f"soft:empty_description:{name or '<unnamed>'}")
            suspicious.append(name or "<unnamed>")

    gap_abandoned = list(dict.fromkeys(gap_abandoned))  # dedupe, preserve order
    if not reasons:
        return ToolVerdict(flagged=False, gap_abandoned=gap_abandoned)

    return ToolVerdict(
        flagged=True,
        hard=hard,
        reasons=reasons,
        suspicious_tools=list(dict.fromkeys(suspicious)),  # dedupe, preserve order
        gap_abandoned=gap_abandoned,
    )
