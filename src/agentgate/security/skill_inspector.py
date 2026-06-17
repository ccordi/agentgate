"""Static analysis of `tools[]` definitions in the request body.

Covers the prompt-injection-via-tool-catalog attack surface (distinct from the
injection guard, which scans untrusted *content* in messages).  MCP tools appear
as ordinary function entries in `tools[]`; this module covers them too.

Two-tier enforcement:

  HARD (high precision) — description-level instruction injection.
      "ignore previous instructions", "you must", "always call … first", etc.
      Low FP, justifies a 400 block.

  SOFT (record-only) — coarse heuristics: suspicious tool names, overly-broad
      descriptions.  High FP — legitimate tools are routinely named
      run_shell/execute/http_request.  Log + audit but forward; accumulate
      data before considering a block.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class SkillVerdict:
    flagged: bool
    hard: bool = False
    reasons: list[str] = field(default_factory=list)
    suspicious_tools: list[str] = field(default_factory=list)


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
    (re.compile(r"\bact\s+as\b.*\b(without|bypass|ignore)\b", re.I), "act_as_bypass"),
    (re.compile(r"\boverride\s+(your\s+)?(instructions?|rules?|guidelines?)\b", re.I),
     "override_instructions"),
    (re.compile(r"\bforget\s+(your\s+)?(previous|prior|all)\b", re.I), "forget_previous"),
]

# ---------------------------------------------------------------------------
# SOFT tier (description) — imperative phrasing that ALSO appears in legitimate
# tool descriptions ("You must provide an absolute path", "Always call list_*
# first"). Too high-FP to block in-path; record-only until data justifies it.
# ---------------------------------------------------------------------------
_SOFT_DESC_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\byou\s+must\b", re.I), "imperative_you_must"),
    (re.compile(r"\balways\s+call\b.*\bfirst\b", re.I), "always_call_first"),
]

# ---------------------------------------------------------------------------
# SOFT tier — suspicious tool names (anchored word matches)
# ---------------------------------------------------------------------------
# These names appear in legitimate tools too (run_shell, http_request).
# Record-only until FP data justifies graduating to hard.

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


def inspect_tools(tools: list[dict]) -> SkillVerdict:
    """Inspect a list of tool definitions from a request body.

    Returns a SkillVerdict indicating whether any checks fired and whether the
    verdict warrants a hard block (vs. record-only).
    """
    if not tools:
        return SkillVerdict(flagged=False)

    reasons: list[str] = []
    suspicious: list[str] = []
    hard = False

    for tool in tools:
        name = _tool_name(tool)
        desc = _tool_description(tool)

        # --- HARD tier: description-level instruction injection ---
        for pat, label in _HARD_PATTERNS:
            if pat.search(desc):
                hard = True
                reasons.append(f"hard:{label}:{name or '<unnamed>'}")

        # --- SOFT tier: high-FP imperative description phrasing ---
        for pat, label in _SOFT_DESC_PATTERNS:
            if pat.search(desc):
                reasons.append(f"soft:{label}:{name or '<unnamed>'}")
                suspicious.append(name or "<unnamed>")

        # --- SOFT tier: suspicious name ---
        if name and _SUSPICIOUS_NAME_RE.search(name):
            reasons.append(f"soft:suspicious_name:{name}")
            suspicious.append(name)

        # --- SOFT tier: overly-broad description ---
        if _BROAD_DESC_RE.search(desc):
            reasons.append(f"soft:broad_description:{name or '<unnamed>'}")
            suspicious.append(name or "<unnamed>")

        # --- SOFT tier: empty description (no documentation) ---
        if not desc.strip():
            reasons.append(f"soft:empty_description:{name or '<unnamed>'}")
            suspicious.append(name or "<unnamed>")

    if not reasons:
        return SkillVerdict(flagged=False)

    return SkillVerdict(
        flagged=True,
        hard=hard,
        reasons=reasons,
        suspicious_tools=list(dict.fromkeys(suspicious)),  # dedupe, preserve order
    )
