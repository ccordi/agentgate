"""Egress control: the whole PDP/PEP story in one package.

Off the model wire — this is the control plane for the agent's *tool* calls, not for
its prompts. Split into a decision point and an enforcement point so the two can be
reasoned about separately:

- `policy.evaluate` (PDP) — pure decision logic: what is the destination, how
  sensitive is the payload, allow or deny.
- `api` — the `POST /a/egress/decision` endpoint that exposes the PDP over loopback.
- `pep.safe_http_request` (PEP) — the client that consults the PDP *before* acting
  and obeys the verdict, failing closed if the PDP is unreachable.
- `mcp_server` — the PEP wrapped as an MCP tool, which is how an agent application gets
  a network capability that is gated by construction.

The PEP is cooperative, not containing: it checks only the requests made through it. The
agent application's permission settings can steer network access to it, but cannot close
every other path (see the threat model's Assumptions).

No package-level re-exports — every caller names the submodule it wants. `mcp_server`
carries an optional dependency, so importing this package stays cheap.
"""
