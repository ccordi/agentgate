"""Tier-3 egress control: PDP and PEP in one package.

Off the model wire — this is the control plane for the agent's *tool* calls, not for
its prompts. Split into a decision point and an enforcement point so the two can be
reasoned about separately:

- `policy.evaluate` (PDP) — pure decision logic: what is the destination, how
  sensitive is the payload, allow or deny.
- `api` — the `POST /a/egress/decision` endpoint that exposes the PDP over loopback.
- `pep.safe_http_request` (PEP) — the client that consults the PDP *before* acting
  and obeys the verdict, failing closed if the PDP is unreachable.
- `mcp_server` — the PEP wrapped as an MCP tool, which is how an agent harness gets
  a network capability that is gated by construction.

No package-level re-exports — every caller names the submodule it wants. `mcp_server`
carries an optional dependency, so importing this package stays cheap.
"""
