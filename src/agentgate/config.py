"""Settings, provider registry, and routing-rules loading."""

from __future__ import annotations

from functools import lru_cache

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from agentgate.guards import BACKENDS as GUARD_BACKENDS  # noqa: F401 — re-exported name

# `GUARD_BACKENDS` is `guards.BACKENDS`, re-exported under the name the validators below
# read better with. One definition, in the package that dispatches on it.


class Provider(BaseModel):
    """An upstream LLM endpoint the gateway can forward to."""

    name: str
    base_url: str
    # OpenAI Chat Completions clients use /v1/chat/completions. Google's OpenAI-compat
    # endpoint lives at /v1beta/openai/chat/completions, so for the Gemini upstream we
    # rewrite the path.
    chat_completions_path: str = "/v1/chat/completions"
    is_local: bool = False
    # When set, the gateway rewrites the `model` field in the request body before
    # forwarding. Required for strict local servers (oMLX, llama.cpp) that reject
    # unknown model names. None = pass through unchanged.
    model_name: str | None = None
    # When set, inject `Authorization: Bearer <api_key>` into forwarded headers,
    # replacing any inbound auth. Local servers (oMLX) require a Bearer token even
    # though they don't validate it; cloud providers use the inbound key.
    api_key: str | None = None


# Default registry. Real keys are never stored here — auth is passed through from
# the inbound request header (plain header pass-through).
DEFAULT_PROVIDERS: dict[str, Provider] = {
    "gemini": Provider(
        name="gemini",
        base_url="https://generativelanguage.googleapis.com",
        chat_completions_path="/v1beta/openai/chat/completions",
        is_local=False,
    ),
    "openai": Provider(
        name="openai",
        base_url="https://api.openai.com",
        chat_completions_path="/v1/chat/completions",
        is_local=False,
    ),
    "ollama": Provider(
        name="ollama",
        base_url="http://127.0.0.1:11434",
        chat_completions_path="/v1/chat/completions",
        is_local=True,
    ),
    # Local generation route (sensitive content stays here, zero cloud egress). Points at
    # the local OpenAI-compat server (:8000) — for example oMLX or llama.cpp. Switching
    # between them is a base_url change. Set AGENTGATE_LOCAL_MODEL_OVERRIDE to use a
    # different model name without editing this file.
    "local": Provider(
        name="local",
        base_url="http://127.0.0.1:8000",
        chat_completions_path="/v1/chat/completions",
        is_local=True,
        model_name="gemma-4-26b-a4b-it",
    ),
    # Bench-only: deterministic canned-SSE upstream (bench/mock_upstream.py on :4200).
    # Selected exclusively via AGENTGATE_DEFAULT_PROVIDER=mock for load benchmarks.
    # is_local=False so only the (USD) cloud spend cap applies — and the mock's unknown
    # model prices at $0, so the cap never trips mid-run (the local route's request-count
    # guard would otherwise 429 under load).
    "mock": Provider(
        name="mock",
        base_url="http://127.0.0.1:4200",
        chat_completions_path="/v1/chat/completions",
        is_local=False,
    ),
}


class RoutingRule(BaseModel):
    """One declarative routing rule. First matching rule wins; ``None`` conditions
    are ignored, so a rule with no conditions is an unconditional default."""

    name: str
    sensitivity_in: list[str] | None = None
    agent_in: list[str] | None = None
    any_flags: list[str] | None = None  # e.g. ["cloud_unavailable", "over_spend_cap"]
    action: str  # "route_local" | "prefer_cloud"


class RoutingConfig(BaseModel):
    """The rules-table router config."""

    enabled: bool = True
    default_local: str = "local"   # provider name
    default_cloud: str = "gemini"  # provider name
    # When True, drop "secret" from the sensitive-stays-local rule's sensitivity_in,
    # so secret-bearing content takes the cloud fork and the outbound redaction gate
    # (pipeline.prepare_body, cloud-only) gets exercised. Default False — secrets stay local.
    secrets_to_cloud: bool = False
    # Safety-first order: sensitive always local; if cloud is down/over-cap, local; else
    # honor agent pins; else default cloud.
    rules: list[RoutingRule] = Field(default_factory=lambda: [
        RoutingRule(name="sensitive-stays-local",
                    sensitivity_in=["pii", "secret", "private_repo"], action="route_local"),
        RoutingRule(name="fallback-on-failure",
                    any_flags=["cloud_unavailable", "over_spend_cap"], action="route_local"),
        RoutingRule(name="agent-pins", agent_in=["capture", "web-research"],
                    action="prefer_cloud"),
        RoutingRule(name="default", action="prefer_cloud"),
    ])

    @model_validator(mode="after")
    def _apply_secrets_to_cloud(self) -> RoutingConfig:
        if self.secrets_to_cloud:
            for rule in self.rules:
                if rule.name == "sensitive-stays-local" and rule.sensitivity_in is not None:
                    rule.sensitivity_in = [s for s in rule.sensitivity_in if s != "secret"]
        return self


class EgressConfig(BaseModel):
    """Tier-3 egress PDP config.

    Static allowlist of destinations the PDP treats as safe regardless of payload
    sensitivity. Loopback addresses (127.0.0.1 / localhost / ::1) are always allowed
    even if not listed here. The allowlist is static — there is no dynamic per-session
    approval mechanism.
    """

    allowlist: list[str] = Field(default_factory=list)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AGENTGATE_", env_file=".env", extra="ignore",
        env_nested_delimiter="__",  # AGENTGATE_ROUTING__ENABLED=true → routing.enabled
    )

    host: str = "127.0.0.1"
    port: int = 4100

    # Default upstream when routing is disabled or no rule matches.
    default_provider: str = "gemini"

    # Upstream request timeout (seconds). Generous — agent turns can be long.
    upstream_timeout_s: float = 300.0

    # SQLite audit store. Portable SQLAlchemy types keep a Postgres swap a URL change away.
    database_url: str = "sqlite+aiosqlite:///./data/agentgate.db"

    redis_url: str = "redis://127.0.0.1:6379/0"

    # Inbound injection guard backend: "deberta" (in-process ONNX classifier — far higher
    # recall, needs the `guard` extra + a pulled model; ~10-30ms, run off the event loop) or
    # "heuristic" (regex, always available, ~0.3ms). Default deberta, but the gateway falls
    # back to heuristic at startup if the model/runtime isn't available (never hard-fails —
    # a dead gateway blocks all agent traffic). Install with `uv run --extra guard agentgate`.
    guard_backend: str = "deberta"

    # Per-key override of guard_backend. Maps a raw key_id (as produced by key_id_from_auth)
    # -> backend name. Keys not in
    # this map use guard_backend (the default above). Same JSON-in-env convention as
    # other dict-valued settings (e.g. AGENTGATE_PROVIDERS__...): pydantic-settings
    # parses a JSON object for AGENTGATE_GUARD_BACKEND_OVERRIDES, e.g.
    #   AGENTGATE_GUARD_BACKEND_OVERRIDES='{"<key_id>": "llm"}'
    # Values are validated at startup against GUARD_BACKENDS — an unrecognized backend
    # is a fail-fast config error (a typo here would silently disable the guard for
    # that key).
    guard_backend_overrides: dict[str, str] = Field(default_factory=dict)

    # Observe / flag-but-don't-block mode for the inbound injection guard. When true, a
    # hard-positive verdict is logged + audited (injection_hard=True) but the request is
    # forwarded normally instead of returning 400. Useful for measuring false-positive rate
    # on live traffic without the guard breaking the agent loop (a hard-block on a benign
    # framework control message fails the whole turn). The scan still runs; only the block
    # action is suppressed. AGENTGATE_GUARD_OBSERVE_MODE=true.
    guard_observe_mode: bool = False

    # --- Passive traffic capture ---
    # Off by default. When on, requests from the tagged capture agent (identified by the
    # /a/<agent_id> tagged route) have their untrusted content appended to the eval corpus
    # for later judge-labeling — the benign / false-positive sampling path. Scoped to ONE
    # intentionally-non-sensitive agent.
    capture_enabled: bool = False
    capture_agent_id: str = "capture"
    # Private-repo markers for the sensitivity classifier (empty → never fires).
    private_repo_markers: list[str] = Field(default_factory=list)
    # src defaults must not point into the eval tree — see capture.py.
    capture_path: str = "data/fp_capture.jsonl"

    providers: dict[str, Provider] = Field(default_factory=lambda: dict(DEFAULT_PROVIDERS))

    # Credentials for the local provider (oMLX/llama.cpp). Stored separately because
    # pydantic-settings can't partially-update dict-valued fields — setting
    # AGENTGATE_PROVIDERS__LOCAL__API_KEY would replace the entire entry. Instead,
    # set AGENTGATE_LOCAL_API_KEY and the validator below merges it in.
    local_api_key: str | None = None

    # Bearer token for the admin plane (`/admin/kill/*`). REQUIRED to serve — startup
    # refuses without it (validate_runtime_settings below). The kill switch is the one
    # control aimed *at* the agent, and the agent reaches loopback by policy; an unarmed
    # deployment read as closed while open, so "unset → open" stopped being a default and
    # became a refused start.
    #
    # Must be its own value: there is deliberately no fallback to local_api_key, because the
    # PEP holds that one to bearer the PDP, so accepting it here would authenticate the
    # governed agent with a credential it already has.
    admin_token: str | None = None

    # Bearer token for the egress PDP (`/a/egress/decision`). REQUIRED to serve, and
    # dedicated — distinct from admin_token (roles separated) and from local_api_key
    # (an upstream credential, not a gateway one). A loopback bind does not make this
    # redundant: loopback excludes remote *sockets*, not remote *code* — a web page in
    # the operator's browser reaches 127.0.0.1, and with DNS rebinding reads responses.
    # The token is what a page can never have.
    pdp_token: str | None = None

    # --- Local-route request overrides ---
    # Env-driven knobs to tune the local upstream WITHOUT a code change, so a scoped debug
    # session can sweep model/params via a restart alone. All default-off (None) → no behavior
    # change unless set. Applied only on the local (is_local) route, after the model rewrite.
    #   AGENTGATE_LOCAL_MODEL_OVERRIDE  — replace the forced local model name
    #   AGENTGATE_LOCAL_STOP            — comma-list → injected as the OpenAI `stop` array
    #   AGENTGATE_LOCAL_MAX_TOKENS      — injected as max_completion_tokens (runaway cap)
    #   AGENTGATE_LOCAL_ENABLE_THINKING — injected as chat_template_kwargs.enable_thinking
    local_model_override: str | None = None
    local_stop: str | None = None
    local_max_tokens: int | None = None
    local_enable_thinking: bool | None = None

    routing: RoutingConfig = Field(default_factory=RoutingConfig)

    # Tier-3 egress PDP config. AGENTGATE_EGRESS__ALLOWLIST=["host1","host2"]
    egress: EgressConfig = Field(default_factory=EgressConfig)

    # Cloud-egress PII/secret redaction.  AGENTGATE_REDACTION_ENABLED=false to disable.
    redaction_enabled: bool = True

    # --- Audit content tier ---
    # Capture is disabled if content_enc_key is unset (fail-closed: never store plaintext).
    # Generate a key: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"  # noqa: E501
    content_capture_enabled: bool = True      # AGENTGATE_CONTENT_CAPTURE_ENABLED
    content_sample_rate: float = 0.05         # AGENTGATE_CONTENT_SAMPLE_RATE
    content_retention_days: int = 30          # AGENTGATE_CONTENT_RETENTION_DAYS
    content_enc_key: str | None = None        # AGENTGATE_CONTENT_ENC_KEY  (flat secret)

    @model_validator(mode="after")
    def _validate_guard_backends(self) -> Settings:
        # Fail fast on an unrecognized backend name — a typo here would silently
        # disable the guard for whichever key (or the whole gateway) it applies to.
        if self.guard_backend not in GUARD_BACKENDS:
            raise ValueError(
                f"guard_backend: unknown backend {self.guard_backend!r} "
                f"(expected one of {sorted(GUARD_BACKENDS)})"
            )
        for key_id, backend in self.guard_backend_overrides.items():
            if backend not in GUARD_BACKENDS:
                raise ValueError(
                    f"guard_backend_overrides[{key_id!r}]: unknown backend {backend!r} "
                    f"(expected one of {sorted(GUARD_BACKENDS)})"
                )
        return self

    @model_validator(mode="after")
    def _apply_local_api_key(self) -> Settings:
        if self.local_api_key and "local" in self.providers:
            self.providers["local"] = self.providers["local"].model_copy(
                update={"api_key": self.local_api_key}
            )
        return self

    def provider(self, name: str | None = None) -> Provider:
        key = name or self.default_provider
        if key not in self.providers:
            raise KeyError(f"unknown provider: {key!r}")
        return self.providers[key]


def validate_runtime_settings(settings: Settings) -> None:
    """Refuse to serve with an unsafe posture. Called from app startup (lifespan) and
    the CLI entry point; NOT from Settings construction, so tests and offline tooling
    (the audit CLI) can build a Settings without a serving posture.

    Two preconditions, both load-bearing rather than stylistic:

    1. **The bind must be loopback, with no override flag.** The threat model's scope
       ("one operator, one host, the gateway on loopback") is what justifies the proxy
       routes having no inbound auth — the proxy's authentication IS the network layer.
       `AGENTGATE_HOST=0.0.0.0` can otherwise expose the proxy silently, especially in a
       container. There is no
       `AGENTGATE_ALLOW_NON_LOOPBACK` escape hatch: no config state makes a non-loopback
       bind safe, because the proxy has no credential to arm — a flag would present an
       unsupportable topology as a supported mode. If remote callers ever become a real
       requirement, the unlock is gateway-issued inbound credentials (which also makes
       key_id a verified identity), not a flag.

    2. **Both gateway tokens must be set.** Conditional (set-it-and-it-arms) auth left
       every unarmed deployment looking closed while open; and loopback alone does not
       substitute — browser-borne code reaches loopback (blind CSRF today, readable with
       DNS rebinding), so the state-changing admin plane and the policy-oracle PDP need
       a secret a web page cannot hold, on every interface.

    Residual, stated rather than hidden: this validates `settings.host`, so a direct
    `uvicorn agentgate.app:app --host 0.0.0.0` — which never consults settings — still
    binds wide. The sanctioned entry points cover the env-var route; the residual is
    documented in the threat model.
    """
    # Deferred to keep `config` import-light — egress.policy pulls in sensitivity,
    # redaction and content. Not a cycle: egress.policy never imports config.
    from agentgate.egress.policy import is_loopback_host

    problems: list[str] = []
    if not is_loopback_host(settings.host):
        problems.append(
            f"AGENTGATE_HOST={settings.host!r} is not a loopback address. The proxy "
            "routes are unauthenticated by design (transparent pass-through), so a "
            "non-loopback bind exposes them to anything routable. There is no override; "
            "remote callers need gateway-issued inbound auth, which does not exist yet."
        )
    if not settings.admin_token:
        problems.append(
            "AGENTGATE_ADMIN_TOKEN is not set. The admin plane (/admin/kill/*) is "
            "state-changing and must never run open: loopback does not exclude "
            "browser-borne requests, and an unarmed deployment reads as closed."
        )
    if not settings.pdp_token:
        problems.append(
            "AGENTGATE_PDP_TOKEN is not set. The egress PDP (/a/egress/decision) is a "
            "policy oracle; it requires its own dedicated bearer (not local_api_key, "
            "which is the local upstream's credential)."
        )
    # The PEP loads the PDP token into the agent's environment. Reusing that value for the
    # admin plane would also give the governed agent control of the kill switch.
    pairs = (
        ("AGENTGATE_ADMIN_TOKEN", settings.admin_token, "AGENTGATE_PDP_TOKEN", settings.pdp_token),
        ("AGENTGATE_ADMIN_TOKEN", settings.admin_token,
         "AGENTGATE_LOCAL_API_KEY", settings.local_api_key),
        ("AGENTGATE_PDP_TOKEN", settings.pdp_token,
         "AGENTGATE_LOCAL_API_KEY", settings.local_api_key),
    )
    for a_name, a_val, b_name, b_val in pairs:
        if a_val and b_val and a_val == b_val:
            problems.append(
                f"{a_name} and {b_name} are the same value. They are separate "
                "credentials on purpose; sharing one collapses the trust boundary "
                "between them."
            )
    if problems:
        raise RuntimeError(
            "refusing to start with an unsafe posture:\n- " + "\n- ".join(problems)
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
