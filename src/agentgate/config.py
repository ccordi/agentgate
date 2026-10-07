"""Settings, provider registry, and routing-rules loading."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import TYPE_CHECKING, overload

from dotenv import dotenv_values
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from agentgate.limits.spend import SpendConfig

from agentgate.guards import BACKENDS as GUARD_BACKENDS  # noqa: F401 — re-exported name

# `GUARD_BACKENDS` is `guards.BACKENDS`, re-exported under the name the validators below
# read better with. One definition, in the package that dispatches on it.

log = logging.getLogger("agentgate")


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
    # though they don't validate it; cloud providers use the inbound key — except when
    # issued keys are required: the inbound credential is then a gateway-minted key,
    # which never leaves the gateway, so a non-local provider without an api_key of its
    # own is unusable on that deployment (the pipeline rejects 503 rather than forward).
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
        model_name="gemma-4-26b-a4b-it-oq4",
    ),
    # Deterministic canned-SSE upstream (bench/mock_upstream.py on :4200) for load
    # tests and the getting-started demo; selected with AGENTGATE_DEFAULT_PROVIDER=mock.
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
    action: str  # "route_local" | "prefer_cloud"


class RoutingConfig(BaseModel):
    """The rules-table router config."""

    enabled: bool = True
    default_local: str = "local"  # provider the route_local branch resolves to
    # Provider the prefer_cloud branch resolves to. Ships as "local", so a new install has
    # no cloud egress until the operator sets this to "gemini" or "openai".
    default_cloud: str = "local"
    # Safety-first order: sensitive always local; else default cloud.
    rules: list[RoutingRule] = Field(default_factory=lambda: [
        RoutingRule(name="sensitive-stays-local",
                    sensitivity_in=["pii", "secret", "private_repo"], action="route_local"),
        RoutingRule(name="default", action="prefer_cloud"),
    ])


class EgressConfig(BaseModel):
    """Egress PDP config.

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

    # Container deployments only: permits a non-loopback BIND — and nothing else — on
    # the operator's assertion that this process runs in an isolated network namespace
    # whose ingress is controlled outside it (the shipped compose file publishes the
    # port to host loopback only). Tokens and the Host guard apply unchanged. Full
    # contract in validate_runtime_settings; asserting it on a bare host recreates
    # exactly the wide-bind exposure the validator refuses.
    container_bind: bool = False

    # Upstream when routing is disabled. Ships as "local"; with routing on, the rules table
    # picks between routing.default_local and routing.default_cloud and this is not read.
    default_provider: str = "local"

    # Upstream PER-OPERATION timeout (seconds). Generous — agent turns can be long.
    # Note this is httpx's connect/read/write/pool budget, and `read` is per-read: a stream
    # that emits one chunk every 299 s never trips it. The wall-clock budget for a whole
    # request is upstream_total_deadline_s below.
    upstream_timeout_s: float = 300.0

    # TOTAL wall-clock budget for one request, including the streamed response. Without
    # it a request can run well past the per-operation "timeout" above, holding a slot
    # against a model server that serves one request at a time. The default sits well
    # above a long agent turn, so it acts only as a backstop; lower it for a less trusted
    # upstream.
    upstream_total_deadline_s: float = 900.0

    # Upstream connection-pool caps. Defaults mirror httpx's own (100/20); they exist as
    # settings so a load run can raise them deliberately: at concurrency > max_connections
    # the pool becomes a hidden admission control and queueing there is unattributable to
    # any pipeline stage. 0 means unlimited.
    upstream_max_connections: int = 100
    upstream_max_keepalive: int = 20

    # SQLite audit store. Portable SQLAlchemy types keep a Postgres swap a URL change away.
    database_url: str = "sqlite+aiosqlite:///./data/agentgate.db"

    redis_url: str = "redis://127.0.0.1:6379/0"
    # Refuse to boot if Redis is unreachable instead of falling back to in-process
    # counters. Off for single-process use; required in multi-replica deployments, where
    # a silent fallback would give each replica its own spend counters and kill switch.
    require_shared_limits: bool = False

    # Require a gateway-minted API key on the proxy (chat) routes. Off by default: on
    # loopback the network position authenticates, and key_id stays a plain hash of
    # whatever credential the client presents. On, inbound credentials are verified
    # against the issued_keys table (hashes only; mint via POST /admin/keys) and
    # missing/unknown/revoked keys are rejected 401 before admission — key_id becomes
    # a verified identity. The admin plane and egress PDP keep their own dedicated
    # bearers either way.
    require_issued_keys: bool = False

    # Inbound injection guard backend: "deberta" (in-process ONNX classifier — far higher
    # recall, needs the `guard` extra + the classifier's model files; latency on the Results page;
    # run off the event loop) or
    # "heuristic" (regex, always available, ~0.3ms). Default deberta. No availability
    # fallback: a backend whose model will not load refuses startup (`app.lifespan`).
    # Install with `uv run --extra guard agentgate`.
    guard_backend: str = "deberta"

    # Width of the shared worker-thread pool that guard scans run on (asyncio's
    # to_thread executor). 0 = asyncio's default (cores + 4). Load runs pin this to a
    # deliberate compute budget so scans cannot occupy every core on the box; pairs
    # with AGENTGATE_GUARD_INTRA_OP_THREADS (read by the deberta module) which bounds
    # threads per inference.
    scan_executor_threads: int = 0

    # Per-key override of guard_backend. Maps a raw key_id (as produced by
    # key_id_from_auth) -> backend name. Keys not in
    # this map use guard_backend (the default above). Same JSON-in-env convention as
    # other dict-valued settings (e.g. AGENTGATE_PROVIDERS): pydantic-settings
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

    providers: dict[str, Provider] = Field(default_factory=lambda: dict(DEFAULT_PROVIDERS))

    # Credentials for the local provider (oMLX/llama.cpp). Stored separately because
    # pydantic-settings can't partially-update dict-valued fields — setting
    # AGENTGATE_PROVIDERS__LOCAL__API_KEY would replace the entire entry. Instead,
    # set AGENTGATE_LOCAL_API_KEY and the validator below merges it in.
    local_api_key: str | None = None

    # Dedicated bearer for the admin plane (/admin/kill/*, /admin/keys). REQUIRED to
    # serve — see validate_runtime_settings(). There is deliberately no local_api_key
    # fallback: that key is the local *upstream's* credential, held by the PEP by
    # construction (it reads the same `.env`), so accepting it here would authenticate
    # the one caller the control exists to exclude.
    admin_token: str | None = None

    # Dedicated bearer for the egress PDP (/a/egress/decision). REQUIRED to serve.
    # Distinct from admin_token (roles separated) and from local_api_key (which is an
    # upstream credential, not a gateway one). A loopback bind does not make this
    # redundant: loopback excludes remote *sockets*, not remote *code* — a web page in
    # the operator's browser reaches 127.0.0.1, and with DNS rebinding reads responses.
    # The token is what a page can never have.
    pdp_token: str | None = None

    # --- Local-route request overrides ---
    # Env-driven knobs for the local upstream; `local_route` applies them and says why.
    # All default-off (None) → no behavior change unless set. Applied only on the local
    # (is_local) route, after the model rewrite.
    #   AGENTGATE_LOCAL_MODEL_OVERRIDE  — replace the forced local model name
    #   AGENTGATE_LOCAL_STOP            — comma-list → injected as the OpenAI `stop` array
    #   AGENTGATE_LOCAL_MAX_TOKENS      — injected as max_completion_tokens and max_tokens
    #   AGENTGATE_LOCAL_ENABLE_THINKING — injected as chat_template_kwargs.enable_thinking
    local_model_override: str | None = None
    local_stop: str | None = None
    local_max_tokens: int | None = None
    local_enable_thinking: bool | None = None

    # Private-repo markers for the sensitivity classifier (empty → never fires).
    private_repo_markers: list[str] = Field(default_factory=list)

    routing: RoutingConfig = Field(default_factory=RoutingConfig)

    # Egress PDP config. AGENTGATE_EGRESS__ALLOWLIST=["host1","host2"]
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

    # --- Spend caps + kill switch ---
    # Defaults match SpendConfig's own.
    spend_cloud_usd_cap: float = 5.0          # AGENTGATE_SPEND_CLOUD_USD_CAP
    spend_local_request_cap: int = 10_000     # AGENTGATE_SPEND_LOCAL_REQUEST_CAP
    spend_window_s: int = 3600                # AGENTGATE_SPEND_WINDOW_S  (fixed window)
    spend_kill_ttl_s: int | None = None       # AGENTGATE_SPEND_KILL_TTL_S (None = sticky)

    # --- Admission control (per-key concurrency slots) ---
    # OFF by default: with admission_enabled false nothing in the request path consults
    # the controller at all. A slot is held from the front door until the response has
    # fully drained, so these are concurrency bounds, not rates.
    #   AGENTGATE_ADMISSION_ENABLED           — arm it
    #   AGENTGATE_ADMISSION_PER_KEY           — concurrent requests one key may hold
    #   AGENTGATE_ADMISSION_GLOBAL            — process-wide ceiling (try-acquire, never waits)
    #   AGENTGATE_ADMISSION_QUEUE_DEPTH       — waiters queued per key before shedding
    #   AGENTGATE_ADMISSION_WAIT_DEADLINE_S   — how long a queued request may wait
    admission_enabled: bool = False
    admission_per_key: int = 8
    admission_global: int = 64
    admission_queue_depth: int = 32
    admission_wait_deadline_s: float = 5.0

    def spend_config(self) -> SpendConfig:
        # Local import: limits.spend is a leaf module and this keeps config.py free of a
        # package-level dependency on it.
        from agentgate.limits.spend import SpendConfig
        return SpendConfig(
            cloud_usd_cap=self.spend_cloud_usd_cap,
            local_request_cap=self.spend_local_request_cap,
            window_s=self.spend_window_s,
            kill_ttl_s=self.spend_kill_ttl_s,
        )

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
    def _validate_admission(self) -> Settings:
        # A non-positive cap would shed every request the moment admission is armed, and
        # a non-positive deadline would shed every request that ever has to queue. Both
        # read as "throttled" rather than "misconfigured" in production, so reject them
        # at construction. Checked whether or not admission is enabled: the value is
        # wrong either way.
        limits = [
            ("admission_per_key", self.admission_per_key),
            ("admission_global", self.admission_global),
        ]
        for field_name, value in limits:
            if value < 1:
                raise ValueError(f"{field_name}: must be >= 1 (got {value})")
        if self.admission_queue_depth < 0:
            raise ValueError(
                f"admission_queue_depth: must be >= 0 (got {self.admission_queue_depth})"
            )
        if self.admission_wait_deadline_s <= 0:
            raise ValueError(
                f"admission_wait_deadline_s: must be > 0 (got {self.admission_wait_deadline_s})"
            )
        return self

    @model_validator(mode="after")
    def _validate_routing_targets(self) -> Settings:
        """Every setting that names another setting's key is validated at construction.

        `_validate_guard_backends` above applies the same rule to backend names. Provider
        names need it because `provider()` resolves at *request* time: without this check
        a one-character typo in AGENTGATE_ROUTING__DEFAULT_LOCAL would produce a Settings
        object that constructs cleanly, a process that starts, a /healthz that returns ok,
        and then a KeyError → framework 500 on every request, with no audit row and no
        metric.
        """
        known = set(self.providers)
        checked: list[tuple[str, str]] = [("default_provider", self.default_provider)]
        if self.routing.enabled:
            checked += [
                ("routing.default_local", self.routing.default_local),
                ("routing.default_cloud", self.routing.default_cloud),
            ]
        for field_name, value in checked:
            if value not in known:
                raise ValueError(
                    f"{field_name}: unknown provider {value!r} "
                    f"(configured providers: {sorted(known)})"
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
        """Resolve a provider name to its registered Provider.

        Called at REQUEST time (routing.resolve → here), so an unknown name is a
        per-request KeyError rather than a startup failure — which is why the names that
        can reach it are pre-validated at construction by `_validate_routing_targets`.
        Any new setting that names a provider must join that validator.
        """
        key = name or self.default_provider
        if key not in self.providers:
            raise KeyError(f"unknown provider: {key!r}")
        return self.providers[key]


def validate_runtime_settings(settings: Settings) -> None:
    """Refuse to serve with an unsafe posture. Called from app startup (lifespan) and
    the CLI entry point; NOT from Settings construction, so tests and offline tooling
    can build a Settings without a serving posture.

    Two preconditions, both load-bearing rather than stylistic:

    1. **The bind must be loopback, unless the container contract is asserted.** The
       threat model's scope ("one operator, one host, the gateway on loopback") is what
       justifies the proxy routes having no inbound auth by default (issued-key
       enforcement is a separate, default-off gate) — the proxy's authentication
       IS the network layer. A wide host bind has no transport security (TLS at
       minimum), so there is no generic `AGENTGATE_ALLOW_NON_LOOPBACK` escape hatch.

       The one exception is `AGENTGATE_CONTAINER_BIND`, the container topology
       contract. Inside a container, bind address
       and exposure decouple: 0.0.0.0 in an isolated network namespace reaches `lo`
       plus one veth, and actual ingress is whatever the deployment publishes — the
       shipped compose file publishes host-loopback only, keeping the reachable set
       identical to the host bind this check defends (a test pins that publish spec).
       The flag asserts exactly that contract: "ingress is controlled outside this
       process." A process cannot verify its namespace's ingress from inside, so this
       is an operator assertion of the same class as the uvicorn gap below —
       and asserting it on a bare host recreates the wide-bind exposure, which the
       threat model documents. The flag touches only this check: the Host guard and
       both token requirements apply unchanged.

    2. **Both gateway tokens must be set.** Conditional (set-it-and-it-arms) auth would
       leave every unarmed deployment looking closed while open; and loopback alone does not
       substitute — browser-borne code reaches loopback (blind CSRF today, readable with
       DNS rebinding), so the state-changing admin plane and the policy-oracle PDP need
       a secret a web page cannot hold, on every interface.

    One gap: this validates `settings.host`, so a direct
    `uvicorn agentgate.app:app --host 0.0.0.0` — which never consults settings — still
    binds wide. The `agentgate` entry point and the env-var route are covered; the
    threat model documents the gap.
    """
    # Deferred to keep `config` import-light — egress.policy pulls in sensitivity,
    # redaction and content. Not a cycle: egress.policy never imports config.
    from agentgate.egress.policy import is_loopback_host

    problems: list[str] = []
    if not settings.container_bind and not is_loopback_host(settings.host):
        problems.append(
            f"AGENTGATE_HOST={settings.host!r} is not a loopback address. By default the "
            "model routes accept requests without a gateway credential, so listening on "
            "another address would expose them to anything that can reach it. Inside a "
            "container whose port is published only on the host's loopback interface, "
            "set AGENTGATE_CONTAINER_BIND=1."
        )
    if not settings.admin_token:
        problems.append(
            "AGENTGATE_ADMIN_TOKEN is not set. The admin API (/admin/kill/*, /admin/keys) "
            "changes gateway state and always requires its own token: a web page open in "
            "a local browser can reach a loopback address."
        )
    if not settings.pdp_token:
        problems.append(
            "AGENTGATE_PDP_TOKEN is not set. The egress policy endpoint "
            "(/a/egress/decision) requires its own token, separate from "
            "AGENTGATE_LOCAL_API_KEY (the local model server's credential)."
        )
    # Presence is not separation. The whole argument for two dedicated tokens is that one
    # value must not span two trust boundaries — and the PEP loads the PDP token into the
    # agent's own environment, so a shared value hands the governed agent the admin plane
    # (and with it the kill switch meant to halt it).
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
    # Companion to the pipeline's forward-point invariant (a minted key never leaves the
    # gateway): with issued keys required, any request resolving to a non-local provider
    # with no api_key to inject is rejected 503 (upstream_credentials_missing). Warn —
    # never refuse — at boot: a keyless cloud entry is valid as long as routing never
    # selects it (forced-local deployments run exactly that config).
    if settings.require_issued_keys:
        keyless = sorted(
            name for name, p in settings.providers.items()
            if not p.is_local and p.api_key is None
        )
        if keyless:
            log.warning(
                "AGENTGATE_REQUIRE_ISSUED_KEYS is on but non-local provider(s) %s have no "
                "api_key to inject: every request routed to one will be rejected 503 "
                "(upstream_credentials_missing) — the gateway never forwards its own "
                "minted key upstream",
                ", ".join(keyless),
            )


@lru_cache
def get_settings() -> Settings:
    return Settings()


@overload
def env_setting(name: str) -> str | None:
    ...


@overload
def env_setting(name: str, default: str) -> str:
    ...


def env_setting(name: str, default: str | None = None) -> str | None:
    """One setting by name, read the way ``Settings`` reads its own: the process
    environment first, then the `.env` file ``Settings`` is configured with, then
    ``default``.

    For the few values read at their point of use rather than through ``Settings`` —
    the classifier's model directory and thread budget, the LLM judge's cache path,
    the tracing endpoint (read when the app module is imported) and the MCP server's
    PDP token and URL (a separate process) — so that one `.env` serves every entry
    point: the gateway, the MCP server and the LiteLLM guardrail. Same parser, same
    precedence: a variable set in the environment wins even when it is empty. The file
    is read, never sourced, and nothing is written into the environment.
    """
    if name in os.environ:
        return os.environ[name]
    env_file = Settings.model_config.get("env_file")
    if isinstance(env_file, str | os.PathLike):
        value = dotenv_values(env_file).get(name)
        if value is not None:
            return value
    return default
