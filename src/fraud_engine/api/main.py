"""FastAPI app — wires Sprint 5.1.a-e primitives behind /predict, /health, /ready, /metrics.

Sprint 5 prompt 5.1.f: the keystone module of Sprint 5. Until this lands,
the five primitives built across 5.1.a-e are individually unit-tested
but not invocable as a service. Post-PR, `curl localhost:8000/predict`
returns a real scored decision and Prometheus's `fraud-api` scrape job
flips from DOWN to UP.

Routes:
    - `POST /predict`  — TransactionRequest → PredictionResponse;
       request_id-correlated logs at every stage; per-stage Prometheus
       latency histograms.
    - `GET /health`    — liveness probe (always 200 if process is up).
    - `GET /ready`     — readiness probe (200 iff Redis + Postgres + model
       all OK; 503 with per-source diagnostics otherwise).
    - `GET /metrics`   — Prometheus scrape surface (HTTP + per-stage).

Public surface:
    - `app: FastAPI` — module-level instance for `uvicorn fraud_engine.api.main:app`.
    - `create_app(settings: Settings | None) → FastAPI` — test-side factory
       that lets each integration test spin a fresh, lifespan-isolated
       app with a tweaked Settings (e.g. unreachable Redis URL for the
       degraded-mode test).

Business rationale:
    The five 5.1.a-e primitives meet their unit-test contracts in
    isolation, but only their composition is the actual production
    surface. Wiring them behind a real HTTP service is where contract
    drift between primitives surfaces, where the <100ms P95 latency
    budget (CLAUDE.md §3) becomes a binding gate, and where the
    operational concerns (lifespan startup, request correlation,
    per-stage observability, degraded-mode propagation, readiness vs
    liveness) get exercised end-to-end. This module is the single
    place all of those concerns are answered.

Trade-offs considered:
    - **`create_app(settings)` factory + module-level `app = create_app()`.**
      Production startup uses the simple module-level `app` (so
      `uvicorn fraud_engine.api.main:app` works and Prometheus's
      auto-discovery resolves). Tests use the factory to spin a fresh
      app with a tweaked Settings — each test owns its own lifespan,
      no singleton-state interference. The model joblib re-loads per
      test app (~50–100 ms), but the p95 timing test reuses one client
      so the cost amortises to negligible. Rejected: a global `app`
      with `dependency_overrides` for tests — couples test isolation
      to pytest's run order and confuses the lifespan-state surface.

    - **Lifespan tolerates Redis/Postgres unreachable at startup; logs
      WARNING and continues.** Wraps `feature_service.connect()` in
      `try/except (ConnectionError, asyncpg.PostgresError, RuntimeError,
      OSError)`. On failure, logs a WARNING (`startup_dependency_unreachable`)
      and proceeds with model loading. The FeatureService's per-call
      probe (Decision #2 from PR #49 / 5.1.c) catches the not-connected
      state at request time and flips `degraded_mode=True` per request.
      Consequence: the service starts during a Redis/Postgres reboot
      rather than crash-looping; requests during the outage return real
      predictions on Tier-1 features + population defaults; `/ready`
      returns 503 until the source recovers. Matches production-grade
      fraud APIs (degraded > down). Rejected: fail-fast at startup —
      would mean a Redis hiccup pages on-call to restart the API,
      conflicting with the explicit "degraded mode works when Redis
      down" spec test.

      **Model artefacts: still fail-fast.** `InferenceService.load()`
      and `ShapExplainer.__init__` raise `FileNotFoundError` if the
      joblib/manifest/YAML are missing — these are immutable deployment
      artefacts, not runtime dependencies. A missing model file is a
      deployment bug, not a transient outage.

    - **`inference.predict()` runs in-loop (no `asyncio.to_thread`).**
      LightGBM `predict_proba` on a 1-row DataFrame is ~1–2 ms and
      fully CPU-bound. Calling it inside an async route handler blocks
      the event loop for that duration; we accept this rather than
      offloading to a thread pool. Saves the ~50 µs `asyncio.to_thread`
      overhead per request and keeps the call stack flat. SHAP
      TreeExplainer is similarly in-loop (~5–10 ms expected). If a
      future Sprint 5.x adds large-batch scoring or measured profiling
      shows the event loop stalling, switching to `asyncio.to_thread`
      is a one-line change.

    - **Custom per-stage Prometheus histograms + `prometheus-fastapi-instrumentator`.**
      The instrumentator gives us standard HTTP request rate /
      duration / size for free. Four custom Histograms
      (`fraud_engine_feature_fetch_seconds`, `_inference_seconds`,
      `_shap_seconds`, `_predict_total_seconds`) provide the per-stage
      breakdown the spec requires. Buckets `[0.005, 0.010, 0.025, 0.050, 0.100, 0.250]`
      cover the typical sub-10ms case, the 100 ms budget gate, and a
      tail. Both surfaces register against the same global REGISTRY
      and appear in the same `/metrics` scrape automatically — no
      coordination needed. Sprint 6's Grafana dashboards consume the
      breakdown directly.

    - **Per-stage timing via `time.perf_counter()` inline, not a decorator.**
      Two surfaces, two right answers: `@log_call` already gives us
      per-method `duration_ms` in structured logs (the right grain for
      incident triage); `time.perf_counter()` + `Histogram.observe()`
      gives us route-scoped Prometheus observations (the right grain
      for percentile dashboards). Re-implementing one via the other
      would lose either the structured-log drill-down or the standard
      Histogram bucket semantics.

    - **Middleware honours `X-Request-Id`; body's metadata.request_id
      is ignored in 5.1.f.** Middleware reads the header (if present),
      falls back to UUID4, and binds via `bind_request_id(rid)`. The
      route handlers don't touch request_id directly — every
      `@log_call`-decorated FeatureService / InferenceService /
      ShapExplainer call inherits the bound ID via the structlog
      ContextVar, giving us free per-request correlation across all
      ~40 log lines a single /predict generates. The body's optional
      `metadata.request_id` is currently ignored to avoid two-source
      precedence ambiguity; deferred to a Sprint 5.x prompt that wants
      to sort it out.

    - **`/predict` returns 200 even on `degraded_mode=True`.** Degraded
      mode is a partial answer, not an error. Clients can choose how
      to handle it (raise the threshold, route to a fallback rules
      engine, escalate to manual review). Returning a non-2xx would
      hide the model's still-useful Tier-1 prediction behind error
      handling that most clients would just retry. Only schema-validation
      failures (422) and unhandled exceptions (500) produce non-2xx
      codes.

    - **`/ready` returns 503 when any source is down; `/health` always
      200 if process is up.** Kubernetes-style two-probe model.
      `/health` is liveness (drives restart); `/ready` is readiness
      (drives traffic admission). A degraded-but-running service still
      serves `/predict` (with `degraded_mode=true`), but a load
      balancer in front would route around it for new connections —
      correct behaviour during a Redis outage.

Cross-references:
    - `src/fraud_engine/api/schemas.py` (5.1.a) — request/response shapes.
    - `src/fraud_engine/api/redis_store.py` (5.1.b) — async pool + MGET.
    - `src/fraud_engine/api/feature_service.py` (5.1.c) — `get_features`
      + `health_check`.
    - `src/fraud_engine/api/inference.py` (5.1.d) — `predict` + atomic
      reload.
    - `src/fraud_engine/api/shap_explainer.py` (5.1.e) — top-k + reason
      mapping.
    - `src/fraud_engine/utils/logging.py` — `bind_request_id` /
      `reset_request_id` ContextVar primitives.
    - `configs/prometheus/prometheus.yml:29-32` — the `fraud-api` scrape
      target this app exposes `/metrics` for.
    - `Makefile:62-63` — `make serve` entry point pointing here.
    - `CLAUDE.md` §3 (latency budget), §5.5 (logging discipline), §8
      (cost-optimal threshold).
"""

from __future__ import annotations

import dataclasses
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Final
from uuid import UUID, uuid4

import asyncpg  # type: ignore[import-untyped]  # asyncpg ships no type stubs (PEP-561 absent)
import redis.exceptions
from fastapi import FastAPI, Request, Response, status
from prometheus_client import Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from fraud_engine.api.feature_service import FeatureService
from fraud_engine.api.inference import InferenceService
from fraud_engine.api.redis_store import RedisFeatureStore
from fraud_engine.api.schemas import (
    HealthResponse,
    PredictionResponse,
    ReadyResponse,
    Reason,
    TransactionRequest,
)
from fraud_engine.api.shap_explainer import ShapExplainer
from fraud_engine.config.settings import Settings, get_settings
from fraud_engine.utils.logging import (
    bind_request_id,
    get_logger,
    get_request_id,
    reset_request_id,
)

# ---------------------------------------------------------------------
# Module constants.
# ---------------------------------------------------------------------

_SERVICE_NAME: Final[str] = "fraud-engine-api"
_SERVICE_VERSION: Final[str] = _pkg_version("fraud-engine")
# Top-k for the route-handler call into ShapExplainer. The schema caps
# `PredictionResponse.top_reasons` at 10; we request the same so the
# explainer doesn't have to think about it.
_TOP_K_REASONS: Final[int] = 10
# Histogram buckets covering typical sub-10ms case, the 100ms budget
# gate (CLAUDE.md §3), and a tail. Buckets are in seconds — Prometheus
# convention, NOT milliseconds.
_LATENCY_BUCKETS: Final[tuple[float, ...]] = (
    0.005,
    0.010,
    0.025,
    0.050,
    0.100,
    0.250,
)

_logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Per-stage Prometheus histograms.
#
# Registered against the global `prometheus_client.REGISTRY` so they
# appear in the same `/metrics` scrape that the
# `prometheus-fastapi-instrumentator` lib exposes. The four stages
# mirror the request flow exactly (feature fetch → inference → SHAP
# → total).
# ---------------------------------------------------------------------

FEATURE_FETCH_SECONDS: Final[Histogram] = Histogram(
    "fraud_engine_feature_fetch_seconds",
    "Time to fetch features (Tier-1 inline + Redis MGET + Postgres probe).",
    buckets=_LATENCY_BUCKETS,
)
INFERENCE_SECONDS: Final[Histogram] = Histogram(
    "fraud_engine_inference_seconds",
    "Time for LightGBM predict_proba + isotonic calibration.",
    buckets=_LATENCY_BUCKETS,
)
SHAP_SECONDS: Final[Histogram] = Histogram(
    "fraud_engine_shap_seconds",
    "Time for SHAP top-k contributions + reason mapping.",
    buckets=_LATENCY_BUCKETS,
)
PREDICT_TOTAL_SECONDS: Final[Histogram] = Histogram(
    "fraud_engine_predict_total_seconds",
    "End-to-end /predict latency (excludes network round-trip).",
    buckets=_LATENCY_BUCKETS,
)


# ---------------------------------------------------------------------
# AppState — the bundle held in `app.state.app_state` post-lifespan.
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class AppState:
    """Per-app live state populated by the lifespan startup.

    Frozen so a typo'd reassignment in a route handler fails loudly
    rather than silently shadowing the lifespan's bundle. Slots-based
    for ~5% access speedup over a dict-backed dataclass.

    Attributes:
        inference: Loaded InferenceService (LightGBM + calibrator).
        feature_service: Connected FeatureService (Redis + Postgres
            pools opened, or warning-logged if either was unreachable
            at startup).
        explainer: Loaded ShapExplainer (TreeExplainer + reason_codes
            YAML).
        settings: The active Settings — either the production singleton
            or the test factory's override.
    """

    inference: InferenceService
    feature_service: FeatureService
    explainer: ShapExplainer
    settings: Settings


# ---------------------------------------------------------------------
# Lifespan.
# ---------------------------------------------------------------------


def _make_lifespan(
    settings_override: Settings | None,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """Build a lifespan that closes over `settings_override`.

    A factory wrapping the lifespan is the cleanest way to thread a
    Settings override into `_lifespan` without a module-level mutable
    or `app.state`-as-pre-init-config. The returned function is what
    FastAPI's `lifespan=` kwarg consumes.

    Args:
        settings_override: If provided, the lifespan uses this instead
            of `get_settings()` — the test path. None → production path.

    Returns:
        An async context-manager function suitable for FastAPI's
        `lifespan=` parameter.
    """

    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Start: load model artefacts (fail-fast); connect dependencies (degrade-warn)."""
        settings = settings_override if settings_override is not None else get_settings()

        # Resolve the configs directory once. The default `parents[3]`
        # path-discovery in the per-module `_resolve_config_path`
        # helpers (redis_store / feature_service / shap_explainer) works
        # in dev (where the module lives at `<repo>/src/fraud_engine/api/`
        # — parents[3] gives `<repo>/`) but breaks in installed-package
        # contexts like Docker (where the module lives at
        # `/opt/venv/lib/python3.11/site-packages/fraud_engine/api/` —
        # parents[3] gives `/opt/venv/lib/python3.11/`, which has no
        # configs/). The Dockerfile sets WORKDIR=/app and COPY's
        # configs/ to /app/configs/. The cwd-relative `Path("configs")`
        # resolves correctly in both contexts: dev (run from repo root)
        # and container (cwd=/app). The FRAUD_ENGINE_CONFIG_DIR env var
        # lets an operator override for unusual layouts.
        config_dir_str = os.environ.get("FRAUD_ENGINE_CONFIG_DIR")
        config_dir = Path(config_dir_str) if config_dir_str else Path.cwd() / "configs"
        _logger.info(
            "lifespan.startup",
            redis_url=settings.redis_url,
            # Postgres URL contains the password; log only the host:port for safety.
            postgres_host=_postgres_host(settings.postgres_url),
            decision_threshold=settings.decision_threshold,
            config_dir=str(config_dir),
        )

        # Step 1: model artefacts. Fail-fast — a missing joblib is a
        # deployment bug, not a transient outage.
        inference = InferenceService(settings=settings)
        inference.load()

        # Step 2: ShapExplainer. Loads its own copy of the LightGBM
        # model (~50 ms duplicate joblib read; acceptable one-time
        # startup cost vs reaching into InferenceService's private
        # `_artefacts` field). Pass the explicit reason_codes path so
        # the resolver doesn't fall back to parents[3] (which breaks
        # under site-packages install — see config_dir comment above).
        explainer = ShapExplainer(reason_codes_path=config_dir / "reason_codes.yaml")

        # Step 3: feature_service. Lifespan tolerates Redis/Postgres
        # unreachable at startup — logs a WARNING and proceeds. The
        # FeatureService's per-call probe (5.1.c Decision #2) catches
        # the not-connected state at request time and flips
        # `degraded_mode=True` on the response.
        #
        # We construct an explicit RedisFeatureStore with the override
        # URL so the test factory's settings override flows through.
        # Without this, FeatureService(redis_store=None) would call
        # `RedisFeatureStore()` which reads from `get_settings()` — the
        # production singleton, not our override.
        redis_store = RedisFeatureStore(
            redis_url=settings.redis_url,
            ttl_config_path=config_dir / "redis_feature_store.yaml",
        )
        feature_service = FeatureService(
            redis_store=redis_store,
            postgres_url=settings.postgres_url,
            settings=settings,
            defaults_config_path=config_dir / "feature_defaults.yaml",
        )
        # The store hasn't been connected yet (we own it; FeatureService
        # only auto-connects when it owns the store). Connect both here
        # under one try/except so a single Redis-down or Postgres-down
        # outage logs one warning, not two.
        try:
            await redis_store.connect()
        except (
            redis.exceptions.ConnectionError,
            redis.exceptions.RedisError,
            OSError,
        ) as exc:
            _logger.warning(
                "lifespan.redis_unreachable",
                error_type=type(exc).__name__,
                detail=str(exc),
            )
        try:
            # FeatureService.connect() with an injected store opens only
            # the Postgres pool (the injected store is presumed
            # caller-managed). Either Postgres is up → succeeds, or
            # `asyncpg.create_pool()` raises here.
            await feature_service.connect()
        except (
            asyncpg.PostgresError,
            ConnectionRefusedError,
            OSError,
            RuntimeError,
            TimeoutError,
        ) as exc:
            _logger.warning(
                "lifespan.postgres_unreachable",
                error_type=type(exc).__name__,
                detail=str(exc),
            )

        app.state.app_state = AppState(
            inference=inference,
            feature_service=feature_service,
            explainer=explainer,
            settings=settings,
        )

        _logger.info("lifespan.ready", model_version=inference.model_version)
        try:
            yield
        finally:
            _logger.info("lifespan.shutdown")
            # Shutdown is best-effort; we log warnings on failure so a
            # leaked pool can't survive but a flaky Redis can't block
            # the process exit either.
            try:
                await feature_service.disconnect()
            except Exception as exc:  # noqa: BLE001 — defensive on shutdown
                _logger.warning(
                    "lifespan.shutdown_disconnect_error",
                    component="feature_service",
                    error_type=type(exc).__name__,
                    detail=str(exc),
                )
            try:
                await redis_store.disconnect()
            except Exception as exc:  # noqa: BLE001 — defensive on shutdown
                _logger.warning(
                    "lifespan.shutdown_disconnect_error",
                    component="redis_store",
                    error_type=type(exc).__name__,
                    detail=str(exc),
                )

    return _lifespan


def _postgres_host(url: str) -> str:
    """Extract `host:port` from a Postgres URL for safe logging.

    Strips the password component so the lifespan log doesn't leak
    credentials into the JSON record stream.

    Args:
        url: Postgres URL with optional credentials before an `@`
            separator (host:port lives between `@` and `/db`).

    Returns:
        The `host:port` portion (e.g. `localhost:5432`); the original
        URL unchanged if parsing fails (defensive — better to log a
        slightly noisy URL than crash on lifespan startup over a log
        line).
    """
    try:
        # Standard postgres URL layout splits on '@' to drop the
        # credentials prefix; everything after is host:port + path.
        after_at = url.split("@", 1)[1]
        return after_at.split("/", 1)[0]
    except (IndexError, AttributeError):
        return url


# ---------------------------------------------------------------------
# create_app factory.
# ---------------------------------------------------------------------


def create_app(settings: Settings | None = None) -> FastAPI:
    """Construct a FastAPI app, optionally with a Settings override.

    Production callers pass nothing → falls back to `get_settings()`
    (the lru-cached singleton). Test callers pass a tweaked Settings
    instance to spin a fresh, lifespan-isolated app — typically with
    `redis_url` pointed at an unreachable port for the degraded-mode
    test, or with `postgres_url` similarly tweaked.

    Args:
        settings: Optional Settings override. None → use
            `get_settings()` at lifespan time.

    Returns:
        A fully-wired FastAPI app — ready to be served by uvicorn or
        driven by an in-process `httpx.AsyncClient(transport=ASGITransport(app))`.
    """
    app = FastAPI(
        title="Fraud Detection API",
        version=_SERVICE_VERSION,
        description=(
            "Real-time transaction scoring with calibrated probability + "
            "SHAP-derived top reasons. P95 latency <100 ms (CLAUDE.md §3); "
            "degraded mode when Redis/Postgres unreachable."
        ),
        lifespan=_make_lifespan(settings),
    )

    # Standard HTTP-level Prometheus metrics. `should_ignore_untemplated=True`
    # prevents the `/metrics` endpoint itself from polluting the
    # request-rate counter (no recursive observation).
    Instrumentator(should_ignore_untemplated=True).instrument(app).expose(
        app, endpoint="/metrics", include_in_schema=False
    )

    # ----- middleware: bind request_id from header or generate ---------
    @app.middleware("http")
    async def _request_id_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Bind a per-request UUID to the structlog ContextVar.

        Reads `X-Request-Id` from the inbound headers if present and
        the value parses as a UUID; otherwise generates a UUID4 hex.
        Binds via `bind_request_id` so every `@log_call`-decorated
        FeatureService / InferenceService / ShapExplainer call
        inherits the ID automatically. Echoes the (now-canonical) ID
        on the response header so an upstream gateway can correlate
        request ↔ response.

        Why parse-as-UUID: the response schema (`PredictionResponse.request_id`)
        is typed `UUID`, so the route handler does `UUID(rid)` to
        construct the response. Accepting an arbitrary string in the
        header would break that conversion. We default to "trust but
        validate": parse the header; if it's a valid UUID accept it;
        if it's not, generate a fresh UUID4 and log a WARNING so the
        upstream gateway operator can fix their convention.

        The body's `metadata.request_id` (Sprint 5.1.a) is ignored in
        5.1.f to avoid two-source precedence ambiguity; deferred to a
        Sprint 5.x prompt.
        """
        header_rid = request.headers.get("X-Request-Id")
        if header_rid is not None:
            try:
                # Accept any UUID format (hex / dashed) — both parse
                # via UUID(...). The canonical form we bind + echo is
                # the hex form for compactness and structlog stability.
                rid = UUID(header_rid).hex
            except (ValueError, TypeError):
                rid = uuid4().hex
                _logger.warning(
                    "request_id_header_invalid",
                    received=header_rid,
                    generated=rid,
                )
        else:
            rid = uuid4().hex
        bind_request_id(rid)
        try:
            response = await call_next(request)
        finally:
            reset_request_id()
        response.headers["X-Request-Id"] = rid
        return response

    # ----- routes ------------------------------------------------------
    _register_routes(app)

    return app


# ---------------------------------------------------------------------
# Routes.
# ---------------------------------------------------------------------


def _register_routes(app: FastAPI) -> None:
    """Attach the four routes (`/health`, `/ready`, `/predict`).

    Defined as a separate function so the route bodies and the app
    construction read top-down. The Prometheus `/metrics` endpoint is
    registered by the `Instrumentator` in `create_app`.
    """

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["probes"],
        summary="Liveness probe — always 200 if the process is up.",
    )
    async def health() -> HealthResponse:
        """Return a static "ok" response.

        Liveness is binary: "the process is up and the import graph is
        intact". Kubernetes (or any orchestrator) probing this endpoint
        only cares about the HTTP status code; the body is for human
        inspection.
        """
        return HealthResponse(
            status="ok",
            service_name=_SERVICE_NAME,
            version=_SERVICE_VERSION,
        )

    @app.get(
        "/ready",
        response_model=ReadyResponse,
        tags=["probes"],
        summary="Readiness probe — 200 iff Redis + Postgres + model all OK.",
    )
    async def ready(response: Response) -> ReadyResponse:
        """Probe each runtime dependency; return 503 if any is down.

        Calls `feature_service.health_check()` (Redis PING + Postgres
        SELECT 1) and adds a `model` check that's "ok" iff
        InferenceService has loaded artefacts. Sets the response status
        to 503 if any check is not "ok" so a load balancer routes around
        the instance for new connections.
        """
        state: AppState = app.state.app_state
        # FeatureService.health_check returns dict[str, "ok"|"degraded"|"unreachable"].
        checks = await state.feature_service.health_check()
        # Add the model probe — if InferenceService loaded artefacts,
        # accessing model_version succeeds; otherwise it raises and we
        # mark unreachable.
        try:
            _ = state.inference.model_version
            checks["model"] = "ok"
        except RuntimeError:
            checks["model"] = "unreachable"

        all_ok = all(v == "ok" for v in checks.values())
        if not all_ok:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

        # Build the details map — populated only for non-ok checks so
        # the payload stays compact when the service is healthy.
        # `str(v)` widens the Literal value type to plain str for the
        # ReadyResponse.details field (dict[str, str]).
        details = {k: str(v) for k, v in checks.items() if v != "ok"}
        return ReadyResponse(
            status="ready" if all_ok else "not_ready",
            checks=checks,
            details=details,
        )

    @app.post(
        "/predict",
        response_model=PredictionResponse,
        tags=["scoring"],
        summary="Score a single transaction; return calibrated probability + SHAP reasons.",
    )
    async def predict(req: TransactionRequest) -> PredictionResponse:
        """End-to-end scoring: features → inference → SHAP → response.

        The hot path. Per-stage latency observed via Prometheus
        Histograms (`feature_fetch / inference / shap / total`) and
        per-method `duration_ms` via the `@log_call` decorator on each
        primitive. Returns 200 even when `degraded_mode=True` — the
        client decides how to handle a partial answer.

        Flow:
            1. Feature service: TransactionRequest → 743-column
               DataFrame. May flip degraded if Redis/Postgres down.
            2. Inference service: predict_proba → isotonic calibrate →
               threshold → InferenceResult.
            3. SHAP: top-k contributions → human-readable reasons.
            4. Assemble PredictionResponse.

        Returns:
            PredictionResponse with calibrated score, decision, top
            reasons, model version, latency_ms, and degraded_mode flag.
        """
        state: AppState = app.state.app_state
        t_total = time.perf_counter()

        # Stage 1: features.
        t = time.perf_counter()
        feature_vector = await state.feature_service.get_features(req)
        FEATURE_FETCH_SECONDS.observe(time.perf_counter() - t)

        # Stage 2: inference. Synchronous, CPU-bound, ~1–2 ms.
        t = time.perf_counter()
        inf = state.inference.predict(feature_vector.df)
        INFERENCE_SECONDS.observe(time.perf_counter() - t)

        # Stage 3: SHAP top-k + reason mapping.
        t = time.perf_counter()
        contribs = state.explainer.top_k_contributions(feature_vector.df, k=_TOP_K_REASONS)
        # Convert Contribution NamedTuple → Pydantic Reason. The
        # explainer uses `shap_value`; the response schema uses
        # `contribution`. The mapping is one-shot at the boundary.
        top_reasons = [
            Reason(
                feature_name=c.feature_name,
                contribution=c.shap_value,
                direction=c.direction,
            )
            for c in contribs
        ]
        SHAP_SECONDS.observe(time.perf_counter() - t)

        total_seconds = time.perf_counter() - t_total
        PREDICT_TOTAL_SECONDS.observe(total_seconds)

        # Resolve the request_id bound by middleware. Always set in our
        # request path; defensive UUID4 only for the (impossible) case
        # where middleware skipped binding.
        rid_str = get_request_id() or uuid4().hex
        return PredictionResponse(
            txn_id=req.TransactionID,
            request_id=UUID(rid_str),
            score=inf.probability,
            decision=inf.decision,
            top_reasons=top_reasons,
            latency_ms=total_seconds * 1000.0,
            model_version=inf.model_version,
            degraded_mode=feature_vector.degraded_mode,
        )


# ---------------------------------------------------------------------
# Module-level production app.
#
# `uvicorn fraud_engine.api.main:app` resolves to this. Tests don't
# touch it — they use `create_app(settings_override)` directly.
# ---------------------------------------------------------------------

app = create_app()


__all__ = ["AppState", "app", "create_app"]
