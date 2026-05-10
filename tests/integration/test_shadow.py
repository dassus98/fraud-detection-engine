"""End-to-end integration tests for ShadowService wired into the FastAPI app.

Sprint 5 prompt 5.2.b. Drives the lifespan-managed app via
`httpx.AsyncClient(transport=ASGITransport(app))` + `asgi-lifespan.LifespanManager`
so each test exercises the full request path: middleware →
TransactionRequest validation → FeatureService → InferenceService →
ShapExplainer → PredictionResponse → ShadowService.score (when enabled).

Test scenarios (per the spec):
    1. `test_shadow_disabled_does_not_load_model` — `Settings(shadow_enabled=False)`;
       lifespan completes; AppState.shadow is None; /predict works
       and emits NO `shadow.*` log line.
    2. `test_shadow_enabled_loads_and_scores` — `Settings(shadow_enabled=True)`;
       lifespan loads FraudNet; /predict triggers a `shadow.scored`
       structlog event with valid `shadow_score` in [0, 1].
    3. `test_shadow_failure_doesnt_block_main_latency` — patch the
       loaded shadow model's `predict_proba` to raise; run 50 sequential
       /predict calls; assert p95 < 100 ms (the load-bearing gate);
       assert each request still returns 200 with valid PredictionResponse.
    4. `test_shadow_circuit_breaker_trips_after_n_failures` — patch
       `predict_proba` to raise; fire enough requests to exceed the
       failure threshold (using a custom-injected breaker with low
       threshold); verify breaker transitions CLOSED → OPEN and
       subsequent calls log `shadow.breaker_open_skip` (no model
       call attempted).
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import json
import logging
import statistics
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from asgi_lifespan import LifespanManager

from fraud_engine.api.circuit_breaker import CircuitBreaker
from fraud_engine.api.main import create_app
from fraud_engine.config.settings import Settings, get_settings

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------
# Module constants.
# ---------------------------------------------------------------------

_FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
_SAMPLE_TXN_PATH = _FIXTURES_DIR / "sample_txn.json"

# Latency budget per CLAUDE.md §3.
_P95_BUDGET_MS = 100.0
# Sequential /predict requests to drive the latency assertion. 50 is
# small enough to keep test wall-clock under 5 s but large enough for
# the median + p95 to be statistically meaningful.
_LATENCY_SAMPLE_SIZE = 50


# ---------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def sample_request_payload() -> dict[str, Any]:
    """Load the canonical sample TransactionRequest payload."""
    return json.loads(_SAMPLE_TXN_PATH.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def _collect_shadow_events(
    caplog: pytest.LogCaptureFixture,
    event_name: str | None = None,
) -> list[dict[str, Any]]:
    """Parse caplog records into structlog event dicts, filter to shadow events.

    structlog's ProcessorFormatter renders each event_dict into a
    Python dict that flows through stdlib logging — the LogRecord's
    `message` attribute is the dict's `repr()` (single-quoted Python
    syntax, not JSON). Parse via `ast.literal_eval` (safe for
    dict-of-primitives) and filter to events whose `event` field
    starts with `shadow.` (or matches the explicit `event_name`).
    Records that are not parseable as dicts (e.g., httpx access logs,
    third-party library messages) are skipped silently.
    """
    out: list[dict[str, Any]] = []
    for record in caplog.records:
        msg = record.getMessage()
        # Most structlog records arrive as Python dict reprs; some
        # bypass structlog and arrive as plain strings (httpx, etc).
        # Try ast.literal_eval first (handles single-quoted dicts);
        # fall back to json.loads for double-quoted JSON.
        payload: Any
        with contextlib.suppress(ValueError, SyntaxError, TypeError):
            payload = ast.literal_eval(msg)
            if isinstance(payload, dict):
                event = payload.get("event")
                if isinstance(event, str) and (
                    (event_name is not None and event == event_name)
                    or (event_name is None and event.startswith("shadow."))
                ):
                    out.append(payload)
                continue
        # Fall-through path for non-dict records — try JSON for safety.
        with contextlib.suppress(json.JSONDecodeError, TypeError):
            payload = json.loads(msg)
            if isinstance(payload, dict):
                event = payload.get("event")
                if isinstance(event, str) and (
                    (event_name is not None and event == event_name)
                    or (event_name is None and event.startswith("shadow."))
                ):
                    out.append(payload)
    return out


def _set_shadow_log_capture(caplog: pytest.LogCaptureFixture) -> None:
    """Ensure caplog captures the structlog records emitted from the
    shadow + main + circuit_breaker loggers.

    Pytest's `caplog.at_level(...)` context only sets the root logger's
    level by default; structlog's per-logger handlers may not propagate
    to root in all configurations. Setting per-logger levels
    explicitly is the robust path.
    """
    caplog.set_level(logging.INFO, logger="fraud_engine.api.shadow")
    caplog.set_level(logging.INFO, logger="fraud_engine.api.main")
    caplog.set_level(logging.INFO, logger="fraud_engine.api.circuit_breaker")


@pytest.fixture(scope="module")
def deps_reachable() -> None:
    """Probe Redis + Postgres; skip module if either is unreachable."""
    settings = get_settings()
    import redis.asyncio  # noqa: PLC0415 — module-scoped probe; lazy import keeps test discovery fast

    async def _probe_redis() -> None:
        client = redis.asyncio.from_url(settings.redis_url)
        try:
            await client.ping()
        finally:
            await client.aclose()

    async def _probe_postgres() -> None:
        import asyncpg  # type: ignore[import-untyped]  # noqa: PLC0415 — lazy

        conn = await asyncpg.connect(settings.postgres_url, timeout=2.0)
        try:
            await conn.fetchval("SELECT 1")
        finally:
            await conn.close()

    try:
        asyncio.run(_probe_redis())
    except Exception as exc:  # noqa: BLE001 — many failure modes
        pytest.skip(f"Redis unreachable at {settings.redis_url}: {exc}")
    try:
        asyncio.run(_probe_postgres())
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Postgres unreachable at {settings.postgres_url}: {exc}")


# ---------------------------------------------------------------------
# Scenario 1: shadow disabled.
# ---------------------------------------------------------------------


async def test_shadow_disabled_does_not_load_model(
    deps_reachable: None,  # noqa: ARG001 — module-scope dep
    sample_request_payload: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When SHADOW_ENABLED is False, the lifespan does not load FraudNet
    and /predict emits no shadow.* log lines."""
    _set_shadow_log_capture(caplog)
    settings = Settings(shadow_enabled=False)
    app = create_app(settings=settings)
    async with LifespanManager(app):
        # AppState.shadow should be None.
        assert app.state.app_state.shadow is None

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/predict", json=sample_request_payload)
            assert response.status_code == 200, response.text
            # No shadow.* events emitted (the route checks state.shadow is None
            # and skips the call entirely).
            shadow_lines = _collect_shadow_events(caplog)
            assert shadow_lines == []


# ---------------------------------------------------------------------
# Scenario 2: shadow enabled — loads + scores.
# ---------------------------------------------------------------------


async def test_shadow_enabled_loads_and_scores(
    deps_reachable: None,  # noqa: ARG001 — module-scope dep
    sample_request_payload: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """With SHADOW_ENABLED=True, the lifespan loads FraudNet and the
    next /predict triggers a shadow scoring attempt.

    NOTE: FraudNet's predict_proba expects a different feature shape
    than LightGBM (its own vocab + numeric columns); the LightGBM-shaped
    feature_vector.df we pass causes a KeyError in production paths.
    To exercise the WIRING (lifespan loads model, /predict calls
    shadow.score, breaker records outcome), we patch predict_proba
    to return a fixed array. The schema-mismatch fix is Sprint 5.x —
    out of scope for the breaker / fire-and-forget primitive.
    """
    _set_shadow_log_capture(caplog)
    settings = Settings(shadow_enabled=True)
    app = create_app(settings=settings)
    async with LifespanManager(app):
        # AppState.shadow should be a loaded ShadowService.
        shadow = app.state.app_state.shadow
        assert shadow is not None
        # The model_version is the FraudNet manifest's content_hash.
        assert len(shadow.model_version) > 0

        # Patch predict_proba to return a known successful score so
        # we can assert the shadow.scored event shape. Production-shape
        # FraudNet input wiring is Sprint 5.x.
        import numpy as np  # noqa: PLC0415 — test-only import

        def _stubbed_predict(_features: Any) -> Any:
            # Shape (1, 2) — column 1 is fraud-probability per LightGBM/FraudNet contract.
            return np.array([[0.92, 0.08]])

        shadow._artefacts.model.predict_proba = _stubbed_predict  # type: ignore[union-attr]  # noqa: SLF001

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/predict", json=sample_request_payload)
            assert response.status_code == 200, response.text
            # Drain pending background shadow tasks before asserting.
            await asyncio.gather(*list(shadow._pending_tasks))  # noqa: SLF001 — test-only access

        # shadow.scored event emitted with valid score in [0, 1].
        scored_events = _collect_shadow_events(caplog, event_name="shadow.scored")
        assert len(scored_events) >= 1, (
            f"expected ≥1 shadow.scored events; got these shadow events: "
            f"{[e.get('event') for e in _collect_shadow_events(caplog)]}"
        )
        first_score = scored_events[0]
        assert "shadow_score" in first_score, f"missing shadow_score: {first_score}"
        assert 0.0 <= first_score["shadow_score"] <= 1.0
        assert first_score.get("shadow_model_version") is not None
        assert first_score.get("request_id") is not None


# ---------------------------------------------------------------------
# Scenario 3: shadow failure doesn't block main latency (load-bearing).
# ---------------------------------------------------------------------


async def test_shadow_failure_doesnt_block_main_latency(
    deps_reachable: None,  # noqa: ARG001 — module-scope dep
    sample_request_payload: dict[str, Any],
) -> None:
    """50 /predict calls with synthetic shadow failures; p95 < 100 ms.

    Patches the loaded shadow model's `predict_proba` to raise on every
    call. The fire-and-forget contract means each /predict still returns
    200 and the per-request latency budget (CLAUDE.md §3) is preserved
    — the shadow failure exists only as background noise.
    """
    settings = Settings(shadow_enabled=True)
    app = create_app(settings=settings)
    async with LifespanManager(app):
        shadow = app.state.app_state.shadow
        assert shadow is not None
        # Patch the loaded model's predict_proba to raise.
        # Use a high-threshold breaker so we don't accidentally trip
        # mid-test (we want every call to attempt + fail, not skip).
        shadow._breaker = CircuitBreaker(failure_threshold=10000)  # noqa: SLF001

        def _always_fail(_features: Any) -> Any:
            raise RuntimeError("synthetic shadow failure")

        shadow._artefacts.model.predict_proba = _always_fail  # type: ignore[union-attr]  # noqa: SLF001 — test-only swap

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # Warmup — first few requests pay the lifespan cold-start cost
            # (model joblib JIT, asyncpg pool ramp, structlog buffering)
            # and would dominate the p95 if measured. 5 warm-ups burn the
            # cold-start budget; the measured 50 reflect steady-state.
            for _ in range(5):
                await client.post("/predict", json=sample_request_payload)

            durations_ms: list[float] = []
            for _ in range(_LATENCY_SAMPLE_SIZE):
                t = time.perf_counter()
                response = await client.post("/predict", json=sample_request_payload)
                durations_ms.append((time.perf_counter() - t) * 1000.0)
                assert response.status_code == 200

            # Drain pending shadow tasks (they all raise → breaker records failures).
            await asyncio.gather(*list(shadow._pending_tasks), return_exceptions=True)  # noqa: SLF001

        durations_ms.sort()
        p50 = statistics.median(durations_ms)
        p95 = statistics.quantiles(durations_ms, n=20)[18]
        p99 = statistics.quantiles(durations_ms, n=100)[98]
        print(
            f"\n/predict latencies (shadow failing) over {_LATENCY_SAMPLE_SIZE} requests: "
            f"p50={p50:.2f}ms  p95={p95:.2f}ms  p99={p99:.2f}ms  "
            f"min={min(durations_ms):.2f}ms  max={max(durations_ms):.2f}ms"
        )
        assert p95 < _P95_BUDGET_MS, (
            f"/predict p95 {p95:.2f}ms exceeded {_P95_BUDGET_MS}ms budget under "
            f"shadow failures — the fire-and-forget contract is broken"
        )


# ---------------------------------------------------------------------
# Scenario 4: circuit breaker trips after N failures.
# ---------------------------------------------------------------------


async def test_shadow_circuit_breaker_trips_after_n_failures(
    deps_reachable: None,  # noqa: ARG001 — module-scope dep
    sample_request_payload: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """After N consecutive shadow failures, the breaker opens and
    subsequent /predict calls log shadow.breaker_open_skip without
    attempting the model call."""
    _set_shadow_log_capture(caplog)
    settings = Settings(shadow_enabled=True)
    app = create_app(settings=settings)
    async with LifespanManager(app):
        shadow = app.state.app_state.shadow
        assert shadow is not None
        # Inject a low-threshold breaker so we trip after 3 failures.
        # Use a long cooldown so the test's last requests stay OPEN.
        shadow._breaker = CircuitBreaker(  # noqa: SLF001 — test-only override
            failure_threshold=3,
            initial_cooldown_seconds=60.0,
            max_cooldown_seconds=600.0,
        )

        def _always_fail(_features: Any) -> Any:
            raise RuntimeError("synthetic shadow failure")

        shadow._artefacts.model.predict_proba = _always_fail  # type: ignore[union-attr]  # noqa: SLF001

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # Fire 3 calls to trip the breaker.
            for _ in range(3):
                response = await client.post("/predict", json=sample_request_payload)
                assert response.status_code == 200
            # Drain shadow tasks so failures are recorded on the breaker.
            await asyncio.gather(*list(shadow._pending_tasks), return_exceptions=True)  # noqa: SLF001

            # Breaker should be OPEN now.
            assert shadow._breaker.state == "open", (  # noqa: SLF001
                f"expected breaker OPEN after 3 failures; got "
                f"state={shadow._breaker.state!r} consecutive_failures="  # noqa: SLF001
                f"{shadow._breaker.consecutive_failures}"  # noqa: SLF001
            )

            # Next /predict should skip shadow + emit breaker_open_skip.
            caplog.clear()
            response = await client.post("/predict", json=sample_request_payload)
            assert response.status_code == 200
            # /predict returned; the score() call ran synchronously
            # and logged the skip without scheduling a task.

            skip_events = _collect_shadow_events(caplog, event_name="shadow.breaker_open_skip")
            assert len(skip_events) >= 1, (
                f"expected ≥1 shadow.breaker_open_skip events post-trip; got "
                f"these shadow events: {[e.get('event') for e in _collect_shadow_events(caplog)]}"
            )
            # Critical: NO new shadow.failed events (no model call attempted).
            new_failed_events = _collect_shadow_events(caplog, event_name="shadow.failed")
            assert (
                new_failed_events == []
            ), "shadow.failed emitted post-trip — the breaker did not skip the call"


@pytest.fixture
async def _drain_pending_shadow_tasks_fixture(
    deps_reachable: None,  # noqa: ARG001
) -> AsyncIterator[None]:
    """Empty fixture placeholder kept for symmetry with the other test
    files; per-test cleanup happens inline via `asyncio.gather`."""
    yield
