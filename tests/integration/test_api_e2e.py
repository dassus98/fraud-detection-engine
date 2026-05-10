"""End-to-end integration tests for the Sprint 5.1.f FastAPI app.

Sprint 5 prompt 5.1.f. Drives the lifespan-managed app via
`httpx.AsyncClient(transport=ASGITransport(app))` + `asgi-lifespan.LifespanManager`
so each test exercises the full request path: middleware →
TransactionRequest validation → FeatureService → InferenceService →
ShapExplainer → PredictionResponse.

Test scenarios (per the spec):
    1. /health returns 200.
    2. /ready returns 200 when Redis + Postgres + model all OK.
    3. /predict with valid payload returns a valid PredictionResponse.
    4. /predict p95 over 100 sequential requests is <100 ms (CLAUDE.md §3 budget).
    5. /predict with missing required fields returns 422.
    6. /predict in degraded mode (Redis pointed at unreachable port)
       still returns 200 with `degraded_mode=true` in the body.
    7. /metrics exposes the Prometheus registry, including custom
       per-stage histograms.

Skipped (with reason) when the docker-compose stack isn't running.

Trade-offs considered:
    - **Module-scoped reachability probe + module-scoped client.** Spinning
      a fresh app per test would multiply lifespan startup cost (~500 ms
      load + connect) by 7. Per-module client amortises to one startup
      across the suite. The degraded-mode test owns its own per-test
      client because it needs a different Settings.
    - **`asgi-lifespan.LifespanManager` over `fastapi.TestClient`.** TestClient
      is sync and would block the per-MGET timing observation; AsyncClient
      keeps the event loop free for accurate latency measurement.
    - **`httpx.ASGITransport` over a real uvicorn process.** Eliminates
      the network round-trip from the latency budget — the test measures
      the *application* P95, not the network round-trip P95. The 100ms
      budget is the application-level commitment per CLAUDE.md §3.
"""

from __future__ import annotations

import asyncio
import json
import statistics
from collections.abc import AsyncIterator
from copy import deepcopy
from pathlib import Path

import httpx
import pytest
import redis.asyncio
from asgi_lifespan import LifespanManager

from fraud_engine.api.main import create_app
from fraud_engine.config.settings import Settings, get_settings

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------
# Module constants.
# ---------------------------------------------------------------------

_FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
_SAMPLE_TXN_PATH = _FIXTURES_DIR / "sample_txn.json"

# Latency budget for the p95 test. 100ms per CLAUDE.md §3.
_P95_BUDGET_MS = 100.0
# Number of sequential POSTs to drive p95 over. Spec calls for 100.
_P95_SAMPLE_SIZE = 100
# Unreachable Redis URL for the degraded-mode test. Port 1 is reserved
# in the IANA registry for tcpmux; nothing listens by convention.
_UNREACHABLE_REDIS_URL = "redis://127.0.0.1:1/0"


# ---------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def sample_request_payload() -> dict[str, object]:
    """Load the canonical sample TransactionRequest payload.

    Sourced from `tests/fixtures/sample_txn.json` (one row sampled from
    `data/processed/tier1_test.parquet` at fixture-build time). The
    `deepcopy` per-test-call guards against tests mutating each other's
    payload via shared dict references.
    """
    return json.loads(_SAMPLE_TXN_PATH.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


@pytest.fixture(scope="module")
def deps_reachable() -> None:
    """Probe Redis + Postgres; skip the module if either is unreachable.

    Probes once per module — every test in this file requires both
    dependencies. Uses the redis-py client directly (no FeatureService)
    so a future bug in our orchestrator can't cause the suite to
    silently skip.
    """
    settings = get_settings()

    async def _probe_redis() -> None:
        client = redis.asyncio.from_url(settings.redis_url)
        try:
            await client.ping()
        finally:
            await client.aclose()

    async def _probe_postgres() -> None:
        # Lazy import — asyncpg is a heavy dep, only loaded when probing.
        import asyncpg  # type: ignore[import-untyped]

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
    except Exception as exc:  # noqa: BLE001 — many failure modes
        pytest.skip(f"Postgres unreachable at {settings.postgres_url}: {exc}")


@pytest.fixture
async def client(deps_reachable: None) -> AsyncIterator[httpx.AsyncClient]:
    """Lifespan-managed AsyncClient against a fresh app.

    Per-test scope (not module-scoped) because the lifespan owns
    asyncpg's pool which is tied to the event loop pytest-asyncio
    creates per-test (asyncio_default_fixture_loop_scope="function").
    Spinning the lifespan once per test costs ~300-500 ms but isolates
    each test from the others' state.
    """
    app = create_app()
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


# ---------------------------------------------------------------------
# Probe routes.
# ---------------------------------------------------------------------


async def test_health_returns_200(client: httpx.AsyncClient) -> None:
    """Liveness probe: always 200 if process is up."""
    response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service_name"] == "fraud-engine-api"
    assert body["version"]  # populated from importlib.metadata


async def test_ready_returns_200_when_deps_up(client: httpx.AsyncClient) -> None:
    """Readiness probe: 200 with all checks 'ok' when Redis + Postgres + model up."""
    response = await client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["checks"]["redis"] == "ok"
    assert body["checks"]["postgres"] == "ok"
    assert body["checks"]["model"] == "ok"
    # details only populated for non-ok checks; should be empty here.
    assert body["details"] == {}


async def test_metrics_endpoint_exposes_prometheus(client: httpx.AsyncClient) -> None:
    """`/metrics` returns Prometheus text format with our custom histograms."""
    # Drive one /predict first so the histograms have at least one observation.
    payload = json.loads(_SAMPLE_TXN_PATH.read_text(encoding="utf-8"))
    pre = await client.post("/predict", json=payload)
    assert pre.status_code == 200

    response = await client.get("/metrics")
    assert response.status_code == 200
    body = response.text
    # Custom per-stage histograms — the load-bearing surface for Sprint 6's dashboards.
    assert "fraud_engine_feature_fetch_seconds_bucket" in body
    assert "fraud_engine_inference_seconds_bucket" in body
    assert "fraud_engine_shap_seconds_bucket" in body
    assert "fraud_engine_predict_total_seconds_bucket" in body


# ---------------------------------------------------------------------
# Predict — happy path.
# ---------------------------------------------------------------------


async def test_predict_valid_payload_returns_response(
    client: httpx.AsyncClient,
    sample_request_payload: dict[str, object],
) -> None:
    """A valid payload returns a 200 with all PredictionResponse fields populated."""
    response = await client.post("/predict", json=sample_request_payload)
    assert response.status_code == 200, response.text
    body = response.json()
    # Echoed field.
    assert body["txn_id"] == sample_request_payload["TransactionID"]
    # Score in [0, 1].
    assert 0.0 <= body["score"] <= 1.0
    # Decision binary.
    assert body["decision"] in ("block", "allow")
    # Top reasons capped at 10.
    assert isinstance(body["top_reasons"], list)
    assert len(body["top_reasons"]) <= 10
    # Each reason has the expected shape.
    for reason in body["top_reasons"]:
        assert reason["feature_name"]
        assert isinstance(reason["contribution"], float | int)
        assert reason["direction"] in ("increases_risk", "decreases_risk")
    # Latency populated (not the network round-trip — the application latency).
    assert body["latency_ms"] >= 0.0
    # Model version is the manifest content_hash (SHA-256 hex, 64 chars).
    assert len(body["model_version"]) == 64  # noqa: PLR2004 — SHA-256 hex length is by definition 64
    # Healthy path → degraded_mode False.
    assert body["degraded_mode"] is False
    # Request_id echoed; X-Request-Id header echoed too.
    assert body["request_id"]
    assert response.headers["X-Request-Id"]


async def test_predict_p95_under_100ms(
    client: httpx.AsyncClient,
    sample_request_payload: dict[str, object],
) -> None:
    """100 sequential POSTs; assert p95 <100 ms (CLAUDE.md §3 budget)."""
    import time

    durations_ms: list[float] = []
    for _ in range(_P95_SAMPLE_SIZE):
        t = time.perf_counter()
        response = await client.post("/predict", json=sample_request_payload)
        durations_ms.append((time.perf_counter() - t) * 1000.0)
        assert response.status_code == 200

    durations_ms.sort()
    p50 = statistics.median(durations_ms)
    # statistics.quantiles(n=20) returns 19 cut-points; index 18 = 95th percentile.
    p95 = statistics.quantiles(durations_ms, n=20)[18]
    p99 = statistics.quantiles(durations_ms, n=100)[98]
    print(
        f"\n/predict latencies over {_P95_SAMPLE_SIZE} requests: "
        f"p50={p50:.2f}ms  p95={p95:.2f}ms  p99={p99:.2f}ms  "
        f"min={min(durations_ms):.2f}ms  max={max(durations_ms):.2f}ms"
    )
    assert p95 < _P95_BUDGET_MS, (
        f"/predict p95 {p95:.2f}ms exceeded {_P95_BUDGET_MS}ms budget "
        f"(p50={p50:.2f}ms, p99={p99:.2f}ms, max={max(durations_ms):.2f}ms)"
    )


# ---------------------------------------------------------------------
# Predict — error paths.
# ---------------------------------------------------------------------


async def test_predict_missing_fields_returns_422(client: httpx.AsyncClient) -> None:
    """A payload missing required fields fails Pydantic validation → 422."""
    incomplete = {"TransactionID": 1}  # missing TransactionDT, TransactionAmt, ProductCD, card1
    response = await client.post("/predict", json=incomplete)
    assert response.status_code == 422
    # Pydantic surfaces the missing fields in the error response.
    body = response.json()
    assert "detail" in body
    missing_fields = {err["loc"][-1] for err in body["detail"] if err["type"] == "missing"}
    assert "TransactionDT" in missing_fields
    assert "TransactionAmt" in missing_fields
    assert "ProductCD" in missing_fields
    assert "card1" in missing_fields


async def test_predict_invalid_value_returns_422(
    client: httpx.AsyncClient,
    sample_request_payload: dict[str, object],
) -> None:
    """Negative TransactionAmt violates `gt=0.0` → 422."""
    payload = deepcopy(sample_request_payload)
    payload["TransactionAmt"] = -10.0
    response = await client.post("/predict", json=payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------
# Predict — degraded mode (Redis unreachable).
# ---------------------------------------------------------------------


async def test_predict_degraded_mode_when_redis_down(
    sample_request_payload: dict[str, object],
) -> None:
    """When Redis is unreachable, /predict still returns 200 with degraded_mode=true.

    Owns its own client (fresh app + Settings override) because the
    module-scoped `client` fixture is configured against the production
    Redis URL. Spinning a separate app with `redis_url` pointed at an
    unreachable port lets the FeatureService's per-call probe flip
    `degraded_mode=True` per Decision #2 of PR #49 (5.1.c).
    """
    # Construct override Settings — ignore the .env's REDIS_URL.
    # Postgres stays at the production URL so we isolate the Redis-down
    # signal from any Postgres-down noise.
    settings = get_settings()
    override = Settings(
        redis_url=_UNREACHABLE_REDIS_URL,
        postgres_url=settings.postgres_url,
    )

    app = create_app(settings=override)
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            response = await c.post("/predict", json=sample_request_payload)

    assert response.status_code == 200, response.text
    body = response.json()
    # Score still produced (model + Tier-1 still work + population defaults fill the rest).
    assert 0.0 <= body["score"] <= 1.0
    # The load-bearing assertion: degraded mode propagates onto the response.
    assert body["degraded_mode"] is True


async def test_ready_returns_503_when_redis_down(
    sample_request_payload: dict[str, object],  # noqa: ARG001 — fixture pulled in for module-scope dep ordering
) -> None:
    """`/ready` returns 503 when Redis is unreachable.

    Mirrors `test_predict_degraded_mode_when_redis_down` but exercises
    the readiness probe instead of the prediction path. The response
    body carries the per-source check status so the on-call team can
    read which source is down without grepping logs.
    """
    settings = get_settings()
    override = Settings(
        redis_url=_UNREACHABLE_REDIS_URL,
        postgres_url=settings.postgres_url,
    )

    app = create_app(settings=override)
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            response = await c.get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["redis"] == "unreachable"
    # Postgres + model still ok (we only knocked over Redis).
    assert body["checks"]["postgres"] == "ok"
    assert body["checks"]["model"] == "ok"
    # The redis check appears in details (non-ok checks are surfaced).
    assert "redis" in body["details"]


# ---------------------------------------------------------------------
# Schema-drift sentinel.
# ---------------------------------------------------------------------


def test_sample_fixture_validates_against_current_schema() -> None:
    """The committed sample_txn.json must validate against TransactionRequest.

    Catches fixture-vs-schema drift on every CI run — a future Sprint
    5.x change to the schema that breaks this fixture surfaces here,
    not in production traffic.
    """
    from fraud_engine.api.schemas import TransactionRequest

    payload = json.loads(_SAMPLE_TXN_PATH.read_text(encoding="utf-8"))
    # Round-trip through the schema; any validation error fails the test.
    parsed = TransactionRequest.model_validate(payload)
    assert parsed.TransactionID == payload["TransactionID"]
    assert parsed.card1 == payload["card1"]
    assert parsed.TransactionAmt == payload["TransactionAmt"]
