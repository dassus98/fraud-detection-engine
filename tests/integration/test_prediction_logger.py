"""Integration tests for `PredictionLogger` against real Postgres.

Sprint 5 prompt 5.2.a. Mirrors the integration-test convention from
`test_redis_store_integration.py` and `test_feature_service.py`:
module-scoped reachability probe + `pytest.skip` if Postgres is down,
per-test UUID4 namespace for isolation, post-test DELETE for cleanup.

Test scenarios (per the spec):
    1. `test_log_round_trip_matches_schema` — single write; read back
       via raw asyncpg; assert all 9 columns match exactly.
    2. `test_top_reasons_json_round_trip` — three Reasons in top_reasons;
       round-trip via JSONB; assert each Reason's fields survive bit-exactly.
    3. `test_concurrent_writes_dont_block_each_other` — 50 logs fired
       concurrently; the per-call schedule time stays well under the
       per-call write time (the "never blocks" gate); all 50 rows land.
    4. `test_log_returns_immediately_on_postgres_down` — pool pointed
       at unreachable port; `log()` doesn't crash; structlog WARNING.
    5. `test_log_no_op_before_connect` — `log()` before `connect()`;
       no crash; structlog WARNING.
    6. `test_lifecycle_async_context_manager` — `async with logger:
       logger.log(...)`; pool opened + closed; pending tasks drained
       on exit.

Trade-offs considered:
    - **Per-test UUID4 namespace + post-test DELETE.** Tests stay
      isolated even when run concurrently (parallel pytest workers).
      No transaction rollback (asyncpg transactions don't compose
      cleanly with the fire-and-forget pattern).
    - **One-shot module-scoped DDL fixture.** Runs
      `scripts/create_predictions_table.sql` once per module via
      idempotent CREATE TABLE IF NOT EXISTS — safe even if the table
      already exists from a prior dev run.
    - **`asyncio.gather(*pending)` to drain background writes** in the
      assertion phase of test #3 — the "never blocks" gate is the
      schedule-vs-execute timing comparison; the post-drain SELECT
      verifies all 50 rows landed.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import UUID, uuid4

import asyncpg  # type: ignore[import-untyped]  # asyncpg ships no type stubs (PEP-561 absent)
import pytest

from fraud_engine.api.prediction_logger import PredictionLogger
from fraud_engine.api.schemas import PredictionResponse, Reason
from fraud_engine.config.settings import get_settings

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------
# Module constants.
# ---------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA_SQL_PATH = _PROJECT_ROOT / "scripts" / "create_predictions_table.sql"

# Unreachable Postgres URL for the "Postgres down" test. Port 1 is
# reserved (tcpmux); nothing should listen. The user/password are dev
# defaults that match docker-compose.dev.yml.
_UNREACHABLE_POSTGRES_URL = "postgresql://fraud:fraud@127.0.0.1:1/fraud"  # pragma: allowlist secret — dev defaults pointed at unreachable port

# Concurrent writes — sized so the "schedule time" is dominated by
# `create_task` overhead and the "drain time" is dominated by the
# Postgres write time. 50 keeps test wall-clock under 1 s.
_CONCURRENT_N = 50

# "Never blocks" gate: the wall-clock spent inside the for-loop of
# `log()` calls must be well under the per-write Postgres latency.
# A loopback Postgres write is ~1-3 ms; `create_task` schedule is
# ~10-50 µs. 100 ms ceiling on the schedule loop is comfortably above
# 50 calls × 50 µs ≈ 2.5 ms but well below 50 × 3 ms = 150 ms (which
# would be the synchronous baseline).
_SCHEDULE_BUDGET_MS = 100.0

# Post-write SELECT count budget; gives the background writes time to
# complete after `gather` returns.
_POST_DRAIN_SETTLE_S = 0.5


# ---------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def postgres_url() -> str:
    """Resolve Postgres URL from Settings; skip module if unreachable."""
    settings = get_settings()
    url = settings.postgres_url

    async def _probe() -> None:
        conn = await asyncpg.connect(url, timeout=2.0)
        try:
            await conn.fetchval("SELECT 1")
        finally:
            await conn.close()

    try:
        asyncio.run(_probe())
    except Exception as exc:  # noqa: BLE001 — many failure modes
        pytest.skip(f"Postgres unreachable at {url}: {exc}")
    return url


@pytest.fixture(scope="module")
def predictions_table_ready(postgres_url: str) -> str:
    """Create the `predictions` table once per module via the canonical SQL.

    Idempotent: the SQL uses `CREATE TABLE IF NOT EXISTS` and `CREATE
    INDEX IF NOT EXISTS`, so re-running on an already-populated table
    leaves existing rows untouched. We don't drop the table at teardown
    so that a developer running the suite locally can `psql` into the
    database and inspect what was logged.
    """
    sql_text = _SCHEMA_SQL_PATH.read_text(encoding="utf-8")

    async def _setup() -> None:
        conn = await asyncpg.connect(postgres_url, timeout=2.0)
        try:
            await conn.execute(sql_text)
        finally:
            await conn.close()

    asyncio.run(_setup())
    return postgres_url


@pytest.fixture
async def logger(
    predictions_table_ready: str,
) -> AsyncIterator[PredictionLogger]:
    """A connected PredictionLogger; disconnects + drains on test exit."""
    log = PredictionLogger(postgres_url=predictions_table_ready)
    await log.connect()
    try:
        yield log
    finally:
        await log.disconnect()


@pytest.fixture
async def pg_conn(
    predictions_table_ready: str,
) -> AsyncIterator[asyncpg.Connection]:
    """A raw asyncpg connection for round-trip readback assertions."""
    conn = await asyncpg.connect(predictions_table_ready, timeout=2.0)
    try:
        yield conn
    finally:
        await conn.close()


def _make_response(  # noqa: PLR0913 — eight knobs map 1:1 to PredictionResponse fields; folding into a dict obscures the per-field test-customisation surface
    request_id: UUID | None = None,
    *,
    txn_id: int = 2987000,
    score: float = 0.0273,
    decision: str = "allow",
    top_reasons: list[Reason] | None = None,
    latency_ms: float = 47.21,
    model_version: str = "a3f8c2d9b1e7c5d4f8a2b6e9c1d5f7a3b8e9c2d4f6a8b1e3c5d7f9a2b4c6d8e1",  # pragma: allowlist secret — synthetic SHA-256 fake for tests
    degraded_mode: bool = False,
) -> PredictionResponse:
    """Construct a PredictionResponse with sensible defaults + overrides."""
    return PredictionResponse(
        txn_id=txn_id,
        request_id=request_id if request_id is not None else uuid4(),
        score=score,
        decision=decision,  # type: ignore[arg-type]  # Literal narrowing is at runtime
        top_reasons=top_reasons if top_reasons is not None else [],
        latency_ms=latency_ms,
        model_version=model_version,
        degraded_mode=degraded_mode,
    )


async def _delete_test_rows(
    pg_conn: asyncpg.Connection,
    request_ids: list[UUID],
) -> None:
    """Per-test teardown: remove the rows this test wrote."""
    if not request_ids:
        return
    await pg_conn.execute(
        "DELETE FROM predictions WHERE request_id = ANY($1::uuid[])",
        request_ids,
    )


# ---------------------------------------------------------------------
# Scenario 1: schema-match round-trip.
# ---------------------------------------------------------------------


async def test_log_round_trip_matches_schema(
    logger: PredictionLogger,
    pg_conn: asyncpg.Connection,
) -> None:
    """All 9 columns round-trip exactly: schema mirror is correct."""
    request_id = uuid4()
    response = _make_response(
        request_id=request_id,
        txn_id=3485113,
        score=0.123,
        decision="allow",
        top_reasons=[
            Reason(
                feature_name="card1_velocity_24h", contribution=0.42, direction="increases_risk"
            ),
            Reason(feature_name="tier1_amount_log", contribution=-0.15, direction="decreases_risk"),
        ],
        latency_ms=68.12,
        model_version="990ef848fb8bf578a31a6baf659e8757db189359c59beb9a14d6c67f22f0cf26",  # pragma: allowlist secret — production model_version, not a credential
        degraded_mode=False,
    )
    client_id = "wealthsimple-prod"

    logger.log(response, client_id=client_id)
    # Drain pending tasks so the row lands before we SELECT.
    await asyncio.gather(*list(logger._pending_tasks))  # noqa: SLF001 — test-only access

    try:
        row = await pg_conn.fetchrow(
            "SELECT * FROM predictions WHERE request_id = $1",
            request_id,
        )
        assert row is not None, "the logged row was not found in Postgres"
        # 9 mirrored columns + 2 audit columns (id, created_at).
        assert row["request_id"] == request_id
        assert row["txn_id"] == response.txn_id
        assert row["client_id"] == client_id
        assert row["score"] == pytest.approx(response.score)
        assert row["decision"] == response.decision
        # `top_reasons` is JSONB; asyncpg returns it as a JSON-decoded
        # list of dicts. The shape must match what we'd get from
        # `[r.model_dump() for r in response.top_reasons]`.
        loaded = json.loads(row["top_reasons"])
        assert loaded == [r.model_dump() for r in response.top_reasons]
        assert row["latency_ms"] == pytest.approx(response.latency_ms)
        assert row["model_version"] == response.model_version
        assert row["degraded_mode"] == response.degraded_mode
        # Audit columns set by Postgres / BIGSERIAL.
        assert row["id"] > 0
        assert row["created_at"] is not None
    finally:
        await _delete_test_rows(pg_conn, [request_id])


# ---------------------------------------------------------------------
# Scenario 2: JSONB top_reasons fidelity.
# ---------------------------------------------------------------------


async def test_top_reasons_json_round_trip(
    logger: PredictionLogger,
    pg_conn: asyncpg.Connection,
) -> None:
    """Each Reason's 3 fields round-trip via JSONB without loss or coercion."""
    request_id = uuid4()
    reasons = [
        Reason(
            feature_name="card1_fraud_v_ewm_lambda_0.05",
            contribution=0.937,
            direction="increases_risk",
        ),
        Reason(
            feature_name="P_emaildomain_target_enc",
            contribution=-0.357,
            direction="decreases_risk",
        ),
        Reason(
            feature_name="V137",
            contribution=0.076,
            direction="increases_risk",
        ),
    ]
    response = _make_response(request_id=request_id, top_reasons=reasons)

    logger.log(response)
    await asyncio.gather(*list(logger._pending_tasks))  # noqa: SLF001 — test-only access

    try:
        raw_json = await pg_conn.fetchval(
            "SELECT top_reasons FROM predictions WHERE request_id = $1",
            request_id,
        )
        loaded = json.loads(raw_json)
        assert len(loaded) == len(reasons)
        for got, expected in zip(loaded, reasons, strict=True):
            assert got["feature_name"] == expected.feature_name
            assert got["contribution"] == pytest.approx(expected.contribution)
            assert got["direction"] == expected.direction
    finally:
        await _delete_test_rows(pg_conn, [request_id])


# ---------------------------------------------------------------------
# Scenario 3: concurrent writes don't block each other.
# ---------------------------------------------------------------------


async def test_concurrent_writes_dont_block_each_other(
    logger: PredictionLogger,
    pg_conn: asyncpg.Connection,
) -> None:
    """50 fire-and-forget logs schedule in << per-write time; all rows land.

    The load-bearing assertion: the for-loop of `log()` calls completes
    in well under the time it would take to do 50 synchronous Postgres
    writes (~150 ms at 3 ms/write on loopback). Schedule cost is per-call
    `asyncio.create_task` overhead (~10-50 µs) so 50 calls should finish
    in <100 ms even with structlog overhead.
    """
    request_ids = [uuid4() for _ in range(_CONCURRENT_N)]
    responses = [_make_response(request_id=rid) for rid in request_ids]

    # The "never blocks" measurement: schedule loop wall-clock.
    t0 = time.perf_counter()
    for response in responses:
        logger.log(response)
    schedule_wall_ms = (time.perf_counter() - t0) * 1000

    # Drain pending writes.
    await asyncio.gather(*list(logger._pending_tasks))  # noqa: SLF001 — test-only access
    # Settle: a few ms for Postgres to finish committing.
    await asyncio.sleep(_POST_DRAIN_SETTLE_S)

    try:
        # Verify all 50 rows landed.
        count = await pg_conn.fetchval(
            "SELECT COUNT(*) FROM predictions WHERE request_id = ANY($1::uuid[])",
            request_ids,
        )
        assert count == _CONCURRENT_N, (
            f"expected {_CONCURRENT_N} rows; got {count}. "
            f"schedule_wall_ms={schedule_wall_ms:.2f}"
        )
        # The "never blocks" gate.
        assert schedule_wall_ms < _SCHEDULE_BUDGET_MS, (
            f"schedule loop wall-clock {schedule_wall_ms:.2f}ms exceeded "
            f"{_SCHEDULE_BUDGET_MS}ms budget — log() is synchronously blocking"
        )
        print(
            f"\nschedule_loop_wall={schedule_wall_ms:.2f}ms for "
            f"{_CONCURRENT_N} log() calls "
            f"({schedule_wall_ms / _CONCURRENT_N * 1000:.1f}µs / call)"
        )
    finally:
        await _delete_test_rows(pg_conn, request_ids)


# ---------------------------------------------------------------------
# Scenario 4: Postgres down — graceful drop, no crash.
# ---------------------------------------------------------------------


async def test_log_returns_immediately_on_postgres_down(
    predictions_table_ready: str,  # noqa: ARG001 — pulled in for module-scope dep ordering
) -> None:
    """Pool pointed at port 1; `log()` doesn't crash; write fails silently."""
    log = PredictionLogger(postgres_url=_UNREACHABLE_POSTGRES_URL)

    # `connect()` itself fails on the unreachable URL (asyncpg pool runs
    # an eager probe). We expect the failure; exercise the "log() before
    # connect" path which is the production lifespan-degraded mode.
    with pytest.raises((asyncpg.PostgresError, OSError, ConnectionRefusedError)):
        await log.connect()

    # Even though connect() failed, log() should not crash — it sees
    # `_pg_pool is None` and emits a structlog WARNING.
    response = _make_response()
    log.log(response)  # no assertion; the gate is "no exception"

    # Disconnect is a no-op when there's nothing to drain.
    await log.disconnect()


# ---------------------------------------------------------------------
# Scenario 5: log before connect — graceful no-op.
# ---------------------------------------------------------------------


async def test_log_no_op_before_connect(
    predictions_table_ready: str,
) -> None:
    """Calling `log()` on an un-connected logger emits a WARNING, no crash."""
    log = PredictionLogger(postgres_url=predictions_table_ready)
    response = _make_response()
    # Pool was never opened → no_pg_pool branch.
    log.log(response)  # the gate is "no exception"
    # No-op disconnect (nothing was connected).
    await log.disconnect()


# ---------------------------------------------------------------------
# Scenario 6: async context manager.
# ---------------------------------------------------------------------


async def test_lifecycle_async_context_manager(
    predictions_table_ready: str,
    pg_conn: asyncpg.Connection,
) -> None:
    """`async with PredictionLogger() as log: log.log(...)` opens + drains + closes."""
    request_id = uuid4()
    response = _make_response(request_id=request_id)

    async with PredictionLogger(postgres_url=predictions_table_ready) as log:
        # Pool opened.
        assert log._pg_pool is not None  # noqa: SLF001 — test-only access
        log.log(response)
        # _aexit__ drains pending tasks via disconnect's drain timeout.

    # After context exit, pool closed; pending tasks drained.
    assert log._pg_pool is None  # noqa: SLF001 — test-only access

    try:
        # Row should be present (drain completed before pool close).
        count = await pg_conn.fetchval(
            "SELECT COUNT(*) FROM predictions WHERE request_id = $1",
            request_id,
        )
        assert count == 1, f"expected 1 row post-context-exit; got {count}"
    finally:
        await _delete_test_rows(pg_conn, [request_id])
