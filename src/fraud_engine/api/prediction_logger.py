"""Async fire-and-forget Postgres writer for the prediction audit log.

Sprint 5 prompt 5.2.a: the writer half of the audit-log surface that
Sprint 5.1.f's `/predict` route currently lacks. Every call to
`PredictionLogger.log(response)` schedules a background `asyncio.Task`
that INSERTs the response into the `predictions` table; the call
returns immediately, never blocking the HTTP handler that produced
the response. Sprint 5.2.b will wire this into `main.py`'s route.

Module surface (re-exported from `fraud_engine.api`):
    - PredictionLogger

Business rationale:
    A production fraud-detection service must audit every decision —
    regulators, model-monitoring (PSI / drift), incident-response, and
    customer-dispute workflows all depend on being able to look up "what
    did the model decide for transaction X at time T, with what model
    version, and what SHAP reasons drove that decision". Without an
    audit log, every prediction evaporates after the response is sent,
    making post-hoc analysis impossible.

    The "never blocks response" contract matters because the project's
    P95 latency budget is 100 ms (CLAUDE.md §3) and a synchronous
    Postgres write would burn ~5-10 ms per request. Fire-and-forget
    via `asyncio.create_task` lets the response return in <0.1 ms while
    the write completes in the background — the tail-latency penalty
    of a Postgres hiccup never reaches the API client.

Trade-offs considered:
    - **Fire-and-forget via `asyncio.create_task`, NOT a queue + worker.**
      The simpler primitive. ContextVars (request_id) are inherited by
      `create_task` so the background write's structlog lines carry the
      same correlation ID — free per-request log correlation. Per-task
      try/except wraps the write so a Postgres outage during background
      flush logs a warning + drops the record (the response was already
      sent). Rejected: a bounded queue + dedicated worker task — more
      machinery (lifespan needs to manage worker lifecycle) for a
      backpressure problem the project's 0.4 RPS sustained baseline
      doesn't have. The asyncpg pool's `max_size=5` provides natural
      concurrency limit at the database layer; excess in-flight tasks
      block on `pool.acquire()` within the task body, never on the
      route handler. Sprint 5.x can promote to a queue-and-worker if a
      high-RPS deployment ever materialises.

    - **Own asyncpg pool (NOT shared with FeatureService).** Decoupled
      failure modes — a FeatureService Postgres-side hiccup doesn't
      kill audit logging; an audit-log pool exhaustion doesn't
      degrade prediction quality. Cost: ~5 idle Postgres connections
      in production (1 min, 5 max). Pool sizing chosen from the same
      math as 5.1.c (Postgres SELECT 1 ≈ 10 ms; 5 slots × 100 RPS =
      500 RPS ceiling — comfortable for the project's RPS).

    - **Track `_pending_tasks: set[Task]` for graceful shutdown.**
      Without strong references to the spawned tasks, Python's GC
      can collect them mid-write (the only refs would be in the event
      loop's task queue, which is not a strong-reference container).
      `set.add(task)` keeps the strong ref; `add_done_callback(set.discard)`
      drops it post-completion. `disconnect()` awaits the remaining
      tasks with a timeout so a clean shutdown drains in-flight writes.
      This is the standard Python idiom for fire-and-forget GC safety
      (cited in PEP 3148 + asyncio docs).

    - **Per-task try/except catches `asyncpg.PostgresError`,
      `OSError`, `TimeoutError`, and `RuntimeError`.** A failed write
      cannot bubble up to the event loop — the response was already
      sent; the only valid action is "log a warning and drop". The
      structlog WARNING includes `request_id` so an operator can
      correlate the dropped log with the (already-served) response.

    - **JSON serialisation via `json.dumps([r.model_dump() for r in
      response.top_reasons])`.** Pydantic v2's `model_dump()` returns
      a plain dict (no Pydantic-specific types); `json.dumps()`
      handles the float / str / Literal trio cleanly; asyncpg's
      `$N::jsonb` cast handles the wire format. Rejected: pass the
      list of Reason instances directly — asyncpg doesn't know how
      to serialise Pydantic models.

    - **No retries.** A failed write is dropped, not retried. A retry
      loop would (a) double the worst-case write cost on a flaky
      Postgres connection, and (b) risk double-logging if the failure
      was actually a write-then-disconnect rather than a connect-then-fail.
      The audit log is best-effort; a missing row is a known-acceptable
      failure mode (the response was still served correctly). Sprint
      5.x can add a retry-with-jitter primitive if production
      observability shows non-trivial drop rates.

    - **Schema management via `scripts/create_predictions_table.sql`,
      NOT `Logger.ensure_table()`.** The logger expects the table to
      exist and crashes loudly (in the warning stream, not the route
      handler) if it doesn't. Schema management is a deployment
      concern (one-time `psql -f` step); coupling it to the runtime
      writer would require DDL privileges in production roles.

Cross-references:
    - `scripts/create_predictions_table.sql` — the schema this writer
      INSERTs against.
    - `src/fraud_engine/api/schemas.py:693-819` — the `PredictionResponse`
      contract that drives the column list.
    - `src/fraud_engine/api/feature_service.py:117,143-146,457-468` —
      the asyncpg pool construction + acquire/release pattern this
      class mirrors.
    - `src/fraud_engine/utils/logging.py:75-130` — `bind_request_id` /
      `get_request_id` ContextVar primitives that propagate via
      `asyncio.create_task`.
    - `CLAUDE.md` §3 (latency budget), §5.5 (logging discipline).
"""

from __future__ import annotations

import asyncio
import json
from types import TracebackType
from typing import Final

import asyncpg  # type: ignore[import-untyped]  # asyncpg ships no type stubs (PEP-561 absent)

from fraud_engine.api.schemas import PredictionResponse
from fraud_engine.config.settings import Settings, get_settings
from fraud_engine.utils.logging import get_logger, log_call

# ---------------------------------------------------------------------
# Module constants.
# ---------------------------------------------------------------------

# Pool sizing. 5 slots × 100 RPS/slot ≈ 500 RPS ceiling — comfortable
# for the project's economic-eval baseline (1M txns/month ≈ 0.4 RPS
# sustained; CLAUDE.md §8). Smaller than FeatureService's (2, 10) since
# logging is single-INSERT-per-request, not multi-query like FeatureService.
_DEFAULT_MIN_POOL: Final[int] = 1
_DEFAULT_MAX_POOL: Final[int] = 5

# Per-call pool acquire timeout. Aggressive: a fire-and-forget writer
# should not queue connections for long; if the pool is saturated, drop
# the record rather than hold the task open indefinitely.
_ACQUIRE_TIMEOUT_S: Final[float] = 1.0

# Graceful-shutdown drain timeout. `disconnect()` waits up to this
# many seconds for pending background writes to complete before
# closing the pool. 5 s covers a normal-latency Postgres + 4 s slack
# for the worst-case bounded queue depth (5 slots × ~5 ms per write).
_DRAIN_TIMEOUT_S: Final[float] = 5.0

# Single-row INSERT statement. Constants for prepared-statement-friendly
# tooling later; today asyncpg parses on every execute.
_INSERT_SQL: Final[str] = """
    INSERT INTO predictions (
        request_id, txn_id, client_id, score, decision,
        top_reasons, latency_ms, model_version, degraded_mode
    )
    VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8, $9)
"""

_logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Public class.
# ---------------------------------------------------------------------


class PredictionLogger:
    """Async fire-and-forget Postgres writer for the prediction audit log.

    Public API:
        - `connect()` / `disconnect()` — pool lifecycle.
        - `__aenter__` / `__aexit__` — `async with` support.
        - `log(response, *, client_id=None)` — schedule a background
          INSERT; returns immediately.

    Lifecycle:
        Constructor is cheap and side-effect-free. `connect()` opens
        the asyncpg pool. `log()` schedules a write via
        `asyncio.create_task` — does NOT await the write. `disconnect()`
        drains pending writes (with a timeout) then closes the pool.

    Schema:
        Defined by `scripts/create_predictions_table.sql`. Production
        deployment runs that file once via `psql -f`. The integration
        test fixture executes it on test-module setup.

    Business rationale + trade-offs considered: see module docstring.
    """

    def __init__(
        self,
        postgres_url: str | None = None,
        min_pool: int = _DEFAULT_MIN_POOL,
        max_pool: int = _DEFAULT_MAX_POOL,
        settings: Settings | None = None,
    ) -> None:
        """Configure the logger; does NOT open the pool.

        Args:
            postgres_url: Override `Settings.postgres_url`. None → use
                Settings.
            min_pool: asyncpg pool minimum size. Default 1.
            max_pool: asyncpg pool maximum size. Default 5.
            settings: Inject a Settings instance for tests. None →
                `get_settings()`.

        Raises:
            ValueError: If `min_pool` < 0 or `max_pool` < `min_pool`.
        """
        if min_pool < 0:
            raise ValueError(f"PredictionLogger: min_pool must be >= 0, got {min_pool}")
        if max_pool < min_pool:
            raise ValueError(
                f"PredictionLogger: max_pool ({max_pool}) must be >= " f"min_pool ({min_pool})"
            )
        self._settings: Settings = settings if settings is not None else get_settings()
        self._postgres_url: str = (
            postgres_url if postgres_url is not None else self._settings.postgres_url
        )
        self._min_pool: int = min_pool
        self._max_pool: int = max_pool
        self._pg_pool: asyncpg.Pool | None = None
        # Track in-flight write tasks. Strong references prevent GC of
        # the spawned coroutine before it completes (Python's standard
        # idiom for fire-and-forget — see PEP 3148 + asyncio docs).
        self._pending_tasks: set[asyncio.Task[None]] = set()

    # ---------- lifecycle ----------------------------------------------

    @log_call
    async def connect(self) -> None:
        """Open the asyncpg connection pool.

        Idempotent: a second call while already connected is a no-op.

        Raises:
            asyncpg.PostgresError / OSError: If Postgres is unreachable
                — `asyncpg.create_pool()` runs an immediate connection
                probe by default.
        """
        if self._pg_pool is not None:
            return
        self._pg_pool = await asyncpg.create_pool(
            self._postgres_url,
            min_size=self._min_pool,
            max_size=self._max_pool,
        )

    @log_call
    async def disconnect(self) -> None:
        """Drain pending writes (with a timeout) and close the pool.

        Idempotent. Awaits in-flight background tasks for up to
        `_DRAIN_TIMEOUT_S` seconds. Any task still running after the
        timeout is cancelled — the audit log is best-effort, not
        a transactional guarantee.
        """
        # Snapshot the pending tasks before draining; the
        # `add_done_callback(self._pending_tasks.discard)` will remove
        # entries as they complete, but we want a stable view.
        pending = list(self._pending_tasks)
        if pending:
            done, not_done = await asyncio.wait(
                pending,
                timeout=_DRAIN_TIMEOUT_S,
                return_when=asyncio.ALL_COMPLETED,
            )
            for task in not_done:
                task.cancel()
                _logger.warning(
                    "prediction_logger.shutdown_drain_timeout",
                    pending_remaining=len(not_done),
                )
        if self._pg_pool is not None:
            await self._pg_pool.close()
            self._pg_pool = None

    async def __aenter__(self) -> PredictionLogger:
        """Open the pool on context entry."""
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Drain pending writes + close the pool on context exit."""
        await self.disconnect()

    # ---------- the hot path -------------------------------------------

    @log_call
    def log(
        self,
        response: PredictionResponse,
        *,
        client_id: str | None = None,
    ) -> None:
        """Schedule an asynchronous Postgres INSERT for this prediction.

        Returns immediately. The actual INSERT runs in a background
        `asyncio.Task` and may complete after the calling route handler
        has already returned its HTTP response — this is the
        "never blocks" contract this method exists to provide.

        If the pool isn't connected, logs a WARNING and drops the
        record (no exception propagates). If the background INSERT
        fails (Postgres outage, malformed payload), the per-task
        try/except in `_write_one` logs a WARNING and drops the
        record — the response has already been served, so the only
        valid action is "log + move on".

        Args:
            response: The PredictionResponse the API is about to send.
            client_id: Optional client identifier from
                `RequestMetadata.client_id`. Stored as the `client_id`
                column on the row.
        """
        if self._pg_pool is None:
            _logger.warning(
                "prediction_logger.not_connected_drop",
                request_id=str(response.request_id),
            )
            return
        task = asyncio.create_task(self._write_one(response, client_id))
        # Strong ref + GC cleanup. Without the set.add, the task is
        # reference-counted only by the event loop's task queue, which
        # is not a strong-reference container — Python may GC the task
        # before it completes.
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

    async def _write_one(
        self,
        response: PredictionResponse,
        client_id: str | None,
    ) -> None:
        """Background task body — the actual INSERT.

        All exceptions are caught + logged + swallowed. The
        fire-and-forget contract requires that a failed write cannot
        propagate to the event loop (the response was already sent).
        """
        # Pool may have been closed between schedule and execute (e.g.,
        # during a graceful shutdown). Drop the record cleanly.
        if self._pg_pool is None:
            _logger.warning(
                "prediction_logger.pool_closed_during_write",
                request_id=str(response.request_id),
            )
            return
        try:
            top_reasons_json = json.dumps([reason.model_dump() for reason in response.top_reasons])
            async with self._pg_pool.acquire(timeout=_ACQUIRE_TIMEOUT_S) as conn:
                await conn.execute(
                    _INSERT_SQL,
                    response.request_id,
                    response.txn_id,
                    client_id,
                    response.score,
                    response.decision,
                    top_reasons_json,
                    response.latency_ms,
                    response.model_version,
                    response.degraded_mode,
                )
        except (asyncpg.PostgresError, OSError, TimeoutError, RuntimeError) as exc:
            # Audit log is best-effort. Don't crash the event loop.
            _logger.warning(
                "prediction_logger.write_failed",
                request_id=str(response.request_id),
                txn_id=response.txn_id,
                error_type=type(exc).__name__,
                detail=str(exc),
            )
        except asyncio.CancelledError:
            # Re-raise so the cancellation propagates to the gather() in
            # `disconnect()` — otherwise we'd hide the cancellation from
            # the shutdown sequence.
            _logger.warning(
                "prediction_logger.write_cancelled",
                request_id=str(response.request_id),
            )
            raise


__all__ = ["PredictionLogger"]
