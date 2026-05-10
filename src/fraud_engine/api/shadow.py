"""Async fire-and-forget challenger-model scoring (shadow mode).

Sprint 5 prompt 5.2.b: the shadow surface that lets us run Model B
(FraudNet, the entity-embedding NN trained in Sprint 3.2.a) alongside
the production Model A (LightGBM) without affecting served decisions.
Every `/predict` call (when `Settings.shadow_enabled=True`) fires a
background `ShadowService.score(features, request_id, champion_score)`
that runs FraudNet's `predict_proba` in a thread (so the CPU-bound
torch call doesn't stall the event loop) and emits a structured
`shadow.scored` log line carrying both the champion and challenger
probabilities. Offline analysis (`jq` over the structlog stream) can
compute champion-vs-challenger AUC, agreement rate, and decision
divergence — the data feed for production model-monitoring.

The shadow path is wrapped in a `CircuitBreaker` (Sprint 5.2.b
sibling): after N consecutive failures, the breaker opens and
subsequent `score()` calls are silently skipped (no model call
attempted) until the cooldown elapses and a probe succeeds. The
breaker's exponential-backoff cooldown handles sustained outages
gracefully — we don't burn CPU on a known-broken challenger.

Module surface (re-exported from `fraud_engine.api`):
    - ShadowService

Business rationale:
    Production model rollouts need shadow data before traffic
    promotion. The classic question is "would Model B have caught
    that fraud Model A missed (or vice-versa)?". Without a shadow
    surface, the only way to answer it is to retrain offline on
    historical features — but that misses the live-feature
    distribution + the per-request entity state. The shadow service
    captures real-traffic challenger predictions during normal
    operation; offline analysis joins the structlog stream by
    request_id to compare champion vs challenger pair-wise.

    The "never blocks" contract matters because the project's P95
    latency budget is 100 ms (CLAUDE.md §3) and FraudNet's per-row
    `predict_proba` is ~2 ms on CPU. A synchronous call would be
    invisible at typical loads but blow the budget if the challenger
    ever degrades (e.g., a future deeper model). Fire-and-forget +
    `asyncio.to_thread` + circuit-breaker keeps the main path
    isolated from challenger pathology.

Trade-offs considered:
    - **Fire-and-forget mirroring `PredictionLogger` (5.2.a).** The
      `_pending_tasks: set[asyncio.Task]` + `add_done_callback(set.discard)`
      idiom is the textbook fire-and-forget GC-safety pattern.
      Per-task try/except catches all expected failure types and
      logs via structlog. `disconnect()` drains pending tasks with
      a timeout for graceful shutdown.

    - **`asyncio.to_thread` for `predict_proba`, NOT direct call on
      the event loop.** FraudNet is CPU-bound torch tensor work
      (~2 ms). Sprint 5.1.f's main inference runs in-loop because
      it's ~2 ms AND it's the whole point of the request. Shadow is
      best-effort and runs alongside the main path; offloading to a
      thread costs ~50 µs of event-loop overhead but eliminates the
      head-of-line risk if FraudNet ever takes >10 ms (e.g., a
      future deeper model). Also lets concurrent `/predict` requests
      schedule their shadow scoring on the default thread-pool
      executor without blocking each other.

    - **CircuitBreaker on the call path.** Sustained challenger
      failures (process crash loop, OOM, model-load corruption)
      shouldn't burn predict cycles + log volume in perpetuity. The
      breaker trips after 5 consecutive failures, refuses calls for
      30 s, then probes; on probe failure, doubles the cooldown.
      Maps directly onto the spec's "exponential backoff + circuit
      breaker on failure".

    - **Atomic-swap `_ShadowArtefacts` mirroring `InferenceService`
      (5.1.d).** Frozen dataclass holds (model, content_hash); a
      single attribute rebind is GIL-atomic; mid-session reload via
      `load()` works the same way main-model reload does. Sprint 5.x
      can ship a `POST /admin/reload-shadow` endpoint that calls
      `shadow.load()` to swap without restart.

    - **Output to structlog, NOT Postgres.** The structured-log
      stream is the audit surface — the `shadow.scored` event
      carries `request_id`, `champion_score`, `shadow_score`,
      `shadow_model_version`, `agree_decision`, `duration_ms`. An
      offline `jq` pass joins by `request_id` against the main
      `prediction.logged` stream (Sprint 5.2.a) for full
      champion-vs-challenger comparison. Rejected: extending the
      `predictions` Postgres table with `shadow_*` columns —
      couples the audit-log schema to a feature flag (rows would
      be NULL when shadow is disabled) and doubles per-prediction
      write volume; defer Postgres persistence to Sprint 5.x.

    - **Degrade-warn on load failure** (handled in main.py's
      lifespan, not here): a missing FraudNet artefact logs a
      WARNING at startup; the API still serves Model A predictions.
      This module's `load()` raises `FileNotFoundError`/`RuntimeError`
      on missing artefacts — the lifespan catches and degrades.

Cross-references:
    - `src/fraud_engine/api/circuit_breaker.py` (Sprint 5.2.b
      sibling) — the failure-isolation primitive.
    - `src/fraud_engine/api/prediction_logger.py` (Sprint 5.2.a) —
      fire-and-forget `_pending_tasks` + drain pattern this class
      mirrors.
    - `src/fraud_engine/api/inference.py` (Sprint 5.1.d) —
      `_Artefacts` frozen dataclass + atomic-swap pattern this class
      mirrors.
    - `src/fraud_engine/models/neural_model.py:480-1020` — FraudNetModel
      + `load(path)` + `predict_proba(X)` API.
    - `CLAUDE.md` §3 (latency budget), §5.5 (logging discipline).
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import time
from pathlib import Path
from types import TracebackType
from typing import Any, Final

import pandas as pd

from fraud_engine.api.circuit_breaker import CircuitBreaker
from fraud_engine.api.schemas import DecisionLiteral
from fraud_engine.config.settings import Settings, get_settings
from fraud_engine.models.neural_model import FraudNetModel
from fraud_engine.utils.logging import get_logger, log_call

# ---------------------------------------------------------------------
# Module constants.
# ---------------------------------------------------------------------

_DEFAULT_MODEL_DIR: Final[Path] = Path("models/sprint3/fraudnet")
_MODEL_FILENAME: Final[str] = "neural_model.pt"
_MANIFEST_FILENAME: Final[str] = "neural_model_manifest.json"

# Graceful-shutdown drain timeout. Mirrors PredictionLogger (5.2.a):
# 5 s covers normal-latency challenger predict + slack for the worst
# case bounded queue depth (default executor 8 threads × ~5 ms / call).
_DRAIN_TIMEOUT_S: Final[float] = 5.0

_logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Frozen artefact bundle for atomic swap.
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class _ShadowArtefacts:
    """Bundle of (model, content_hash) for atomic swap.

    Held in `ShadowService._artefacts`; replaced wholesale on `load()`.
    Frozen + slots for the same reasons as `InferenceService._Artefacts`
    (Sprint 5.1.d Decision 1): immutability under concurrent reads,
    minimal memory, fast slot-based attribute access.
    """

    model: FraudNetModel
    content_hash: str


# ---------------------------------------------------------------------
# Module-private helpers.
# ---------------------------------------------------------------------


def _load_artefacts(model_dir: Path) -> _ShadowArtefacts:
    """Read FraudNetModel + manifest content_hash.

    Args:
        model_dir: Directory containing `neural_model.pt` +
            `neural_model_manifest.json`.

    Returns:
        Frozen `_ShadowArtefacts` bundle.

    Raises:
        FileNotFoundError: If either artefact file is missing.
        ValueError: If the manifest is missing the `content_hash` key
            or it is empty.
    """
    model_path = model_dir / _MODEL_FILENAME
    manifest_path = model_dir / _MANIFEST_FILENAME

    if not model_path.is_file():
        raise FileNotFoundError(
            f"ShadowService: FraudNet artefact not found at {model_path} — "
            f"run `uv run python scripts/train_neural_model.py` first."
        )
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"ShadowService: FraudNet manifest not found at {manifest_path} — "
            f"manifest is written alongside the joblib by `FraudNetModel.save()`."
        )

    model = FraudNetModel.load(model_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    content_hash = manifest.get("content_hash")
    if not isinstance(content_hash, str) or not content_hash:
        raise ValueError(
            f"ShadowService: manifest at {manifest_path} missing or "
            f"malformed `content_hash` field; got {content_hash!r}"
        )
    return _ShadowArtefacts(model=model, content_hash=content_hash)


# ---------------------------------------------------------------------
# Public class.
# ---------------------------------------------------------------------


class ShadowService:
    """Async fire-and-forget challenger-model scoring with circuit breaker.

    Public API:
        - `load()` — read FraudNet artefacts + atomic-swap.
        - `disconnect()` — drain pending shadow tasks (with timeout).
        - `__aenter__` / `__aexit__` — `async with` support.
        - `score(features, *, request_id, champion_score=None)` — fire
          a background shadow scoring; returns immediately.

    Read-only properties:
        - `model_version` — current `_ShadowArtefacts.content_hash`
          (raises RuntimeError if not loaded).
        - `breaker` — the CircuitBreaker for diagnostic inspection.

    Lifecycle:
        Constructor is cheap and side-effect-free. `load()` opens the
        artefacts (raises on missing files; main.py's lifespan catches
        and degrade-warns). `score()` schedules a write via
        `asyncio.create_task` — does NOT await the prediction.
        `disconnect()` drains pending tasks with a timeout.

    Business rationale + trade-offs considered: see module docstring.
    """

    def __init__(
        self,
        model_dir: Path | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Configure the service; does NOT load the model.

        Args:
            model_dir: Override the FraudNet artefacts directory.
                None → `models/sprint3/fraudnet/`.
            circuit_breaker: Inject a pre-configured `CircuitBreaker`
                (e.g., for tests with shorter cooldowns). None →
                construct a default with project-tuned thresholds.
            settings: Inject a Settings instance for tests. None →
                `get_settings()`. (Currently unused but kept for
                consistency with the other services' constructor
                signatures.)
        """
        self._settings: Settings = settings if settings is not None else get_settings()
        self._model_dir: Path = model_dir if model_dir is not None else _DEFAULT_MODEL_DIR
        self._breaker: CircuitBreaker = (
            circuit_breaker if circuit_breaker is not None else CircuitBreaker()
        )
        # Artefacts populated by `load()` / `reload()`. None → not loaded.
        self._artefacts: _ShadowArtefacts | None = None
        # Track in-flight scoring tasks. Strong references prevent GC
        # of spawned coroutines before they complete (Python's standard
        # idiom for fire-and-forget — see PEP 3148 + asyncio docs).
        self._pending_tasks: set[asyncio.Task[None]] = set()

    # ---------- lifecycle ----------------------------------------------

    @log_call
    def load(self) -> None:
        """Read FraudNet artefacts from disk and atomic-swap.

        Idempotent: a second call replaces the in-memory bundle (same
        atomic-swap semantic as `InferenceService.reload()` in 5.1.d).

        Raises:
            FileNotFoundError: If either artefact is missing. Lifespan
                in main.py catches this and degrade-warns.
            ValueError: If the manifest's `content_hash` is missing/empty.
            RuntimeError: Propagated from `FraudNetModel.load` (e.g.,
                joblib deserialisation failure).
        """
        new_artefacts = _load_artefacts(self._model_dir)
        # Single-attribute rebind — GIL-atomic. A concurrent `_score_one`
        # binding `local = self._artefacts` at task top sees either the
        # old bundle (if its bind happens before this assignment) or
        # the new one (after); never a partially-replaced view.
        self._artefacts = new_artefacts

    @log_call
    async def disconnect(self) -> None:
        """Drain pending shadow tasks (with a timeout).

        Idempotent. Awaits in-flight background tasks for up to
        `_DRAIN_TIMEOUT_S` seconds. Any task still running after the
        timeout is cancelled — shadow is best-effort, not a
        transactional guarantee.
        """
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
                    "shadow.shutdown_drain_timeout",
                    pending_remaining=len(not_done),
                )

    async def __aenter__(self) -> ShadowService:
        """Enter context. Caller is responsible for calling load() separately."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Drain pending tasks on context exit."""
        await self.disconnect()

    # ---------- read-only accessors ------------------------------------

    @property
    def model_version(self) -> str:
        """The currently-loaded shadow model's `content_hash`.

        Raises:
            RuntimeError: If `load()` has not been called.
        """
        if self._artefacts is None:
            raise RuntimeError("ShadowService: call load() before accessing model_version.")
        return self._artefacts.content_hash

    @property
    def breaker(self) -> CircuitBreaker:
        """The CircuitBreaker for diagnostic inspection (state, counters)."""
        return self._breaker

    # ---------- the hot path -------------------------------------------

    @log_call
    def score(
        self,
        features: pd.DataFrame,
        *,
        request_id: str,
        champion_score: float | None = None,
        champion_decision: DecisionLiteral | None = None,
    ) -> None:
        """Schedule an asynchronous shadow scoring for this request.

        Returns immediately. The actual `predict_proba` runs in a
        background `asyncio.Task` that offloads to a thread (so the
        torch CPU work doesn't stall the event loop). The result is
        logged via structlog with the `request_id` correlation tag —
        this is the audit surface for offline champion-vs-challenger
        comparison.

        If the model isn't loaded (load failed or shadow disabled),
        the call is a silent no-op. If the circuit breaker is open,
        the call logs a `shadow.breaker_open_skip` info event (no
        model call attempted). Otherwise schedules the background
        task; a per-task try/except catches all expected failures and
        records breaker outcomes appropriately.

        Args:
            features: Single-row DataFrame matching FraudNet's input
                contract (typically the FeatureVector.df from 5.1.c).
            request_id: The bound request_id for correlation.
            champion_score: Optional champion (Model A) probability
                for direct logging alongside the challenger score.
            champion_decision: Optional champion decision string for
                computing the agreement bit.
        """
        if self._artefacts is None:
            # Not loaded — silent no-op (degrade-warn already
            # logged at lifespan startup).
            return
        if not self._breaker.can_proceed():
            _logger.info(
                "shadow.breaker_open_skip",
                request_id=request_id,
                breaker_state=self._breaker.state,
                consecutive_failures=self._breaker.consecutive_failures,
            )
            return
        task = asyncio.create_task(
            self._score_one(features, request_id, champion_score, champion_decision)
        )
        # Strong ref + GC cleanup. Without the set.add, the task is
        # reference-counted only by the event loop's task queue, which
        # is not a strong-reference container — Python may GC the task
        # before it completes.
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

    async def _score_one(
        self,
        features: pd.DataFrame,
        request_id: str,
        champion_score: float | None,
        champion_decision: DecisionLiteral | None,
    ) -> None:
        """Background task body — the actual `predict_proba` + log.

        All exceptions are caught + logged + breaker-failure-recorded
        + swallowed. The fire-and-forget contract requires that a
        failed shadow prediction cannot propagate to the event loop
        (the response was already sent).
        """
        local = self._artefacts
        if local is None:  # pragma: no cover — defensive; load() should have populated
            _logger.warning("shadow.artefacts_none_in_score", request_id=request_id)
            return

        t0 = time.perf_counter()
        try:
            # Run torch's CPU-bound predict_proba in the default thread
            # pool so the event loop stays responsive. Returns ndarray
            # of shape (1, 2); column 1 is fraud-probability — same
            # contract as LightGBM (Sprint 5.1.d).
            proba_array: Any = await asyncio.to_thread(local.model.predict_proba, features)
            shadow_score = float(proba_array[0, 1])
            duration_ms = (time.perf_counter() - t0) * 1000.0

            self._breaker.record_success()

            # Compute decision agreement when champion_decision provided.
            agree_decision: bool | None = None
            shadow_decision: DecisionLiteral | None = None
            if champion_decision is not None:
                shadow_decision = (
                    "block" if shadow_score >= self._settings.decision_threshold else "allow"
                )
                agree_decision = shadow_decision == champion_decision

            _logger.info(
                "shadow.scored",
                request_id=request_id,
                shadow_score=shadow_score,
                shadow_decision=shadow_decision,
                shadow_model_version=local.content_hash,
                champion_score=champion_score,
                champion_decision=champion_decision,
                agree_decision=agree_decision,
                duration_ms=round(duration_ms, 3),
            )
        except (RuntimeError, ValueError, TypeError, OSError) as exc:
            # Shadow is best-effort. Don't crash the event loop.
            self._breaker.record_failure()
            _logger.warning(
                "shadow.failed",
                request_id=request_id,
                error_type=type(exc).__name__,
                detail=str(exc),
                breaker_state=self._breaker.state,
                consecutive_failures=self._breaker.consecutive_failures,
            )
        except asyncio.CancelledError:
            # Re-raise so cancellation propagates to the gather() in
            # `disconnect()` — otherwise we'd hide the cancellation
            # from the shutdown sequence.
            _logger.warning("shadow.cancelled", request_id=request_id)
            raise


__all__ = ["ShadowService"]
