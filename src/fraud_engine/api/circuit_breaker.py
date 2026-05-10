"""Generic three-state circuit breaker with exponential backoff.

Sprint 5 prompt 5.2.b: the failure-isolation primitive that wraps the
ShadowService's challenger-model calls. The breaker tracks consecutive
failures, trips OPEN after a configurable threshold, refuses calls
during a cooldown, then probes HALF-OPEN; a successful probe closes
the breaker, a failed probe re-opens it with the cooldown doubled
(exponential backoff, capped at `max_cooldown_seconds`).

Module surface (re-exported from `fraud_engine.api`):
    - CircuitBreaker
    - CircuitBreakerStateLiteral

Business rationale:
    Production fraud APIs run on top of a tower of failure-prone
    dependencies (Redis, Postgres, ML models). A single backend
    blowing up shouldn't cascade into wedged worker threads + queued
    timeouts that take down the whole API. The circuit-breaker
    pattern is the textbook isolation mechanism: detect the failure
    burst, fail fast for the cooldown window, then probe carefully
    when the dependency might have recovered.

    Sprint 5.2.b's specific use case is `ShadowService` — the
    challenger model's failures must not bleed into main-path
    latency or log volume. The breaker means "after the FraudNet
    process crashes 5 times in a row, stop trying for 30 s; then
    probe; if the next call works, resume; otherwise wait 60 s; etc."

Trade-offs considered:
    - **Three-state (closed / open / half_open) NOT two-state**
      (closed / open). The half-open probe is the textbook way to
      detect recovery without committing to full-traffic resumption
      on a single success — but the failure modes that justify a
      breaker are typically transient (network blip, OOM kill,
      restart). Without half-open, the breaker either flaps
      (re-opens on every probe in a sustained outage) or stays
      closed too long (cooldown is the only signal). Half-open lets
      a single success bring the breaker fully closed; a single
      failure resets the cooldown with exponential growth.

    - **Threshold = N consecutive failures, NOT N failures in a
      window**. Spec says "Threshold failures → open"; the simplest
      reading is "N consecutive failures". A sliding window (`>50%
      failure rate over last 100 calls`) would require maintaining a
      ring buffer and is harder to reason about under low-traffic
      conditions (5 failures in 30 minutes when there are 6 total
      calls in 30 minutes is much worse than 5 failures in 30
      seconds among 1000 calls — but the consecutive-failures rule
      treats them equivalently, which matches the spec's intent).
      A `record_success` resets the counter — so isolated failures
      don't accumulate.

    - **`threading.Lock` over `asyncio.Lock`**. The breaker is used
      from async code (ShadowService) but its public methods are
      all synchronous (sub-microsecond state mutations). A
      `threading.Lock` works correctly across both sync and async
      callers and avoids the asyncio-cascade where every method
      becomes async because one thing inside is async. The lock
      guards ~10 ns of pure-python state mutation; lock acquisition
      cost (~50 ns under no contention) is negligible vs the
      operations the breaker is meant to wrap (Postgres writes,
      model inference — milliseconds).

    - **`clock = time.monotonic` as a constructor injection point**.
      Tests need to advance time deterministically (the cooldown
      transitions are time-based). Production callers should NEVER
      pass `clock=`; the default is correct. Documented as a
      test-only injection point in the constructor docstring.

    - **`record_success` ALWAYS clears the failure count + closes
      the breaker**, even when called from the closed state. This
      lets a long-running stable workload reset the failure counter
      between transient failures — a single failure followed by 100
      successes shouldn't leave the breaker "1 failure away from
      tripping". Mirrors the standard Hystrix / resilience4j
      semantic.

    - **`record_failure` from the closed state does NOT trip until
      the threshold is reached**; from the half_open state it
      re-opens immediately AND doubles the cooldown. The asymmetry
      is intentional: the closed state has tolerance for transient
      blips; the half_open state is a probe — its job is to detect
      a still-broken dependency, so a single failure is decisive.

    - **Exponential backoff up to `max_cooldown_seconds`**. After
      the breaker has tripped repeatedly (e.g., a sustained outage),
      the cooldown grows: 30 s → 60 s → 120 s → 240 s → capped at
      300 s by default. This protects the failing dependency from
      probe-storm during a major outage and matches the "circuit
      breaker stays open longer the more it has tripped" semantic
      seen in production resilience libraries.

    - **No "best-effort" half-open semantics**: under concurrent
      use, multiple callers can read `state == "half_open"` and
      both attempt the probe. This is acceptable for the
      ShadowService use case (probes are fire-and-forget; a few
      duplicate probes during the half-open window cost only the
      duplicate work, not correctness). A strict latch + reset
      would add machinery for no business benefit at this project's
      RPS.

Cross-references:
    - `src/fraud_engine/api/shadow.py` (Sprint 5.2.b consumer) —
      records breaker outcomes from the fire-and-forget shadow path.
    - Resilience4j / Hystrix references for the textbook semantics.
    - `CLAUDE.md` §5.4 (no hardcoded values), §5.5 (logging).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Final, Literal

from fraud_engine.utils.logging import get_logger

# ---------------------------------------------------------------------
# Module constants.
# ---------------------------------------------------------------------

# Default thresholds. Tuned for the Sprint 5.2.b ShadowService:
# - 5 consecutive failures = ~25 ms of failed predict_proba calls
#   before tripping. Gives one or two genuine outages of headroom
#   before declaring the challenger broken.
# - 30 s initial cooldown = covers the typical model-process-restart
#   window without burning predict cycles.
# - 5 min (300 s) max cooldown = the longest interval before the
#   breaker probes during a sustained outage. Anything longer means
#   "the dependency is permanently dead; surface a louder alert".
# - 2× backoff factor = the standard exponential-backoff multiplier.
_DEFAULT_FAILURE_THRESHOLD: Final[int] = 5
_DEFAULT_INITIAL_COOLDOWN_SECONDS: Final[float] = 30.0
_DEFAULT_MAX_COOLDOWN_SECONDS: Final[float] = 300.0
_DEFAULT_BACKOFF_FACTOR: Final[float] = 2.0

# Public state literal — re-exported for downstream type hints.
CircuitBreakerStateLiteral = Literal["closed", "open", "half_open"]

# Internal state values — match the Literal alias values verbatim so
# `state` returns the Literal-typed constant directly.
_STATE_CLOSED: Final[CircuitBreakerStateLiteral] = "closed"
_STATE_OPEN: Final[CircuitBreakerStateLiteral] = "open"
_STATE_HALF_OPEN: Final[CircuitBreakerStateLiteral] = "half_open"

_logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Public class.
# ---------------------------------------------------------------------


class CircuitBreaker:
    """Three-state circuit breaker with exponential-backoff cooldown.

    Public API:
        - `can_proceed()` — read state; may transition OPEN → HALF_OPEN
          if the cooldown has elapsed. Returns True iff the caller
          should attempt the wrapped operation.
        - `record_success()` — caller observed a success; resets state
          to CLOSED and clears the cooldown back to the initial value.
        - `record_failure()` — caller observed a failure; increments
          the failure counter and, if at threshold (or in HALF_OPEN),
          trips the breaker open.
        - `reset()` — manual override for tests + ops; clears state to
          CLOSED with the initial cooldown.

    Read-only properties:
        - `state` — current `CircuitBreakerStateLiteral`.
        - `consecutive_failures` — current failure counter.
        - `current_cooldown_seconds` — current cooldown duration
          (grows via exponential backoff after each HALF_OPEN failure).

    Concurrency:
        Thread-safe via `threading.Lock`. Used safely from both sync
        and async code. Lock is held for ~10 ns of pure-python state
        mutation per method call; no callable runs under the lock.

    Business rationale + trade-offs considered: see module docstring.
    """

    def __init__(
        self,
        failure_threshold: int = _DEFAULT_FAILURE_THRESHOLD,
        initial_cooldown_seconds: float = _DEFAULT_INITIAL_COOLDOWN_SECONDS,
        max_cooldown_seconds: float = _DEFAULT_MAX_COOLDOWN_SECONDS,
        backoff_factor: float = _DEFAULT_BACKOFF_FACTOR,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Configure the breaker.

        Args:
            failure_threshold: Consecutive failures from CLOSED that
                trip the breaker OPEN. Must be >= 1.
            initial_cooldown_seconds: Initial OPEN-state duration
                before the breaker transitions to HALF_OPEN. Must be > 0.
            max_cooldown_seconds: Cap on the exponentially-grown
                cooldown. Must be >= initial_cooldown_seconds.
            backoff_factor: Multiplier applied to the current cooldown
                each time a HALF_OPEN probe fails. Must be > 1.0.
            clock: Monotonic-time function. Defaults to
                `time.monotonic`. **Test-only injection point** —
                production callers must not pass this.

        Raises:
            ValueError: If any argument is out of its valid range.
        """
        if failure_threshold < 1:
            raise ValueError(
                f"CircuitBreaker: failure_threshold must be >= 1, got {failure_threshold}"
            )
        if initial_cooldown_seconds <= 0:
            raise ValueError(
                f"CircuitBreaker: initial_cooldown_seconds must be > 0, "
                f"got {initial_cooldown_seconds}"
            )
        if max_cooldown_seconds < initial_cooldown_seconds:
            raise ValueError(
                f"CircuitBreaker: max_cooldown_seconds ({max_cooldown_seconds}) "
                f"must be >= initial_cooldown_seconds ({initial_cooldown_seconds})"
            )
        if backoff_factor <= 1.0:
            raise ValueError(
                f"CircuitBreaker: backoff_factor must be > 1.0 for "
                f"exponential growth, got {backoff_factor}"
            )

        self._failure_threshold: int = failure_threshold
        self._initial_cooldown_seconds: float = initial_cooldown_seconds
        self._max_cooldown_seconds: float = max_cooldown_seconds
        self._backoff_factor: float = backoff_factor
        self._clock: Callable[[], float] = clock

        # State — guarded by self._lock.
        self._state: CircuitBreakerStateLiteral = _STATE_CLOSED
        self._consecutive_failures: int = 0
        self._current_cooldown_seconds: float = initial_cooldown_seconds
        self._opened_at: float | None = None

        self._lock: threading.Lock = threading.Lock()

    # ---------- read-only accessors ------------------------------------

    @property
    def state(self) -> CircuitBreakerStateLiteral:
        """Current state. Pure read; no transition."""
        with self._lock:
            return self._state

    @property
    def consecutive_failures(self) -> int:
        """Current failure counter (since last success)."""
        with self._lock:
            return self._consecutive_failures

    @property
    def current_cooldown_seconds(self) -> float:
        """Current cooldown duration (grows via exponential backoff)."""
        with self._lock:
            return self._current_cooldown_seconds

    # ---------- state transitions --------------------------------------

    def can_proceed(self) -> bool:
        """Should the caller attempt the wrapped operation?

        Returns True if state is CLOSED or HALF_OPEN, OR if state is
        OPEN but the cooldown has elapsed (in which case this call
        transitions OPEN → HALF_OPEN as a side effect — the next
        caller is the probe).

        Returns False only when state is OPEN and cooldown has not
        yet elapsed.
        """
        with self._lock:
            if self._state == _STATE_CLOSED:
                return True
            if self._state == _STATE_HALF_OPEN:
                # Already probing — let the caller through.
                return True
            # OPEN: check whether the cooldown has elapsed.
            if self._opened_at is None:
                # Defensive: shouldn't happen (always set on transition
                # to OPEN), but if it does, treat as "cooldown elapsed".
                self._transition_to_half_open_locked()
                return True
            elapsed = self._clock() - self._opened_at
            if elapsed >= self._current_cooldown_seconds:
                self._transition_to_half_open_locked()
                return True
            return False

    def record_success(self) -> None:
        """Caller observed a successful operation.

        Closes the breaker and resets the cooldown back to its
        initial value. Clears the consecutive-failure counter. Idempotent.
        """
        with self._lock:
            previous_state = self._state
            self._state = _STATE_CLOSED
            self._consecutive_failures = 0
            self._current_cooldown_seconds = self._initial_cooldown_seconds
            self._opened_at = None
            if previous_state != _STATE_CLOSED:
                _logger.info(
                    "circuit_breaker.closed_after_success",
                    previous_state=previous_state,
                )

    def record_failure(self) -> None:
        """Caller observed a failed operation.

        From CLOSED: increments the failure counter; trips OPEN if
        the counter reaches `failure_threshold`.

        From HALF_OPEN: trips back to OPEN immediately AND doubles
        the cooldown (capped at `max_cooldown_seconds`) — exponential
        backoff for repeated probe failures.

        From OPEN: no-op (the call shouldn't have happened —
        `can_proceed` returned False — but we tolerate the race
        gracefully).
        """
        with self._lock:
            self._consecutive_failures += 1
            if self._state == _STATE_HALF_OPEN:
                # Probe failed: reopen with doubled cooldown.
                self._current_cooldown_seconds = min(
                    self._current_cooldown_seconds * self._backoff_factor,
                    self._max_cooldown_seconds,
                )
                self._state = _STATE_OPEN
                self._opened_at = self._clock()
                _logger.warning(
                    "circuit_breaker.reopened_after_probe_failure",
                    cooldown_seconds=self._current_cooldown_seconds,
                    consecutive_failures=self._consecutive_failures,
                )
                return
            if (
                self._state == _STATE_CLOSED
                and self._consecutive_failures >= self._failure_threshold
            ):
                self._state = _STATE_OPEN
                self._opened_at = self._clock()
                _logger.warning(
                    "circuit_breaker.opened_after_threshold",
                    threshold=self._failure_threshold,
                    cooldown_seconds=self._current_cooldown_seconds,
                )

    def reset(self) -> None:
        """Manual override: force CLOSED state + reset all counters.

        Intended for tests + ops (e.g., a clear-the-breaker admin
        endpoint after fixing a downstream issue). Production callers
        should rely on `record_success` to close the breaker via the
        normal flow.
        """
        with self._lock:
            self._state = _STATE_CLOSED
            self._consecutive_failures = 0
            self._current_cooldown_seconds = self._initial_cooldown_seconds
            self._opened_at = None

    # ---------- internal helpers (require lock held) -------------------

    def _transition_to_half_open_locked(self) -> None:
        """Transition OPEN → HALF_OPEN. Caller must hold `self._lock`."""
        self._state = _STATE_HALF_OPEN
        # Don't clear `_opened_at` — preserve for diagnostic logging.
        _logger.info(
            "circuit_breaker.half_open_probe",
            cooldown_was_seconds=self._current_cooldown_seconds,
        )


__all__ = ["CircuitBreaker", "CircuitBreakerStateLiteral"]
