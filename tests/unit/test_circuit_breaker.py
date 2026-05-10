"""Unit tests for the generic CircuitBreaker state machine.

Sprint 5 prompt 5.2.b. Pure-python tests with no asyncio dependency
— all transitions are exercised via the synchronous public API
(`can_proceed` / `record_success` / `record_failure` / `reset`). Time
is advanced deterministically via the `clock=` constructor injection.

Test scenarios (12 tests covering every transition + edge case):
    1. Initial state is CLOSED.
    2. `can_proceed()` returns True when CLOSED.
    3. Failures below threshold leave the breaker CLOSED.
    4. Reaching the threshold trips the breaker OPEN.
    5. `can_proceed()` returns False when OPEN within cooldown.
    6. After the cooldown elapses, `can_proceed()` transitions
       OPEN → HALF_OPEN and returns True.
    7. A success in HALF_OPEN closes the breaker AND resets the
       cooldown to its initial value.
    8. A failure in HALF_OPEN re-opens with the cooldown DOUBLED
       (the load-bearing exponential-backoff gate).
    9. Repeated half-open failures cap the cooldown at
       `max_cooldown_seconds`.
    10. `record_success()` from CLOSED resets the consecutive-failure
        counter.
    11. `reset()` clears all state to fresh-CLOSED.
    12. Concurrent `record_failure()` from 10 threads is thread-safe
        — counter ends at 1000, state is OPEN, no torn reads.
"""

from __future__ import annotations

import threading

import pytest

from fraud_engine.api.circuit_breaker import CircuitBreaker

# ---------------------------------------------------------------------
# Helpers — injectable monotonic clock for deterministic time.
# ---------------------------------------------------------------------


class _FakeClock:
    """Test-double monotonic clock that advances only when told to."""

    def __init__(self, start: float = 0.0) -> None:
        self._t = start

    def __call__(self) -> float:
        return self._t

    def advance(self, seconds: float) -> None:
        self._t += seconds


# ---------------------------------------------------------------------
# Scenario 1: initial state.
# ---------------------------------------------------------------------


def test_initial_state_closed() -> None:
    breaker = CircuitBreaker()
    assert breaker.state == "closed"
    assert breaker.consecutive_failures == 0


# ---------------------------------------------------------------------
# Scenario 2: can_proceed True when CLOSED.
# ---------------------------------------------------------------------


def test_can_proceed_when_closed() -> None:
    breaker = CircuitBreaker()
    assert breaker.can_proceed() is True
    # Idempotent — no transitions occur.
    assert breaker.state == "closed"
    assert breaker.can_proceed() is True


# ---------------------------------------------------------------------
# Scenario 3: failures below threshold leave breaker CLOSED.
# ---------------------------------------------------------------------


def test_record_failure_below_threshold_stays_closed() -> None:
    breaker = CircuitBreaker(failure_threshold=5)
    for _ in range(4):
        breaker.record_failure()
    assert breaker.state == "closed"
    assert breaker.consecutive_failures == 4
    assert breaker.can_proceed() is True


# ---------------------------------------------------------------------
# Scenario 4: threshold reached → OPEN.
# ---------------------------------------------------------------------


def test_threshold_failures_open_circuit() -> None:
    breaker = CircuitBreaker(failure_threshold=3)
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == "closed"
    breaker.record_failure()
    assert breaker.state == "open"
    assert breaker.consecutive_failures == 3


# ---------------------------------------------------------------------
# Scenario 5: can_proceed False when OPEN within cooldown.
# ---------------------------------------------------------------------


def test_can_proceed_false_when_open_within_cooldown() -> None:
    clock = _FakeClock()
    breaker = CircuitBreaker(failure_threshold=2, initial_cooldown_seconds=10.0, clock=clock)
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == "open"
    # Cooldown not yet elapsed.
    clock.advance(5.0)
    assert breaker.can_proceed() is False
    assert breaker.state == "open"


# ---------------------------------------------------------------------
# Scenario 6: cooldown elapsed → HALF_OPEN.
# ---------------------------------------------------------------------


def test_open_to_half_open_after_cooldown() -> None:
    clock = _FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, initial_cooldown_seconds=10.0, clock=clock)
    breaker.record_failure()  # open
    assert breaker.state == "open"
    clock.advance(10.0)  # exactly at boundary
    # can_proceed() side-effect: transitions to HALF_OPEN.
    assert breaker.can_proceed() is True
    assert breaker.state == "half_open"


# ---------------------------------------------------------------------
# Scenario 7: HALF_OPEN + success → CLOSED + cooldown reset.
# ---------------------------------------------------------------------


def test_half_open_success_closes_and_resets_cooldown() -> None:
    clock = _FakeClock()
    initial = 10.0
    breaker = CircuitBreaker(failure_threshold=1, initial_cooldown_seconds=initial, clock=clock)
    # Open, half-open via failure-then-cooldown-then-half-open-failure
    # to bump the cooldown above its initial value.
    breaker.record_failure()
    clock.advance(initial)
    breaker.can_proceed()  # half_open
    breaker.record_failure()  # back to OPEN; cooldown doubled to 20s
    assert breaker.current_cooldown_seconds == initial * 2
    # Now wait the doubled cooldown, half_open, succeed.
    clock.advance(initial * 2)
    breaker.can_proceed()  # half_open
    breaker.record_success()
    assert breaker.state == "closed"
    assert breaker.consecutive_failures == 0
    assert breaker.current_cooldown_seconds == initial  # reset to initial


# ---------------------------------------------------------------------
# Scenario 8: HALF_OPEN + failure → OPEN with doubled cooldown.
# ---------------------------------------------------------------------


def test_half_open_failure_reopens_with_doubled_cooldown() -> None:
    clock = _FakeClock()
    initial = 10.0
    factor = 2.0
    breaker = CircuitBreaker(
        failure_threshold=1,
        initial_cooldown_seconds=initial,
        backoff_factor=factor,
        clock=clock,
    )
    breaker.record_failure()  # OPEN; cooldown=10s
    clock.advance(initial)
    breaker.can_proceed()  # HALF_OPEN
    breaker.record_failure()  # back to OPEN; cooldown=20s
    assert breaker.state == "open"
    assert breaker.current_cooldown_seconds == initial * factor
    # Verify the next probe uses the doubled cooldown.
    clock.advance(initial)  # only half the new cooldown
    assert breaker.can_proceed() is False  # still OPEN
    clock.advance(initial)  # now at doubled cooldown
    assert breaker.can_proceed() is True  # HALF_OPEN


# ---------------------------------------------------------------------
# Scenario 9: max cooldown caps exponential backoff.
# ---------------------------------------------------------------------


def test_max_cooldown_caps_exponential_backoff() -> None:
    clock = _FakeClock()
    breaker = CircuitBreaker(
        failure_threshold=1,
        initial_cooldown_seconds=10.0,
        max_cooldown_seconds=50.0,  # cap at 50 (5× initial)
        backoff_factor=2.0,
        clock=clock,
    )
    breaker.record_failure()
    # Drive 10 → 20 → 40 → 50 (capped) → 50.
    expected_cooldowns = [20.0, 40.0, 50.0, 50.0]
    for expected in expected_cooldowns:
        clock.advance(breaker.current_cooldown_seconds)
        breaker.can_proceed()  # HALF_OPEN
        breaker.record_failure()  # back to OPEN; cooldown doubled or capped
        assert breaker.current_cooldown_seconds == expected


# ---------------------------------------------------------------------
# Scenario 10: success resets consecutive_failures from CLOSED.
# ---------------------------------------------------------------------


def test_record_success_resets_consecutive_failures() -> None:
    breaker = CircuitBreaker(failure_threshold=5)
    for _ in range(3):
        breaker.record_failure()
    assert breaker.consecutive_failures == 3
    breaker.record_success()
    assert breaker.consecutive_failures == 0
    assert breaker.state == "closed"
    # Now we need 5 fresh failures (not 2) to trip.
    for _ in range(4):
        breaker.record_failure()
    assert breaker.state == "closed"


# ---------------------------------------------------------------------
# Scenario 11: reset() clears state.
# ---------------------------------------------------------------------


def test_reset_clears_state_to_fresh_closed() -> None:
    clock = _FakeClock()
    initial = 10.0
    breaker = CircuitBreaker(failure_threshold=1, initial_cooldown_seconds=initial, clock=clock)
    breaker.record_failure()
    clock.advance(initial)
    breaker.can_proceed()
    breaker.record_failure()  # cooldown bumped to 20s
    assert breaker.state == "open"
    assert breaker.current_cooldown_seconds == initial * 2
    breaker.reset()
    assert breaker.state == "closed"
    assert breaker.consecutive_failures == 0
    assert breaker.current_cooldown_seconds == initial  # back to initial


# ---------------------------------------------------------------------
# Scenario 12: thread safety — 10 threads × 100 failures.
# ---------------------------------------------------------------------


def test_concurrent_record_failure_thread_safe() -> None:
    """10 threads × 100 record_failure() each → counter == 1000, state OPEN."""
    breaker = CircuitBreaker(failure_threshold=1000)
    n_threads = 10
    n_failures_per_thread = 100

    def worker() -> None:
        for _ in range(n_failures_per_thread):
            breaker.record_failure()

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    expected_total = n_threads * n_failures_per_thread
    assert breaker.consecutive_failures == expected_total
    assert breaker.state == "open"


# ---------------------------------------------------------------------
# Bonus: invalid constructor args are rejected.
# ---------------------------------------------------------------------


def test_invalid_constructor_args_raise() -> None:
    with pytest.raises(ValueError, match="failure_threshold"):
        CircuitBreaker(failure_threshold=0)
    with pytest.raises(ValueError, match="initial_cooldown_seconds"):
        CircuitBreaker(initial_cooldown_seconds=0)
    with pytest.raises(ValueError, match="max_cooldown_seconds"):
        CircuitBreaker(initial_cooldown_seconds=10.0, max_cooldown_seconds=5.0)
    with pytest.raises(ValueError, match="backoff_factor"):
        CircuitBreaker(backoff_factor=1.0)
