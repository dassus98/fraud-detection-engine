"""Unit tests for the Sprint 6.1.a Prometheus metrics module.

Twelve tests across four categories:

    Registration (3) — every metric exists in REGISTRY at import time;
        types match spec; histogram buckets match the documented tuples.

    Behaviour (5) — increment / observe / gauge-set affect the underlying
        sample values exactly as expected, including the
        `set_shadow_breaker_state` one-hot helper.

    Label discipline (2) — no metric carries a forbidden high-cardinality
        label; every labelled metric's documented value set is a finite
        tuple of strings.

    Scrape format (2) — `prometheus_client.generate_latest()` returns
        valid Prometheus text format with all metric names + HELP / TYPE
        comments.

Tests use the **delta pattern**: capture sample value before, perform the
increment, capture after, assert the difference.  The metrics are global
singletons (registered against `prometheus_client.REGISTRY` at import
time) and tests must NOT use `importlib.reload` on the metrics module
(would raise `Duplicated timeseries`).  The delta pattern means tests are
order-independent without needing a custom fixture to reset state.
"""

from __future__ import annotations

from typing import Any

import pytest
from prometheus_client import REGISTRY as _GLOBAL_REGISTRY, Counter, Gauge, Histogram
from prometheus_client.exposition import generate_latest

from fraud_engine.monitoring.prometheus_metrics import (
    DECISION_LABELS,
    DEGRADED_MODE_TOTAL,
    DEPENDENCY_LABELS,
    DEPENDENCY_UP,
    FEATURE_FETCH_SECONDS,
    INFERENCE_SECONDS,
    LATENCY_BUCKETS,
    MODEL_INFO,
    PREDICT_TOTAL_SECONDS,
    PREDICTION_SCORE,
    PREDICTIONS_TOTAL,
    SCORE_BUCKETS,
    SHADOW_BREAKER_STATE,
    SHADOW_BREAKER_STATE_LABELS,
    SHADOW_EVENT_LABELS,
    SHADOW_TOTAL,
    SHAP_SECONDS,
    set_shadow_breaker_state,
)

# ---------------------------------------------------------------------
# Test constants — what the spec requires.
# ---------------------------------------------------------------------

_EXPECTED_METRIC_NAMES: tuple[str, ...] = (
    # 4 latency histograms (existing — moved from main.py).
    "fraud_engine_feature_fetch_seconds",
    "fraud_engine_inference_seconds",
    "fraud_engine_shap_seconds",
    "fraud_engine_predict_total_seconds",
    # New histogram (drift seed).
    "fraud_engine_prediction_score",
    # New counters.
    "fraud_engine_predictions_total",
    "fraud_engine_degraded_mode_total",
    "fraud_engine_shadow_total",
    # New gauges.
    "fraud_engine_dependency_up",
    "fraud_engine_shadow_breaker_state",
    "fraud_engine_model_info",
)

# A metric labelled with one of these would explode the time-series
# storage on a busy production scrape.  No metric is allowed to declare
# any of these as a label name (see Decision 4 in the module docstring).
_FORBIDDEN_LABELS: frozenset[str] = frozenset(
    {
        "request_id",
        "txn_id",
        "card1",
        "addr1",
        "entity_id",
        "user_id",
        "device_id",
    }
)


# ---------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------


def _sample_value(name: str, labels: dict[str, str] | None = None) -> float:
    """Return the current absolute value of a Prometheus sample, or 0.0.

    `prometheus_client.REGISTRY.get_sample_value(name, labels)` returns
    None when the labelled series has not been observed yet; we coerce
    None → 0.0 so callers can write `after - before` without `None`
    arithmetic.
    """
    value = _GLOBAL_REGISTRY.get_sample_value(name, labels or {})
    return float(value) if value is not None else 0.0


def _registered_metric_names() -> set[str]:
    """All Counter / Histogram / Gauge metric names known to REGISTRY."""
    names: set[str] = set()
    for collector in list(_GLOBAL_REGISTRY._collector_to_names.values()):  # type: ignore[attr-defined]
        names.update(collector)
    return names


# ---------------------------------------------------------------------
# Registration tests (3).
# ---------------------------------------------------------------------


class TestRegistration:
    """Every metric the API depends on exists in REGISTRY at import time."""

    def test_all_metrics_exist_in_registry(self) -> None:
        """Every name in `_EXPECTED_METRIC_NAMES` is registered."""
        registered = _registered_metric_names()
        for name in _EXPECTED_METRIC_NAMES:
            assert name in registered, (
                f"Expected metric '{name}' missing from REGISTRY. "
                f"This typically means the prometheus_metrics module "
                f"failed to import or a metric was renamed."
            )

    def test_metric_types_match_spec(self) -> None:
        """Each constant binds to the type the spec calls for."""
        # Counters — monotonic volumes.
        assert isinstance(PREDICTIONS_TOTAL, Counter)
        assert isinstance(DEGRADED_MODE_TOTAL, Counter)
        assert isinstance(SHADOW_TOTAL, Counter)
        # Histograms — distributions.
        assert isinstance(FEATURE_FETCH_SECONDS, Histogram)
        assert isinstance(INFERENCE_SECONDS, Histogram)
        assert isinstance(SHAP_SECONDS, Histogram)
        assert isinstance(PREDICT_TOTAL_SECONDS, Histogram)
        assert isinstance(PREDICTION_SCORE, Histogram)
        # Gauges — instantaneous state.
        assert isinstance(DEPENDENCY_UP, Gauge)
        assert isinstance(SHADOW_BREAKER_STATE, Gauge)
        assert isinstance(MODEL_INFO, Gauge)

    def test_histogram_buckets_match_spec(self) -> None:
        """Latency histograms reuse `LATENCY_BUCKETS`; score uses `SCORE_BUCKETS`.

        The bucket choice for latency is a load-bearing alerting
        contract: the [0.050, 0.100] bucket boundary is exactly the
        CLAUDE.md §3 P95 budget so `histogram_quantile(0.95, ...) > 0.100`
        is a clean alert query.  A drift here would silently break that.

        prometheus-client appends a `+Inf` bucket internally; we check
        that the user-facing buckets (everything before +Inf) match.
        """
        for hist in (
            FEATURE_FETCH_SECONDS,
            INFERENCE_SECONDS,
            SHAP_SECONDS,
            PREDICT_TOTAL_SECONDS,
        ):
            buckets: tuple[float, ...] = tuple(
                b
                for b in hist._upper_bounds
                if b != float("inf")  # type: ignore[attr-defined]
            )
            assert buckets == LATENCY_BUCKETS, (
                f"{hist._name} buckets {buckets} != "  # type: ignore[attr-defined]
                f"LATENCY_BUCKETS {LATENCY_BUCKETS}"
            )

        score_buckets: tuple[float, ...] = tuple(
            b
            for b in PREDICTION_SCORE._upper_bounds
            if b != float("inf")  # type: ignore[attr-defined]
        )
        assert score_buckets == SCORE_BUCKETS, (
            f"PREDICTION_SCORE buckets {score_buckets} != " f"SCORE_BUCKETS {SCORE_BUCKETS}"
        )


# ---------------------------------------------------------------------
# Behaviour tests (5).
# ---------------------------------------------------------------------


class TestBehaviour:
    """Increment / observe / gauge-set move the underlying sample values."""

    def test_predictions_total_increments_by_decision_label(self) -> None:
        """`PREDICTIONS_TOTAL.labels(decision='block').inc()` adds 1 to that series."""
        before_block = _sample_value("fraud_engine_predictions_total", {"decision": "block"})
        before_allow = _sample_value("fraud_engine_predictions_total", {"decision": "allow"})

        PREDICTIONS_TOTAL.labels(decision="block").inc()
        PREDICTIONS_TOTAL.labels(decision="block").inc()
        PREDICTIONS_TOTAL.labels(decision="allow").inc()

        after_block = _sample_value("fraud_engine_predictions_total", {"decision": "block"})
        after_allow = _sample_value("fraud_engine_predictions_total", {"decision": "allow"})

        assert after_block - before_block == pytest.approx(2.0)
        assert after_allow - before_allow == pytest.approx(1.0)

    def test_degraded_mode_total_increments_unlabeled(self) -> None:
        """`DEGRADED_MODE_TOTAL.inc()` advances the single unlabelled series."""
        before = _sample_value("fraud_engine_degraded_mode_total")
        DEGRADED_MODE_TOTAL.inc()
        after = _sample_value("fraud_engine_degraded_mode_total")
        assert after - before == pytest.approx(1.0)

    def test_shadow_total_increments_by_event_label(self) -> None:
        """All three documented event labels accept increments independently."""
        before: dict[str, float] = {
            evt: _sample_value("fraud_engine_shadow_total", {"event": evt})
            for evt in SHADOW_EVENT_LABELS
        }
        for evt in SHADOW_EVENT_LABELS:
            SHADOW_TOTAL.labels(event=evt).inc()
        for evt in SHADOW_EVENT_LABELS:
            after = _sample_value("fraud_engine_shadow_total", {"event": evt})
            assert after - before[evt] == pytest.approx(
                1.0
            ), f"event={evt!r} did not increment by 1"

    def test_prediction_score_observes_into_correct_bucket(self) -> None:
        """An observation of value v lands in every bucket whose `le` >= v.

        Histogram bucket counts are cumulative — a value of 0.07 lands in
        the buckets `le=0.08`, `le=0.1`, `le=0.2`, ..., `le=+Inf`.  This
        encodes the "fraction of samples ≤ X" semantics that
        `histogram_quantile` reads from.
        """
        observed_value = 0.07  # falls between 0.05 and 0.08

        # Capture the cumulative count of every bucket boundary BEFORE.
        # `le="0.05"` is the highest bucket the 0.07 observation must
        # NOT increment; `le="0.08"` is the lowest it must.
        bucket_le_below = "0.05"
        bucket_le_at = "0.08"
        before_below = _sample_value(
            "fraud_engine_prediction_score_bucket", {"le": bucket_le_below}
        )
        before_at = _sample_value("fraud_engine_prediction_score_bucket", {"le": bucket_le_at})
        before_count = _sample_value("fraud_engine_prediction_score_count")
        before_sum = _sample_value("fraud_engine_prediction_score_sum")

        PREDICTION_SCORE.observe(observed_value)

        after_below = _sample_value("fraud_engine_prediction_score_bucket", {"le": bucket_le_below})
        after_at = _sample_value("fraud_engine_prediction_score_bucket", {"le": bucket_le_at})
        after_count = _sample_value("fraud_engine_prediction_score_count")
        after_sum = _sample_value("fraud_engine_prediction_score_sum")

        # 0.07 is NOT ≤ 0.05; that bucket must not advance.
        assert after_below - before_below == pytest.approx(0.0)
        # 0.07 IS ≤ 0.08; that bucket and everything above must advance.
        assert after_at - before_at == pytest.approx(1.0)
        # _count is the number of observations regardless of bucket.
        assert after_count - before_count == pytest.approx(1.0)
        # _sum tracks the sum of observed values; a single observe(0.07) adds 0.07.
        assert after_sum - before_sum == pytest.approx(observed_value)

    def test_set_shadow_breaker_state_helper_sets_one_zeros_others(self) -> None:
        """The helper enforces the one-hot invariant across all 3 states."""
        for active in SHADOW_BREAKER_STATE_LABELS:
            set_shadow_breaker_state(active)
            for state in SHADOW_BREAKER_STATE_LABELS:
                value = _sample_value("fraud_engine_shadow_breaker_state", {"state": state})
                expected = 1.0 if state == active else 0.0
                assert value == pytest.approx(expected), (
                    f"After set_shadow_breaker_state({active!r}), "
                    f"state={state!r} gauge = {value}, expected {expected}"
                )

        # Defensive: an unknown state silently zeros all known states.
        set_shadow_breaker_state("not_a_real_state")
        for state in SHADOW_BREAKER_STATE_LABELS:
            value = _sample_value("fraud_engine_shadow_breaker_state", {"state": state})
            assert value == pytest.approx(0.0), (
                f"After set_shadow_breaker_state(unknown), "
                f"state={state!r} gauge = {value}, expected 0"
            )


# ---------------------------------------------------------------------
# Label discipline tests (2).
# ---------------------------------------------------------------------


class TestLabelDiscipline:
    """Cardinality discipline — no high-cardinality labels."""

    def test_no_high_cardinality_labels(self) -> None:
        """No metric carries any of `_FORBIDDEN_LABELS` as a label name.

        Adding `request_id` as a label would create one time-series per
        unique request — at the project's target RPS, that's ~35K series
        per day per metric, which would OOM the Prometheus instance
        within hours.  A test catches the regression at PR-time before
        it ships.
        """
        all_metrics: tuple[Any, ...] = (
            PREDICTIONS_TOTAL,
            DEGRADED_MODE_TOTAL,
            SHADOW_TOTAL,
            DEPENDENCY_UP,
            SHADOW_BREAKER_STATE,
            MODEL_INFO,
            FEATURE_FETCH_SECONDS,
            INFERENCE_SECONDS,
            SHAP_SECONDS,
            PREDICT_TOTAL_SECONDS,
            PREDICTION_SCORE,
        )
        for metric in all_metrics:
            label_names: tuple[str, ...] = tuple(metric._labelnames)  # type: ignore[attr-defined]
            forbidden_present = set(label_names) & _FORBIDDEN_LABELS
            assert not forbidden_present, (
                f"Metric {metric._name!r} declares forbidden "  # type: ignore[attr-defined]
                f"high-cardinality label(s): {forbidden_present}"
            )

    def test_label_value_sets_are_finite(self) -> None:
        """Every documented label-value tuple is a finite tuple of strings."""
        for label_set in (
            DECISION_LABELS,
            SHADOW_EVENT_LABELS,
            DEPENDENCY_LABELS,
            SHADOW_BREAKER_STATE_LABELS,
        ):
            assert isinstance(label_set, tuple), (
                "Documented label sets must be immutable tuples so "
                "callers can't accidentally mutate them at runtime."
            )
            assert len(label_set) > 0, "Empty label set documented"
            assert len(label_set) < 20, (
                f"Label set has {len(label_set)} values — verify this "
                f"is a bounded enumeration, not free-form text."
            )
            for value in label_set:
                assert isinstance(value, str), f"Label value {value!r} is not a string"
                assert value, "Empty-string label value documented"


# ---------------------------------------------------------------------
# Scrape-format tests (2).
# ---------------------------------------------------------------------


class TestScrapeFormat:
    """`generate_latest()` produces valid Prometheus text format."""

    def test_generate_latest_returns_valid_prometheus_text(self) -> None:
        """Output is bytes, decodes as UTF-8, contains TYPE comments."""
        # First drive at least one observation so the histogram series
        # are present (without observations, prometheus_client may emit
        # only the bare HELP/TYPE lines).
        PREDICTION_SCORE.observe(0.5)
        PREDICTIONS_TOTAL.labels(decision="allow").inc()

        scrape: bytes = generate_latest(_GLOBAL_REGISTRY)
        assert isinstance(scrape, bytes), "generate_latest must return bytes"
        text = scrape.decode("utf-8")

        # Each metric carries a TYPE comment in the standard format.
        assert "# TYPE fraud_engine_predictions_total counter" in text
        assert "# TYPE fraud_engine_prediction_score histogram" in text
        assert "# TYPE fraud_engine_dependency_up gauge" in text

    def test_scrape_text_contains_all_metric_names_and_help(self) -> None:
        """Every expected metric appears in the scrape output with a HELP line."""
        scrape_text: str = generate_latest(_GLOBAL_REGISTRY).decode("utf-8")
        for name in _EXPECTED_METRIC_NAMES:
            # HELP comment format: "# HELP <name> <description>".
            assert (
                f"# HELP {name} " in scrape_text
            ), f"metric '{name}' is missing its HELP comment in scrape output"
