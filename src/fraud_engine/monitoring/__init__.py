"""Observability, drift detection, and alerting.

Sprint 6.1.a: re-exports the Prometheus metric constants from
`prometheus_metrics`.  Importing them here lets callers write
`from fraud_engine.monitoring import PREDICTIONS_TOTAL` instead of the
fully-qualified `from fraud_engine.monitoring.prometheus_metrics import …`,
without forcing a second module import (we re-export references; we don't
re-import the module).
"""

from __future__ import annotations

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

__all__ = [
    "DECISION_LABELS",
    "DEGRADED_MODE_TOTAL",
    "DEPENDENCY_LABELS",
    "DEPENDENCY_UP",
    "FEATURE_FETCH_SECONDS",
    "INFERENCE_SECONDS",
    "LATENCY_BUCKETS",
    "MODEL_INFO",
    "PREDICTIONS_TOTAL",
    "PREDICTION_SCORE",
    "PREDICT_TOTAL_SECONDS",
    "SCORE_BUCKETS",
    "SHADOW_BREAKER_STATE",
    "SHADOW_BREAKER_STATE_LABELS",
    "SHADOW_EVENT_LABELS",
    "SHADOW_TOTAL",
    "SHAP_SECONDS",
    "set_shadow_breaker_state",
]
