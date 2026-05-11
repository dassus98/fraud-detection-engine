"""Observability, drift detection, and alerting.

Sprint 6.1.a: re-exports the Prometheus metric constants from
`prometheus_metrics`.  Importing them here lets callers write
`from fraud_engine.monitoring import PREDICTIONS_TOTAL` instead of the
fully-qualified `from fraud_engine.monitoring.prometheus_metrics import …`,
without forcing a second module import (we re-export references; we don't
re-import the module).

Sprint 6.1.b: adds `DriftMonitor` + `DriftBaselineBuilder` for offline
PSI-based drift detection on production features against a training
baseline.

Sprint 6.1.c: adds `PerformanceMonitor` for offline rolling AUC / AUC-PR /
economic-cost monitoring against a training baseline, with alerting on
degradations >5%.
"""

from __future__ import annotations

from fraud_engine.monitoring.drift import DriftBaselineBuilder, DriftMonitor
from fraud_engine.monitoring.performance_monitor import PerformanceMonitor
from fraud_engine.monitoring.prometheus_metrics import (
    DECISION_LABELS,
    DEGRADED_MODE_TOTAL,
    DEPENDENCY_LABELS,
    DEPENDENCY_UP,
    DRIFT_ALERTS_TOTAL,
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
    SHADOW_DISAGREEMENT_TOTAL,
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
    "DRIFT_ALERTS_TOTAL",
    "DriftBaselineBuilder",
    "DriftMonitor",
    "PerformanceMonitor",
    "SHADOW_DISAGREEMENT_TOTAL",
    "set_shadow_breaker_state",
]
