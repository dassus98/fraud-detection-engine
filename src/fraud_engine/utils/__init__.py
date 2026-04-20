"""Cross-cutting utilities: logging, tracing, metrics, MLflow wiring, seeding."""

from __future__ import annotations

from fraud_engine.utils.logging import (
    bind_request_id,
    configure_logging,
    get_logger,
    get_request_id,
    log_call,
    log_dataframe,
    new_run_id,
    reset_request_id,
)
from fraud_engine.utils.metrics import (
    compute_psi,
    economic_cost,
    precision_recall_at_k,
    recall_at_fpr,
)
from fraud_engine.utils.mlflow_setup import (
    configure_mlflow,
    log_dataframe_stats,
    log_economic_metrics,
    setup_experiment,
)
from fraud_engine.utils.seeding import set_all_seeds
from fraud_engine.utils.tracing import Run, attach_artifact, run_context

__all__ = [
    "Run",
    "attach_artifact",
    "bind_request_id",
    "compute_psi",
    "configure_logging",
    "configure_mlflow",
    "economic_cost",
    "get_logger",
    "get_request_id",
    "log_call",
    "log_dataframe",
    "log_dataframe_stats",
    "log_economic_metrics",
    "new_run_id",
    "precision_recall_at_k",
    "recall_at_fpr",
    "reset_request_id",
    "run_context",
    "set_all_seeds",
    "setup_experiment",
]
