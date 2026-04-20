"""MLflow tracking wiring, experiment bootstrap, and logging helpers.

Every model-training sprint (3, 4) calls `configure_mlflow()` +
`setup_experiment()` once at entry; all subsequent `mlflow.start_run`
calls then land in the right experiment on the right backend without
per-caller configuration.

Business rationale:
    Fraud-model hyperparameter sweeps and threshold-optimisation runs
    can generate hundreds of MLflow runs per sprint. Without a shared
    setup module, each notebook and script sets the tracking URI and
    experiment name a little differently, and the cross-referencing
    needed for model selection rots. Centralising here means every
    caller sees the same `experiment_id` and the UI shows a coherent
    model family.

Trade-offs considered:
    - `configure_mlflow` is a thin wrapper around
      `mlflow.set_tracking_uri`, not a Pydantic-style config object.
      MLflow's own configuration is globally mutable; wrapping it
      inside a richer class would pretend otherwise.
    - `log_dataframe_stats` splits "shape + structure" into MLflow
      *params* and "observed values" into *metrics*. Params are
      write-once strings; metrics are typed floats that can be plotted
      and overwritten. Getting this boundary right once spares Sprint
      4 from re-discovering it mid-threshold-sweep.
    - `log_economic_metrics` raises if called outside a run. An
      alternative (silently starting a nested run) would make
      misconfigured code appear to succeed, which is worse.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import mlflow

from fraud_engine.config.settings import get_settings
from fraud_engine.utils.logging import get_logger

if TYPE_CHECKING:
    import pandas as pd


def configure_mlflow() -> None:
    """Point MLflow at the configured tracking URI.

    Business rationale:
        `mlflow.set_tracking_uri` is a process-global side effect.
        Hiding that behind a named function means callers read the
        intent rather than the mechanic, and we can swap in extra
        wiring (auth, S3 artifact root) without a codebase-wide edit.

    Trade-offs considered:
        - Idempotent by construction — MLflow silently overwrites the
          tracking URI on each call. Re-invoking is harmless.
        - We do not create the experiment here; that is
          `setup_experiment`'s job. Splitting them lets callers swap
          experiments mid-script (Sprint 3 evaluation vs Sprint 4
          threshold sweep) without re-resolving the URI.
    """
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    get_logger(__name__).info(
        "mlflow.tracking_uri_set",
        tracking_uri=settings.mlflow_tracking_uri,
    )


def setup_experiment(name: str | None = None) -> str:
    """Ensure an MLflow experiment exists and return its ID.

    Business rationale:
        Every training and evaluation run in this repo lands under a
        single experiment tree so the MLflow UI shows model selection
        as one coherent browse. Creating on first call + returning the
        cached ID on subsequent calls hides the lookup-or-create
        ceremony from callers.

    Trade-offs considered:
        - `mlflow.set_experiment(name)` would create-or-get internally,
          but it sets the *current* experiment as a side effect. We
          want to return the ID without committing the caller to it —
          a threshold-sweep notebook opens runs in a separate
          experiment while this one remains configured.
        - Name defaults to `settings.mlflow_experiment_name` so the
          one-env-var override flows through every script.

    Args:
        name: Experiment name. Defaults to
            `settings.mlflow_experiment_name`.

    Returns:
        The experiment_id as a string (MLflow's native ID type).
    """
    settings = get_settings()
    effective_name = name or settings.mlflow_experiment_name

    existing = mlflow.get_experiment_by_name(effective_name)
    if existing is not None:
        experiment_id = existing.experiment_id
    else:
        experiment_id = mlflow.create_experiment(effective_name)

    get_logger(__name__).info(
        "mlflow.experiment_ready",
        name=effective_name,
        experiment_id=experiment_id,
    )
    return str(experiment_id)


def log_dataframe_stats(df: pd.DataFrame, *, prefix: str) -> None:
    """Log a DataFrame's shape/memory/dtypes into the active MLflow run.

    Records:
        Params (write-once strings):
            - `{prefix}_rows`
            - `{prefix}_cols`
            - `{prefix}_memory_mb`
            - `{prefix}_dtypes` (JSON-encoded dtype histogram)
        Metrics (numeric, overwritable):
            - `{prefix}_n_missing`
            - `{prefix}_n_duplicates`

    Business rationale:
        Sprint 3's model runs want a fingerprint of the training data
        attached to every run, not just the hyperparameters. MLflow's
        params-vs-metrics split is a schema choice: shape/structure is
        a write-once fact (param); observed counts can drift
        run-to-run (metric). Getting this contract right now prevents
        threshold sweeps and training runs from disagreeing on where
        to find the row count.

    Trade-offs considered:
        - Params are limited to 500 chars in MLflow ≤ 2.x; the dtype
          histogram is JSON-encoded to stay compact. A 1000-column
          DataFrame still dumps to ~100 chars because of the
          dtype-to-count reduction.
        - Duplicate count uses `df.duplicated().sum()`, which is O(N)
          in column count. Accept the cost — this is called at stage
          boundaries, not in hot loops.

    Args:
        df: The DataFrame to fingerprint.
        prefix: Short label prepended to every key (e.g. "train",
            "val", "raw"). Keeps multi-DataFrame runs unambiguous.
    """
    dtype_counts = {str(dt): int(n) for dt, n in df.dtypes.value_counts().items()}
    memory_mb = float(df.memory_usage(deep=True).sum()) / (1024 * 1024)

    mlflow.log_param(f"{prefix}_rows", int(df.shape[0]))
    mlflow.log_param(f"{prefix}_cols", int(df.shape[1]))
    mlflow.log_param(f"{prefix}_memory_mb", round(memory_mb, 4))
    mlflow.log_param(f"{prefix}_dtypes", json.dumps(dtype_counts))

    mlflow.log_metric(f"{prefix}_n_missing", int(df.isna().sum().sum()))
    mlflow.log_metric(f"{prefix}_n_duplicates", int(df.duplicated().sum()))


def log_economic_metrics(
    fn_rate: float,
    fp_rate: float,
    total_cost_usd: float,
) -> None:
    """Record cost-function outputs into the active MLflow run.

    Logs three metrics: `fn_rate`, `fp_rate`, `total_cost_usd`. Must
    be called inside an active `mlflow.start_run()` block.

    Business rationale:
        Sprint 4's threshold optimisation sweeps hundreds of candidate
        thresholds; each sweep run records these three numbers. The
        MLflow UI then renders a cost curve across `total_cost_usd`
        directly — no ad-hoc plotting required. Using the same key
        names in evaluation (Sprint 4) and monitoring (Sprint 6) means
        a dashboard can trend them over time without a translation
        layer.

    Trade-offs considered:
        - Raising when called outside a run is a loud failure mode.
          Silently starting a nested run would hide misconfiguration;
          logging to the root experiment would pollute the UI.

    Args:
        fn_rate: False-negative rate (missed frauds / total frauds).
        fp_rate: False-positive rate (false alerts / total legit).
        total_cost_usd: Expected cost under the current thresholding,
            computed via `fraud_engine.utils.metrics.economic_cost`.

    Raises:
        RuntimeError: If no MLflow run is active.
    """
    if mlflow.active_run() is None:
        raise RuntimeError(
            "log_economic_metrics requires an active MLflow run. "
            "Wrap the call in `with mlflow.start_run(): ...`."
        )
    mlflow.log_metric("fn_rate", float(fn_rate))
    mlflow.log_metric("fp_rate", float(fp_rate))
    mlflow.log_metric("total_cost_usd", float(total_cost_usd))


__all__ = [
    "configure_mlflow",
    "log_dataframe_stats",
    "log_economic_metrics",
    "setup_experiment",
]
