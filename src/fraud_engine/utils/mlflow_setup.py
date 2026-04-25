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

from fraud_engine.config.settings import Settings, get_settings
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


def setup_experiment(name: str | None = None, settings: Settings | None = None) -> str:
    """Set the MLflow tracking URI, ensure the experiment exists, and return its ID.

    Business rationale:
        Every training and evaluation run in this repo lands under a
        single experiment tree so the MLflow UI shows model selection
        as one coherent browse. Bundling "set tracking URI + create-or-
        get experiment + return ID" behind one call means scripts only
        need a single line of MLflow boilerplate before opening runs,
        which matches the 0.3.c spec contract.

    Trade-offs considered:
        - `mlflow.set_experiment(name)` would create-or-get internally,
          but it sets the *current* experiment as a side effect. We
          want to return the ID without committing the caller to it —
          a threshold-sweep notebook opens runs in a separate
          experiment while this one remains configured.
        - Name defaults to `settings.mlflow_experiment_name` so the
          one-env-var override flows through every script. Passing
          `name=None` is the explicit "use the default" idiom.
        - The `settings` parameter is dependency-injection-style: tests
          inject a `Settings` backed by a tmp tracking URI so the unit
          test never writes to the real `./mlruns`. Production callers
          leave it `None` and the function resolves `get_settings()`.
        - Setting the tracking URI here is idempotent with
          `configure_mlflow()`; MLflow silently overwrites on each
          call. Keeping both entry points avoids breaking the notebook
          caller that already follows the explicit
          `configure_mlflow(); setup_experiment(...)` pattern.

    Args:
        name: Experiment name. Defaults to
            `settings.mlflow_experiment_name`.
        settings: Optional Settings override (dependency-injection for
            tests). If None, resolved from `get_settings()`.

    Returns:
        The experiment_id as a string (MLflow's native ID type).
    """
    effective_settings = settings if settings is not None else get_settings()
    effective_name = name or effective_settings.mlflow_experiment_name

    # Idempotent global side effect — ensures the URI is set even if
    # the caller forgot to invoke `configure_mlflow()` first. Tests
    # rely on this so the Settings-injection override actually
    # reaches MLflow.
    mlflow.set_tracking_uri(effective_settings.mlflow_tracking_uri)

    existing = mlflow.get_experiment_by_name(effective_name)
    if existing is not None:
        experiment_id = existing.experiment_id
    else:
        experiment_id = mlflow.create_experiment(effective_name)

    get_logger(__name__).info(
        "mlflow.experiment_ready",
        name=effective_name,
        experiment_id=experiment_id,
        tracking_uri=effective_settings.mlflow_tracking_uri,
    )
    return str(experiment_id)


def log_dataframe_stats(df: pd.DataFrame, prefix: str) -> None:
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


def log_economic_metrics(  # noqa: PLR0913 — the four confusion-matrix counts and three USD costs are the business contract mirroring `metrics.economic_cost`; collapsing into a dict would hide the cost-model semantics at every call site.
    fn_count: int,
    fp_count: int,
    tp_count: int,
    tn_count: int,
    fraud_cost: float,
    fp_cost: float,
    tp_cost: float = 0.0,
) -> None:
    """Record confusion-matrix counts and derived cost totals into the active MLflow run.

    Logs six metrics:
        - `fn_count`, `fp_count`, `tp_count`, `tn_count` — raw
          confusion-matrix cells.
        - `total_cost_usd` — `fn * fraud_cost + fp * fp_cost + tp *
          tp_cost` (TN cost is zero by convention; see
          `fraud_engine.utils.metrics.economic_cost`).
        - `cost_per_txn` — `total_cost_usd / (fn + fp + tp + tn)` with
          a zero-guard on empty input.

    Business rationale:
        Sprint 4's threshold optimisation sweeps hundreds of candidate
        thresholds; each sweep run records these six numbers. The
        MLflow UI renders a cost curve across `total_cost_usd`
        directly — no ad-hoc plotting required. Logging the raw counts
        alongside the aggregate lets a reviewer diagnose a regression
        (is the spike driven by more FN or more FP?) without
        re-running inference.

    Trade-offs considered:
        - Signature mirrors `metrics.economic_cost` so a caller can
          pass through the outputs of that function by keyword. The
          two live side by side by design: `economic_cost` returns the
          number; this helper persists it. Using matching arg names
          prevents "rate vs count" confusion.
        - Raising when called outside a run is a loud failure mode.
          Silently starting a nested run would hide misconfiguration;
          logging to the root experiment would pollute the UI.
        - TN cost is not a parameter because it is zero by convention
          in fraud ML (no observable event). If a future sprint needs
          a non-zero TN cost (e.g. Sprint 6 monitoring overhead), add
          a `tn_cost: float = 0.0` parameter without breaking existing
          callers.

    Args:
        fn_count: Count of false negatives (missed fraud).
        fp_count: Count of false positives (blocked legit txn).
        tp_count: Count of true positives (caught fraud).
        tn_count: Count of true negatives.
        fraud_cost: USD cost of a single false negative.
        fp_cost: USD cost of a single false positive.
        tp_cost: USD cost of a single true positive (analyst review).
            Defaults to 0.0 per spec.

    Raises:
        RuntimeError: If no MLflow run is active.
    """
    if mlflow.active_run() is None:
        raise RuntimeError(
            "log_economic_metrics requires an active MLflow run. "
            "Wrap the call in `with mlflow.start_run(): ...`."
        )

    total_cost_usd = fn_count * fraud_cost + fp_count * fp_cost + tp_count * tp_cost
    # n_total is the full population; guard empty to avoid
    # ZeroDivisionError. An empty run is pathological (no rows were
    # scored) but the guard keeps the function total about semantics,
    # not crash avoidance.
    n_total = fn_count + fp_count + tp_count + tn_count
    cost_per_txn = total_cost_usd / n_total if n_total > 0 else 0.0

    mlflow.log_metric("fn_count", float(fn_count))
    mlflow.log_metric("fp_count", float(fp_count))
    mlflow.log_metric("tp_count", float(tp_count))
    mlflow.log_metric("tn_count", float(tn_count))
    mlflow.log_metric("total_cost_usd", float(total_cost_usd))
    mlflow.log_metric("cost_per_txn", float(cost_per_txn))


__all__ = [
    "configure_mlflow",
    "log_dataframe_stats",
    "log_economic_metrics",
    "setup_experiment",
]
