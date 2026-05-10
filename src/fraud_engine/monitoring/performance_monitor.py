"""Rolling AUC / AUC-PR / economic-cost monitoring against a training baseline.

Sprint 6 prompt 6.1.c: closes the loop opened by 6.1.a (output drift via the
`fraud_engine_prediction_score` Prometheus histogram) and 6.1.b (input drift
via PSI).  Both detect *some* shift; only labelled-prediction comparison
answers "is the model actually performing worse on the labels we care about?"
PSI on a feature distribution can shift for benign reasons (a marketing
campaign brings new geo mix); only AUC / AUC-PR / cost regression against
the training baseline confirms model degradation.

In production, the labelled-prediction stream is fed by chargeback feeds
with a 30–90 day lag.  For this project's offline-portfolio context the
labels come from the IEEE-CIS test set we already have on disk; the
PerformanceMonitor itself is data-source-agnostic — caller hands it a
DataFrame, mirroring `DriftMonitor`'s contract from 6.1.b.

Public surface (re-exported by `fraud_engine.monitoring`):

    - `PerformanceMonitor(settings)` — stateless analyzer; no per-call
      persistence, no Postgres queries.
        - `compute_rolling_metrics(recent_window) → dict[str, float]`
          returns `{auc, auc_pr, cost, n_recent}`.
        - `compare_to_baseline(metrics) → list[_Degradation]` returns
          one record per metric whose drop exceeds
          `Settings.performance_degradation_threshold`.
        - `check_and_alert(recent_window, *, run_id, alert_log_dir) → int`
          computes → compares → appends one JSONL line per degraded
          metric to `{alert_log_dir}/{run_id}/performance_alerts.jsonl`.
          Returns the count written; 0 means no degradation (no file
          created).

Business rationale:
    Three metrics, three failure modes:

    - **AUC (ROC):** discrimination quality.  Answers "do high scores
      correlate with fraud?".  Stable under class-balance shifts.
    - **AUC-PR (`average_precision_score`):** the right metric for
      class-imbalanced fraud (3.5% positive rate per CLAUDE.md §1).
      Catches degradations that ROC AUC misses on imbalanced classes.
    - **Economic cost (USD):** the business-meaning metric.  AUC may
      stay flat while cost rises if the score-distribution bunching
      changes near the decision threshold.

    The 5% degradation threshold (configurable via
    `Settings.performance_degradation_threshold`) is the spec value.
    Each metric alerts independently — three alerts max per
    `check_and_alert` call, one JSONL line each.

Trade-offs considered:
    - **Stateless analyzer (no per-prediction history).**  Caller slices
      the most-recent N labelled predictions and hands in a DataFrame.
      "Rolling" is satisfied by the caller-side window choice
      (`Settings.performance_window_size` hints at the expected size,
      but PerformanceMonitor doesn't enforce it).  Mirrors DriftMonitor
      6.1.b.  Rejected: a stateful `add_predictions(df) → metrics`
      adds a global, hides the windowing decision from operators, and
      complicates testing.

    - **Caller-passes-DataFrame (no Postgres queries).**  Sprint 5.2.a's
      `predictions` table stores `(score, decision)` keyed by `txn_id`;
      combined with a label feed (test-parquet `isFraud` for the
      simulation) the caller produces a `(score, label, decision?)`
      DataFrame.  This module does the math + alerting; it does NOT
      own the data plumbing.  Same offline-batch contract as 6.1.b.

    - **Three metrics (AUC / AUC-PR / cost), each alerting
      independently.**  Diminishing returns past three.  F1 / accuracy
      / Brier add complexity without surfacing failures the trio misses.

    - **Fractional degradation `(baseline - current) / baseline > 0.05`.**
      Portable across metrics with different scales.  Cost flips the
      sign internally (higher cost = worse).  Rejected: absolute
      thresholds (don't transfer between metrics); per-metric thresholds
      (more config; the 5% default covers the common case; operators
      override globally via Settings).

    - **Training baselines as Settings fields, operator-curated.**  The
      LightGBM model manifest at `models/sprint3/lightgbm_model_manifest.json`
      carries `best_score` (AUC) but NOT AUC-PR or economic cost.
      Auto-loading would require a Sprint 3 retrofit; out of scope here.
      Operator updates the three baselines on each model retrain via
      `.env` or `Settings()` kwargs.  See risk register for drift mitigation.

    - **Decision-column optional.**  In production the predictions table
      already carries a `decision` column (Sprint 5.2.a) — caller forwards
      it.  In offline replay the caller may have only scores; we threshold
      internally via `Settings.decision_threshold` (post-Sprint-4.4
      cost-optimal value 0.080).  AUC / AUC-PR always come from `score`
      regardless.

    - **Append-only JSONL alerts at `logs/performance/{run_id}/...`.**
      Mirrors 6.1.b's `logs/drift/{run_id}/drift_alerts.jsonl` exactly.
      Rejected: Postgres `performance_alerts` table — slow-moving log
      stream; JSONL ships with no schema migration.

    - **`@log_call` + `get_logger(__name__)`.**  Sprint 5.5 logging
      discipline — input shape, output shape, duration in ms on every
      public method.

    - **Single-class window edge case.**  An all-`label=0` (or
      all-`label=1`) recent_window crashes `roc_auc_score` with a
      ValueError.  We catch it, return NaN for AUC + AUC-PR (cost is
      still computable), and let `compare_to_baseline` skip NaN metrics
      (no false alerts).  The WARNING log surfaces the situation to
      the operator.

Cross-references:
    - `src/fraud_engine/utils/metrics.py:68-161` — `economic_cost(y_true,
      y_pred, ...)` returns a dict; we extract `total_cost` and discard
      the rest.
    - `src/fraud_engine/evaluation/stratified.py:95,783-784` — sklearn
      `roc_auc_score` / `average_precision_score` import precedent (no
      new project-side wrapper needed).
    - `src/fraud_engine/monitoring/drift.py` (Sprint 6.1.b) — the
      sibling module whose pattern this one mirrors (stateless, caller-
      passes-DataFrame, JSONL alerts).
    - `src/fraud_engine/monitoring/prometheus_metrics.py` (Sprint 6.1.a)
      — the live-metrics surface.  Operators see all three monitoring
      layers in concert: live histogram from Sprint 6.1.a, offline PSI
      from Sprint 6.1.b, offline performance regression from this PR.
    - `CLAUDE.md` §3 (monitoring as the pipeline endpoint), §4
      (`monitoring/` module home), §5.4 (no hardcoded thresholds), §5.5
      (logging discipline).
"""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Final
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from fraud_engine.config.settings import Settings, get_settings
from fraud_engine.utils.logging import get_logger, log_call
from fraud_engine.utils.metrics import economic_cost

# ---------------------------------------------------------------------
# Module constants.
# ---------------------------------------------------------------------

# Required columns on the recent_window DataFrame.  `decision` is
# optional — caller may forward the Sprint 5.2.a-persisted value, OR
# we threshold scores internally via Settings.decision_threshold.
_REQUIRED_COLUMNS: Final[tuple[str, ...]] = ("score", "label")

# Metric names — single source of truth used by both
# `compute_rolling_metrics` keys and `compare_to_baseline` lookups.
_METRIC_AUC: Final[str] = "auc"
_METRIC_AUC_PR: Final[str] = "auc_pr"
_METRIC_COST: Final[str] = "cost"
_METRIC_NAMES: Final[tuple[str, ...]] = (_METRIC_AUC, _METRIC_AUC_PR, _METRIC_COST)

# Higher-is-better metrics: degradation = (baseline - current) / baseline.
# Lower-is-better metrics (cost): degradation = (current - baseline) / baseline.
# Both flavours: positive degradation == "worse than baseline".
_LOWER_IS_BETTER: Final[frozenset[str]] = frozenset({_METRIC_COST})

# Minimum unique label classes required for AUC / AUC-PR to be defined.
# A single-class window has only positives or only negatives — both
# discrimination metrics are mathematically undefined.
_MIN_CLASSES_FOR_AUC: Final[int] = 2

_logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Internal: per-metric degradation record.
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class _Degradation:
    """One metric whose drop exceeds the alert threshold.

    Attributes:
        metric: One of `_METRIC_NAMES`.
        baseline: The training-time reference value.
        current: The metric on the recent_window.
        degradation: Fractional, sign-corrected so positive means
            "worse than baseline" regardless of the metric's
            higher-is-better orientation.
    """

    metric: str
    baseline: float
    current: float
    degradation: float


# ---------------------------------------------------------------------
# Public class.
# ---------------------------------------------------------------------


class PerformanceMonitor:
    """Stateless rolling-performance analyzer with baseline comparison.

    Public API:
        - `compute_rolling_metrics(recent_window) → dict`
        - `compare_to_baseline(metrics) → list[_Degradation]`
        - `check_and_alert(recent_window, *, run_id, alert_log_dir) → int`

    Lifecycle:
        Constructor reads `Settings` (or accepts an injected instance).
        No on-disk artefacts loaded — baselines + threshold + cost
        coefficients are read straight from Settings on every call so
        operator-side `.env` updates take effect without restart.

    Business rationale + trade-offs considered: see module docstring.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Configure the monitor.

        Args:
            settings: Inject a Settings instance for tests.  None →
                `get_settings()` (the lru-cached singleton).
        """
        self._settings: Settings = settings if settings is not None else get_settings()

    # ---------- core math --------------------------------------------

    @log_call
    def compute_rolling_metrics(self, recent_window: pd.DataFrame) -> dict[str, float]:
        """Return AUC, AUC-PR, economic cost, n_recent on the given window.

        Required columns on `recent_window`: `score` (float ∈ [0, 1])
        and `label` (int ∈ {0, 1}).  Optional column: `decision` (int
        ∈ {0, 1}); if absent, scores are thresholded internally via
        `Settings.decision_threshold`.

        Edge cases:
            - Empty `recent_window` → ValueError (clearer message than
              sklearn's downstream).
            - All-one-class label window → AUC + AUC-PR are NaN (with
              WARNING logged); cost is still computable.

        Args:
            recent_window: DataFrame matching the column contract above.

        Returns:
            `{"auc": float, "auc_pr": float, "cost": float, "n_recent": int}`.
            AUC / AUC-PR may be NaN when the window is single-class;
            `compare_to_baseline` skips NaN metrics so no false alerts.
        """
        missing = [c for c in _REQUIRED_COLUMNS if c not in recent_window.columns]
        if missing:
            raise ValueError(
                f"PerformanceMonitor.compute_rolling_metrics: recent_window "
                f"missing required columns {missing}; got {list(recent_window.columns)}"
            )
        if len(recent_window) == 0:
            raise ValueError("PerformanceMonitor.compute_rolling_metrics: recent_window is empty")

        y_true = recent_window["label"].to_numpy(dtype=np.int64)
        y_score = recent_window["score"].to_numpy(dtype=np.float64)

        if "decision" in recent_window.columns:
            y_pred = recent_window["decision"].to_numpy(dtype=np.int64)
        else:
            y_pred = (y_score >= self._settings.decision_threshold).astype(np.int64)

        # Discrimination metrics — AUC + AUC-PR.  Detect single-class
        # windows up-front because the two sklearn primitives behave
        # differently: roc_auc_score raises ValueError, but
        # average_precision_score silently returns 0.0 (with only a
        # UserWarning).  Returning NaN for both keeps the comparator's
        # skip-NaN logic clean.  We still wrap in try/except for any
        # other ValueError sklearn might raise (e.g. all-NaN scores).
        n_unique_labels = int(len(np.unique(y_true)))
        if n_unique_labels < _MIN_CLASSES_FOR_AUC:
            _logger.warning(
                "performance_monitor.single_class_window",
                n_recent=len(recent_window),
                positive_count=int(y_true.sum()),
            )
            auc = float("nan")
            auc_pr = float("nan")
        else:
            try:
                auc = float(roc_auc_score(y_true, y_score))
                auc_pr = float(average_precision_score(y_true, y_score))
            except ValueError as exc:
                _logger.warning(
                    "performance_monitor.discrimination_undefined",
                    n_recent=len(recent_window),
                    detail=str(exc),
                )
                auc = float("nan")
                auc_pr = float("nan")

        # Economic cost — uses the Sprint-4 `economic_cost` primitive
        # with cost coefficients from Settings (operator can sweep via
        # .env).  We extract `total_cost` only; the dict's per-class
        # counts are useful for Sprint 4's threshold optimiser but
        # not for runtime monitoring.
        cost_dict = economic_cost(
            y_true=y_true,
            y_pred=y_pred,
            fraud_cost=self._settings.fraud_cost_usd,
            fp_cost=self._settings.fp_cost_usd,
            tp_cost=self._settings.tp_cost_usd,
        )
        cost = float(cost_dict["total_cost"])

        return {
            _METRIC_AUC: auc,
            _METRIC_AUC_PR: auc_pr,
            _METRIC_COST: cost,
            "n_recent": float(len(recent_window)),
        }

    # ---------- baseline comparison ----------------------------------

    @log_call
    def compare_to_baseline(self, metrics: dict[str, float]) -> list[_Degradation]:
        """Return one _Degradation per metric whose drop exceeds threshold.

        Skips metrics whose `current` value is NaN (e.g. single-class
        AUC) — no false alerts.

        Sign convention:
            - Higher-is-better metrics (`auc`, `auc_pr`):
                degradation = (baseline - current) / baseline
            - Lower-is-better metrics (`cost`):
                degradation = (current - baseline) / baseline
            Both → positive degradation means "worse than baseline".

        Args:
            metrics: Output of `compute_rolling_metrics` (or any dict
                with the same `_METRIC_NAMES` keys).

        Returns:
            List of `_Degradation` records, one per metric whose
            degradation exceeds `Settings.performance_degradation_threshold`.
            Empty list = healthy (no alerts to fire).
        """
        threshold = float(self._settings.performance_degradation_threshold)
        degradations: list[_Degradation] = []

        for metric in _METRIC_NAMES:
            current = metrics.get(metric)
            if current is None or not np.isfinite(current):
                continue
            baseline = self._baseline_for(metric)
            if baseline <= 0.0:
                # Defensive: a zero or negative baseline makes the
                # fractional degradation undefined (division by zero
                # for AUC/AUC-PR; sign inversion for cost).  Skip with
                # a one-line WARNING so the operator notices.
                _logger.warning(
                    "performance_monitor.invalid_baseline",
                    metric=metric,
                    baseline=baseline,
                )
                continue
            degradation = self._compute_degradation(metric, baseline, current)
            if degradation > threshold:
                degradations.append(
                    _Degradation(
                        metric=metric,
                        baseline=baseline,
                        current=current,
                        degradation=degradation,
                    )
                )

        return degradations

    # ---------- alerting ---------------------------------------------

    @log_call
    def check_and_alert(
        self,
        recent_window: pd.DataFrame,
        *,
        run_id: str | None = None,
        alert_log_dir: Path | None = None,
    ) -> int:
        """Compute → compare → append JSONL per degraded metric.

        Idempotent only across distinct `run_id`s — repeated calls with
        the same `run_id` append to the same file (multiple cron
        invocations on the same logical run concatenate).

        If no metrics degrade, the file is NOT created — operators can
        grep for the file's existence as a clean alerted/not-alerted
        signal.

        Args:
            recent_window: DataFrame matching `compute_rolling_metrics`
                contract.
            run_id: Correlation tag.  Threaded into the JSONL records
                AND the directory path.  None → fresh `uuid4().hex`
                (matches the request_id idiom from Sprint 5.1.f).
            alert_log_dir: Override default `logs/performance`.  None →
                `Settings.performance_alert_log_dir`.  Tests pass a
                `tmp_path` for isolation.

        Returns:
            Count of alert records written.  0 means no degradation
            (no file created).  Caller can `sys.exit(1 if count > 0 else 0)`
            in a daily cron to trip on-call paging.
        """
        effective_run_id = run_id if run_id is not None else uuid4().hex
        effective_dir = (
            alert_log_dir if alert_log_dir is not None else self._settings.performance_alert_log_dir
        )
        threshold = float(self._settings.performance_degradation_threshold)

        metrics = self.compute_rolling_metrics(recent_window)
        degradations = self.compare_to_baseline(metrics)

        if not degradations:
            _logger.info(
                "performance_monitor.no_degradation",
                run_id=effective_run_id,
                metrics={k: v for k, v in metrics.items()},
                threshold=threshold,
            )
            return 0

        target_path = effective_dir / effective_run_id / "performance_alerts.jsonl"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        n_recent = int(metrics["n_recent"])
        timestamp = datetime.now(UTC).isoformat()

        with target_path.open("a", encoding="utf-8") as fh:
            for deg in degradations:
                record = {
                    "timestamp": timestamp,
                    "run_id": effective_run_id,
                    "metric": deg.metric,
                    "baseline": deg.baseline,
                    "current": deg.current,
                    "degradation": deg.degradation,
                    "threshold": threshold,
                    "n_recent": n_recent,
                }
                fh.write(json.dumps(record, sort_keys=True) + "\n")

        _logger.info(
            "performance_monitor.degradation_alerted",
            run_id=effective_run_id,
            n_alerts=len(degradations),
            threshold=threshold,
            output_path=str(target_path),
            degraded_metrics=[d.metric for d in degradations],
        )
        return len(degradations)

    # ---------- helpers ----------------------------------------------

    def _baseline_for(self, metric: str) -> float:
        """Look up the training-time baseline value for `metric` from Settings."""
        if metric == _METRIC_AUC:
            return float(self._settings.performance_training_auc)
        if metric == _METRIC_AUC_PR:
            return float(self._settings.performance_training_auc_pr)
        if metric == _METRIC_COST:
            return float(self._settings.performance_training_cost)
        raise ValueError(f"Unknown metric: {metric!r}")

    @staticmethod
    def _compute_degradation(metric: str, baseline: float, current: float) -> float:
        """Fractional degradation, sign-corrected to 'positive == worse'."""
        if metric in _LOWER_IS_BETTER:
            # Cost: higher current is worse than baseline.
            return (current - baseline) / baseline
        # AUC, AUC-PR: lower current is worse than baseline.
        return (baseline - current) / baseline


__all__ = ["PerformanceMonitor"]
