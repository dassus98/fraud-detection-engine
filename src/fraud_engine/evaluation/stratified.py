"""Stratified evaluation: AUC, PR-AUC, economic cost per business segment.

Sprint 4 prompt 4.2: per-segment evaluator that consumes Model A's
calibrated probabilities and surfaces where they perform unevenly
across business-meaningful slices.

Business rationale:
    A model with global AUC = 0.93 can still fail catastrophically on
    the slices that matter — high-value transactions, mobile sessions,
    no-identity rows (CLAUDE.md §1: "only 24 % of transactions have
    device/identity data"). Aggregate metrics hide segment-level skew
    by construction; this module reports per-stratum metrics so a
    reviewer sees the failure modes the deployment will face on day
    one, not on day fourteen of post-mortem.

    The five stratification axes mirror the slices a senior fraud-team
    reviewer asks about during deployment review: amount bucket
    (low-amount transactions are a known higher-fraud regime), product
    code (different products have different risk profiles), device
    type (mobile sessions are different attack surfaces), identity
    coverage (no-identity rows are the harder fraud detection problem),
    and temporal month (drift across the val / test boundary is the
    canonical IEEE-CIS sanity check).

Trade-offs considered:
    - **Stateless class wrapping a stateful primitive.** Mirrors
      `EconomicCostModel`'s shape — cost-model + threshold +
      min-stratum floor in the constructor; no learned state.
      `evaluate(...)` is the single public orchestrator that fans
      out to five private per-axis helpers.
    - **Long-format DataFrame** (one row per stratum) over wide-format.
      35+ columns is unreadable; long-format lets the caller
      `groupby('stratum_axis')` and pivot at will. Mirrors
      `economic.py`'s cost-curve column convention.
    - **Single `evaluate` orchestrator + 5 private helpers**, NOT
      public per-axis methods. A reviewer wants one call returning
      one frame; per-axis methods would balloon the public surface
      with no flexibility win (the long-format frame is already
      filterable).
    - **`id_01.notna()` as identity-coverage probe** — highest-non-
      null `id_*` column per CLAUDE.md §1. `DeviceType.notna()` would
      conflate identity-coverage with the device-type axis (smearing
      two signals); a single-column probe is reliable and documented.
    - **Skip-with-warning on missing column.** A tier-1-only
      experimental frame missing `DeviceType` shouldn't crash the
      evaluator. Logs `WARNING(stratified.axis_skipped, axis=...,
      reason=...)` and omits that axis from the output. Reviewer-
      friendly on partial frames.
    - **Include single-class strata with NaN AUC/PR-AUC** rather than
      drop them. A reviewer wants to see "this stratum had only
      positives" — dropping hides the skew. Cost is still computed
      (it's well-defined on a single class).
    - **`min_stratum_size: int = 50` floor** — strata below this
      return NaN AUC/PR-AUC (cost still computed). AUC = 1.0 on
      n=3 rows is meaningless noise that would discolour the
      heatmap; the floor pins the trustworthiness threshold.
    - **Heatmap z-scores per metric column with diverging colormap**,
      with axis-aware sign flip for cost columns so red always means
      "this stratum is worse" regardless of metric direction.
      Annotates each cell with raw value + sample size
      (`"AUC=0.920\n(n=15.2K)"`) so a reader can judge cell
      trustworthiness at a glance.
    - **Month axis is a keyword arg, not derived from
      `frame['timestamp']`.** Tier-5 parquet drops `timestamp`
      (`build_features_all_tiers.py:110-111`); making the caller
      pass it explicitly is least-surprising. Pass `month=None`
      to skip the axis.
    - **MLflow logging deferred to Sprint 4.x+** (mirrors 4.1's
      deferral). The `evaluate()` DataFrame and `plot_heatmap()`
      Axes are clean handoff points for a future MLflow-aware
      reporter; this module stays pure-numerical / pure-plotting.

Cross-references:
    - `src/fraud_engine/evaluation/economic.py:260-533` — sibling
      evaluator; cost-model dependency injected in constructor.
    - `src/fraud_engine/evaluation/calibration.py:223-300` — plot
      pattern (Axes return, ax=None semantics, text annotation,
      caller saves).
    - `src/fraud_engine/config/settings.py:228-237` —
      ``decision_threshold``, the default this module reads when
      the caller doesn't supply one.
    - `CLAUDE.md` §1 (24 % identity coverage), §8 (cost defaults).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from numpy.typing import ArrayLike
from sklearn.metrics import average_precision_score, roc_auc_score

from fraud_engine.config.settings import get_settings
from fraud_engine.evaluation.economic import EconomicCostModel
from fraud_engine.utils.logging import get_logger

_logger = get_logger(__name__)

# ---------------------------------------------------------------------
# Module constants.
# ---------------------------------------------------------------------

# Amount-bucket boundaries: half-open intervals [low, high). The lower
# rail is -inf so any non-negative TransactionAmt falls into a bucket;
# the upper rail is +inf so the >$1K bucket catches everything ≥ 1000.
# Edge case: 50.0 lands in $50-200 (not <$50) per the half-open
# convention, pinned in the test suite.
_DEFAULT_AMOUNT_BUCKETS: Final[tuple[tuple[float, float, str], ...]] = (
    (-np.inf, 50.0, "<$50"),
    (50.0, 200.0, "$50-200"),
    (200.0, 500.0, "$200-500"),
    (500.0, 1000.0, "$500-1K"),
    (1000.0, np.inf, ">$1K"),
)

# Stratification axes the evaluator knows how to compute. The order
# here is the order they appear in the long-format DataFrame.
_AXES: Final[tuple[str, ...]] = (
    "amount_bucket",
    "product_cd",
    "device_type",
    "identity_coverage",
    "month",
)

# Long-format DataFrame columns. Mirrors the per-stratum dict shape
# emitted by `_stratum_metrics`.
_RESULT_COLUMNS: Final[tuple[str, ...]] = (
    "stratum_axis",
    "stratum_value",
    "n_rows",
    "n_pos",
    "fraud_rate",
    "auc",
    "pr_auc",
    "total_cost",
    "cost_per_txn",
)

# Heatmap defaults. Three metrics chosen to give a reviewer the
# discriminative power axis (AUC, PR-AUC) AND the operational impact
# axis (cost_per_txn) in one chart.
_DEFAULT_HEATMAP_METRICS: Final[tuple[str, ...]] = ("auc", "pr_auc", "cost_per_txn")

# Metrics where higher value means WORSE outcome (cost-like). Used by
# the heatmap to flip the z-score sign so red consistently means "this
# stratum is performing worse" across all columns.
_COST_LIKE_METRICS: Final[frozenset[str]] = frozenset({"total_cost", "cost_per_txn"})

# Min n_rows below which AUC/PR-AUC are not trusted (return NaN with
# warning). 50 is the smallest size where a 10 % fraud rate gives
# expected n_pos = 5 — barely enough for AUC to be non-trivial.
_DEFAULT_MIN_STRATUM_SIZE: Final[int] = 50

# Column-name constants for the stratification axes. Captures the
# IEEE-CIS contract the evaluator depends on.
_AMOUNT_COL: Final[str] = "TransactionAmt"
_PRODUCT_COL: Final[str] = "ProductCD"
_DEVICE_COL: Final[str] = "DeviceType"
_IDENTITY_PROBE_COL: Final[str] = "id_01"

# Stratum-value labels for axes that produce explicit "(null)" /
# "(missing)" / "has" / "no" buckets.
_DEVICE_NULL_LABEL: Final[str] = "(null)"
_PRODUCT_MISSING_LABEL: Final[str] = "(missing)"
_HAS_IDENTITY_LABEL: Final[str] = "has_identity"
_NO_IDENTITY_LABEL: Final[str] = "no_identity"

# Probability bounds for the y_scores validation guard.
_PROB_LOWER: Final[float] = 0.0
_PROB_UPPER: Final[float] = 1.0

# Heatmap colormap range (z-score units). Clamps very-extreme cells so
# one outlier doesn't compress everything else into the centre.
_HEATMAP_VMIN: Final[float] = -2.0
_HEATMAP_VMAX: Final[float] = 2.0

# Sample-size formatter scale boundaries (compact "K" / "M" labels).
_THOUSAND: Final[int] = 1_000
_MILLION: Final[int] = 1_000_000

# Minimum class count for a stratum to be AUC/PR-AUC eligible (sklearn
# requires both classes present). Below this → NaN with warning.
_MIN_CLASSES_FOR_AUC: Final[int] = 2

# Minimum finite-value count for the heatmap z-score to be meaningful.
# A column with only one finite value has undefined std; we render
# zeros (neutral colour) instead.
_MIN_FINITE_FOR_ZSCORE: Final[int] = 2


# ---------------------------------------------------------------------
# Formatting helpers (heatmap cell labels).
# ---------------------------------------------------------------------


def _format_n(n: int) -> str:
    """Compact sample-size label: 1234 → '1.2K', 1500000 → '1.5M'."""
    if n >= _MILLION:
        return f"{n / _MILLION:.1f}M"
    if n >= _THOUSAND:
        return f"{n / _THOUSAND:.1f}K"
    return str(n)


def _format_cell(metric: str, value: float, n: int) -> str:
    """Build a heatmap cell label: raw value (formatted) + sample size."""
    if not np.isfinite(value):
        return f"NaN\n(n={_format_n(n)})"
    if metric in _COST_LIKE_METRICS:
        return f"${value:,.2f}\n(n={_format_n(n)})"
    if metric == "fraud_rate":
        return f"{value:.1%}\n(n={_format_n(n)})"
    # auc / pr_auc / generic numeric fall through.
    return f"{value:.3f}\n(n={_format_n(n)})"


# ---------------------------------------------------------------------
# Public class.
# ---------------------------------------------------------------------


class StratifiedEvaluator:
    """Per-segment AUC / PR-AUC / economic cost across 5 business axes.

    Public API:
        - `evaluate(y_true, y_scores, frame, *, month=None) -> pd.DataFrame`
          — long-format frame with one row per stratum; columns
          ``stratum_axis, stratum_value, n_rows, n_pos, fraud_rate,
          auc, pr_auc, total_cost, cost_per_txn``.
        - `plot_heatmap(eval_df, metrics=..., ax=None) -> Axes` —
          single heatmap visualisation; rows = (axis, value) pairs,
          columns = metrics, colour = z-score normalised within
          each metric column (cost columns sign-flipped so red =
          worse).

    The class is stateless: configuration (cost model, threshold,
    min-stratum floor) is set in the constructor; data lives in the
    arguments to `evaluate`.
    """

    def __init__(
        self,
        cost_model: EconomicCostModel | None = None,
        threshold: float | None = None,
        min_stratum_size: int = _DEFAULT_MIN_STRATUM_SIZE,
    ) -> None:
        """Construct the evaluator.

        Args:
            cost_model: `EconomicCostModel` instance whose costs drive
                the per-stratum cost computation. Defaults to
                `EconomicCostModel()` (Settings defaults).
            threshold: Decision threshold for binarising `y_scores`
                into `y_pred` for the cost component. AUC / PR-AUC
                are threshold-free; only `compute_cost` needs this.
                Defaults to `Settings.decision_threshold`.
            min_stratum_size: Strata with fewer than this many rows
                return NaN AUC / PR-AUC (cost still computed; warning
                logged). Defaults to 50 — small enough not to drop
                small product groups, large enough to gate against
                noise like AUC=1.0 on n=3.

        Raises:
            ValueError: If `threshold` is outside [0, 1] or
                `min_stratum_size` is < 1.
        """
        if min_stratum_size < 1:
            raise ValueError(
                f"StratifiedEvaluator: min_stratum_size must be >= 1, got {min_stratum_size}"
            )
        settings = get_settings()
        resolved_threshold = settings.decision_threshold if threshold is None else threshold
        if resolved_threshold < _PROB_LOWER or resolved_threshold > _PROB_UPPER:
            raise ValueError(
                f"StratifiedEvaluator: threshold must be in [0, 1], " f"got {resolved_threshold}"
            )
        self._cost_model: EconomicCostModel = (
            cost_model if cost_model is not None else EconomicCostModel()
        )
        self._threshold: float = float(resolved_threshold)
        self._min_stratum_size: int = int(min_stratum_size)

    # -----------------------------------------------------------------
    # Read-only config snapshot (for logging / manifest persistence).
    # -----------------------------------------------------------------

    @property
    def threshold(self) -> float:
        """Effective decision threshold used for cost binarisation."""
        return self._threshold

    @property
    def min_stratum_size(self) -> int:
        """Effective minimum stratum size for trusted AUC/PR-AUC."""
        return self._min_stratum_size

    @property
    def cost_model(self) -> EconomicCostModel:
        """The injected cost model (reference, not snapshot)."""
        return self._cost_model

    # -----------------------------------------------------------------
    # Public: evaluate.
    # -----------------------------------------------------------------

    def evaluate(
        self,
        y_true: ArrayLike,
        y_scores: ArrayLike,
        frame: pd.DataFrame,
        *,
        month: pd.Series[Any] | None = None,
    ) -> pd.DataFrame:
        """Compute per-stratum metrics across all five axes.

        Args:
            y_true: Ground-truth binary labels in {0, 1}.
            y_scores: Calibrated probabilities in [0, 1] (from Model
                A's `Calibrator.transform`).
            frame: DataFrame containing the stratification columns
                (`TransactionAmt`, `ProductCD`, `DeviceType`,
                `id_01`). Must have `len(frame) == len(y_true)`.
                Missing columns are skipped with a warning rather
                than raised — reviewer-friendly on partial frames.
            month: Optional Series of integer month values
                (`len(month) == len(y_true)`). Tier-5 parquet drops
                `timestamp`; the caller derives the month and passes
                it explicitly. Pass `None` to skip the temporal axis.

        Returns:
            Long-format DataFrame with columns ``stratum_axis,
            stratum_value, n_rows, n_pos, fraud_rate, auc, pr_auc,
            total_cost, cost_per_txn``. Sorted by `stratum_axis` in
            the canonical `_AXES` order.

        Raises:
            ValueError: If `y_true` / `y_scores` shapes mismatch, if
                `y_scores` contains values outside [0, 1], if
                `len(frame) != len(y_true)`, or if `month is not None`
                and `len(month) != len(y_true)`.
        """
        y_true_arr, y_scores_arr = self._validate_inputs(y_true, y_scores, frame, month)

        rows: list[dict[str, Any]] = []

        # Amount bucket
        if _AMOUNT_COL in frame.columns:
            rows.extend(
                self._evaluate_by_amount_bucket(y_true_arr, y_scores_arr, frame[_AMOUNT_COL])
            )
        else:
            _logger.warning(
                "stratified.axis_skipped",
                axis="amount_bucket",
                reason=f"column {_AMOUNT_COL!r} missing from frame",
            )

        # ProductCD
        if _PRODUCT_COL in frame.columns:
            rows.extend(self._evaluate_by_product_cd(y_true_arr, y_scores_arr, frame[_PRODUCT_COL]))
        else:
            _logger.warning(
                "stratified.axis_skipped",
                axis="product_cd",
                reason=f"column {_PRODUCT_COL!r} missing from frame",
            )

        # DeviceType
        if _DEVICE_COL in frame.columns:
            rows.extend(self._evaluate_by_device_type(y_true_arr, y_scores_arr, frame[_DEVICE_COL]))
        else:
            _logger.warning(
                "stratified.axis_skipped",
                axis="device_type",
                reason=f"column {_DEVICE_COL!r} missing from frame",
            )

        # Identity coverage (probe column)
        if _IDENTITY_PROBE_COL in frame.columns:
            rows.extend(
                self._evaluate_by_identity_coverage(
                    y_true_arr, y_scores_arr, frame[_IDENTITY_PROBE_COL]
                )
            )
        else:
            _logger.warning(
                "stratified.axis_skipped",
                axis="identity_coverage",
                reason=f"probe column {_IDENTITY_PROBE_COL!r} missing from frame",
            )

        # Month (external arg)
        if month is not None:
            rows.extend(self._evaluate_by_month(y_true_arr, y_scores_arr, month))
        else:
            _logger.warning(
                "stratified.axis_skipped",
                axis="month",
                reason="month argument is None",
            )

        if not rows:
            # Defensive: every axis was skipped. Return an empty
            # frame with the right column shape so downstream code
            # doesn't get an unexpected schema.
            return pd.DataFrame(columns=list(_RESULT_COLUMNS))

        return pd.DataFrame(rows, columns=list(_RESULT_COLUMNS))

    # -----------------------------------------------------------------
    # Public: plot heatmap.
    # -----------------------------------------------------------------

    def plot_heatmap(
        self,
        eval_df: pd.DataFrame,
        metrics: Sequence[str] = _DEFAULT_HEATMAP_METRICS,
        ax: Axes | None = None,
    ) -> Axes:
        """Render a single heatmap of per-stratum metrics.

        Rows are (axis, value) pairs from `eval_df`; columns are the
        chosen metrics. Cell colour is z-score normalised per metric
        column, with cost-like metrics sign-flipped so red always
        means "this stratum is worse" regardless of metric direction.
        Cells are annotated with raw value + sample size.

        Args:
            eval_df: Long-format DataFrame from `evaluate(...)`.
            metrics: Tuple of metric column names to render. Defaults
                to ``("auc", "pr_auc", "cost_per_txn")``.
            ax: Optional matplotlib `Axes`. If `None`, creates a new
                figure sized to the number of strata.

        Returns:
            The `Axes` instance carrying the heatmap. Caller saves
            via `ax.figure.savefig(...)`.

        Raises:
            ValueError: If any metric is not a column of `eval_df`.
        """
        missing = [m for m in metrics if m not in eval_df.columns]
        if missing:
            raise ValueError(
                f"StratifiedEvaluator.plot_heatmap: unknown metric(s) "
                f"{sorted(missing)}; available: {sorted(eval_df.columns.tolist())}"
            )

        n_strata = len(eval_df)

        if ax is None:
            fig_height = max(4.0, 0.4 * max(n_strata, 1) + 1.0)
            _, ax = plt.subplots(figsize=(8.0, fig_height))

        if n_strata == 0:
            # Defensive: empty eval_df. Show a placeholder message.
            ax.text(
                0.5,
                0.5,
                "(no strata to plot)",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            return ax

        # Raw + z-score-normalised matrices.
        raw_data = eval_df[list(metrics)].to_numpy(dtype=np.float64)
        z_data = self._z_score_for_heatmap(raw_data, metrics)
        n_rows_arr = eval_df["n_rows"].to_numpy(dtype=np.int64)
        row_labels = [
            f"{row['stratum_axis']}: {row['stratum_value']}" for _, row in eval_df.iterrows()
        ]

        # Render. Diverging colormap; bad (NaN) cells are light grey
        # so a missing-value cell is visually distinct from a strong
        # negative z-score. matplotlib's `imshow` honours `cmap.set_bad`
        # for NaN cells in plain ndarrays, no masked-array needed.
        cmap = plt.get_cmap("RdYlBu_r").copy()
        cmap.set_bad(color="lightgray")
        im = ax.imshow(
            z_data,
            cmap=cmap,
            aspect="auto",
            vmin=_HEATMAP_VMIN,
            vmax=_HEATMAP_VMAX,
        )

        # Cell annotations: raw value + sample size.
        for i in range(n_strata):
            for j, metric in enumerate(metrics):
                label = _format_cell(metric, float(raw_data[i, j]), int(n_rows_arr[i]))
                ax.text(
                    j,
                    i,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black",
                )

        # Axis decoration.
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(list(metrics))
        ax.set_yticks(range(n_strata))
        ax.set_yticklabels(row_labels)
        ax.set_xlabel("metric")
        ax.set_ylabel("stratum")
        ax.set_title("Stratified evaluation (red = worse, normalised per metric)")

        # Colorbar: z-score axis.
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("z-score (worse →)")

        # TODO(sprint-4.x): MLflow integration — log eval_df + this
        # figure as artefacts in the same run as the model the scores
        # came from. Deferred per plan; this module stays pure-plot.

        return ax

    # -----------------------------------------------------------------
    # Validation helper.
    # -----------------------------------------------------------------

    def _validate_inputs(
        self,
        y_true: ArrayLike,
        y_scores: ArrayLike,
        frame: pd.DataFrame,
        month: pd.Series[Any] | None,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Coerce + validate; return 1-D arrays. Validate ONCE per evaluate call."""
        y_true_arr = np.asarray(y_true).ravel()
        y_scores_arr = np.asarray(y_scores, dtype=np.float64).ravel()
        if y_true_arr.shape != y_scores_arr.shape:
            raise ValueError(
                f"StratifiedEvaluator: y_true and y_scores shape mismatch — "
                f"{y_true_arr.shape} vs {y_scores_arr.shape}"
            )
        if y_scores_arr.size > 0:
            score_min = float(y_scores_arr.min())
            score_max = float(y_scores_arr.max())
            if score_min < _PROB_LOWER or score_max > _PROB_UPPER:
                raise ValueError(
                    f"StratifiedEvaluator: y_scores must be in [0, 1] "
                    f"(calibration contract); got min={score_min}, max={score_max}"
                )
        if len(frame) != len(y_true_arr):
            raise ValueError(
                f"StratifiedEvaluator: frame length ({len(frame)}) does not "
                f"match y_true length ({len(y_true_arr)})"
            )
        if month is not None and len(month) != len(y_true_arr):
            raise ValueError(
                f"StratifiedEvaluator: month series length ({len(month)}) "
                f"does not match y_true length ({len(y_true_arr)})"
            )
        return y_true_arr, y_scores_arr

    # -----------------------------------------------------------------
    # Per-axis helpers (private; receive already-validated arrays).
    # -----------------------------------------------------------------

    def _evaluate_by_amount_bucket(
        self,
        y: np.ndarray[Any, Any],
        s: np.ndarray[Any, Any],
        amt: pd.Series[Any],
    ) -> list[dict[str, Any]]:
        """One row per amount bucket. Half-open [low, high)."""
        rows: list[dict[str, Any]] = []
        amt_arr = amt.to_numpy()
        for low, high, label in _DEFAULT_AMOUNT_BUCKETS:
            mask = (amt_arr >= low) & (amt_arr < high)
            metrics = self._stratum_metrics(
                y[mask],
                s[mask],
                axis="amount_bucket",
                value=label,
            )
            rows.append({"stratum_axis": "amount_bucket", "stratum_value": label, **metrics})
        return rows

    def _evaluate_by_product_cd(
        self,
        y: np.ndarray[Any, Any],
        s: np.ndarray[Any, Any],
        pcd: pd.Series[Any],
    ) -> list[dict[str, Any]]:
        """One row per ProductCD value (sorted) plus a `(missing)` bucket if any NaNs."""
        rows: list[dict[str, Any]] = []
        unique_vals = sorted(pcd.dropna().unique().tolist())
        for val in unique_vals:
            mask = (pcd == val).to_numpy()
            metrics = self._stratum_metrics(y[mask], s[mask], axis="product_cd", value=str(val))
            rows.append(
                {
                    "stratum_axis": "product_cd",
                    "stratum_value": str(val),
                    **metrics,
                }
            )
        nan_mask = pcd.isna().to_numpy()
        if nan_mask.any():
            metrics = self._stratum_metrics(
                y[nan_mask],
                s[nan_mask],
                axis="product_cd",
                value=_PRODUCT_MISSING_LABEL,
            )
            rows.append(
                {
                    "stratum_axis": "product_cd",
                    "stratum_value": _PRODUCT_MISSING_LABEL,
                    **metrics,
                }
            )
        return rows

    def _evaluate_by_device_type(
        self,
        y: np.ndarray[Any, Any],
        s: np.ndarray[Any, Any],
        dev: pd.Series[Any],
    ) -> list[dict[str, Any]]:
        """One row per DeviceType value (sorted) plus an explicit `(null)` bucket."""
        rows: list[dict[str, Any]] = []
        unique_vals = sorted(dev.dropna().unique().tolist())
        for val in unique_vals:
            mask = (dev == val).to_numpy()
            metrics = self._stratum_metrics(y[mask], s[mask], axis="device_type", value=str(val))
            rows.append(
                {
                    "stratum_axis": "device_type",
                    "stratum_value": str(val),
                    **metrics,
                }
            )
        nan_mask = dev.isna().to_numpy()
        if nan_mask.any():
            metrics = self._stratum_metrics(
                y[nan_mask],
                s[nan_mask],
                axis="device_type",
                value=_DEVICE_NULL_LABEL,
            )
            rows.append(
                {
                    "stratum_axis": "device_type",
                    "stratum_value": _DEVICE_NULL_LABEL,
                    **metrics,
                }
            )
        return rows

    def _evaluate_by_identity_coverage(
        self,
        y: np.ndarray[Any, Any],
        s: np.ndarray[Any, Any],
        probe: pd.Series[Any],
    ) -> list[dict[str, Any]]:
        """Two rows: `has_identity` (probe non-null) vs `no_identity` (probe null)."""
        has_mask = probe.notna().to_numpy()
        no_mask = ~has_mask
        has_metrics = self._stratum_metrics(
            y[has_mask], s[has_mask], axis="identity_coverage", value=_HAS_IDENTITY_LABEL
        )
        no_metrics = self._stratum_metrics(
            y[no_mask], s[no_mask], axis="identity_coverage", value=_NO_IDENTITY_LABEL
        )
        return [
            {
                "stratum_axis": "identity_coverage",
                "stratum_value": _HAS_IDENTITY_LABEL,
                **has_metrics,
            },
            {
                "stratum_axis": "identity_coverage",
                "stratum_value": _NO_IDENTITY_LABEL,
                **no_metrics,
            },
        ]

    def _evaluate_by_month(
        self,
        y: np.ndarray[Any, Any],
        s: np.ndarray[Any, Any],
        m: pd.Series[Any],
    ) -> list[dict[str, Any]]:
        """One row per unique month value (sorted ascending)."""
        rows: list[dict[str, Any]] = []
        # Sort to give deterministic ordering; convert each label to
        # str so "5" / "6" / etc. are friendly in the output frame.
        unique_months = sorted(m.dropna().unique().tolist())
        for month_val in unique_months:
            mask = (m == month_val).to_numpy()
            metrics = self._stratum_metrics(y[mask], s[mask], axis="month", value=str(month_val))
            rows.append({"stratum_axis": "month", "stratum_value": str(month_val), **metrics})
        return rows

    # -----------------------------------------------------------------
    # Per-stratum metric workhorse.
    # -----------------------------------------------------------------

    def _stratum_metrics(
        self,
        y_strat: np.ndarray[Any, Any],
        s_strat: np.ndarray[Any, Any],
        *,
        axis: str,
        value: str,
    ) -> dict[str, Any]:
        """Compute n_rows, n_pos, fraud_rate, auc, pr_auc, total_cost, cost_per_txn.

        Single place that handles the three degenerate cases:
            - Empty stratum (n_rows = 0): all metrics NaN; cost = 0.
            - Single-class stratum: AUC / PR-AUC NaN, warning logged;
              cost still computed.
            - Below `min_stratum_size`: AUC / PR-AUC NaN, warning
              logged; cost still computed.

        Args / Returns: see top-level docstring.
        """
        n_rows = int(y_strat.size)
        if n_rows == 0:
            return {
                "n_rows": 0,
                "n_pos": 0,
                "fraud_rate": float("nan"),
                "auc": float("nan"),
                "pr_auc": float("nan"),
                "total_cost": 0.0,
                "cost_per_txn": float("nan"),
            }

        n_pos = int(np.sum(y_strat == 1))
        fraud_rate = float(n_pos / n_rows)

        # Cost is well-defined on any non-empty stratum (including
        # single-class). The 25 sub-frame cost evaluations across all
        # axes are cheap; not worth vectorising at this scale.
        y_pred = (s_strat >= self._threshold).astype(int)
        cost_dict = self._cost_model.compute_cost(y_strat, y_pred)
        total_cost = float(cost_dict["total_cost"])
        cost_per_txn = float(cost_dict["cost_per_txn"])

        # AUC / PR-AUC: requires both classes present AND >=
        # min_stratum_size rows. Below either threshold, return NaN
        # with a warning so the heatmap doesn't display garbage cells.
        n_classes = int(np.unique(y_strat).size)
        if n_classes < _MIN_CLASSES_FOR_AUC:
            _logger.warning(
                "stratified.degenerate_stratum",
                axis=axis,
                value=value,
                reason="single_class",
                n_rows=n_rows,
                n_pos=n_pos,
                n_neg=n_rows - n_pos,
            )
            auc = float("nan")
            pr_auc = float("nan")
        elif n_rows < self._min_stratum_size:
            _logger.warning(
                "stratified.degenerate_stratum",
                axis=axis,
                value=value,
                reason="too_small",
                n_rows=n_rows,
                min_stratum_size=self._min_stratum_size,
            )
            auc = float("nan")
            pr_auc = float("nan")
        else:
            auc = float(roc_auc_score(y_strat, s_strat))
            pr_auc = float(average_precision_score(y_strat, s_strat))

        return {
            "n_rows": n_rows,
            "n_pos": n_pos,
            "fraud_rate": fraud_rate,
            "auc": auc,
            "pr_auc": pr_auc,
            "total_cost": total_cost,
            "cost_per_txn": cost_per_txn,
        }

    # -----------------------------------------------------------------
    # Heatmap helpers.
    # -----------------------------------------------------------------

    @staticmethod
    def _z_score_for_heatmap(
        raw_data: np.ndarray[Any, Any],
        metrics: Sequence[str],
    ) -> np.ndarray[Any, Any]:
        """Per-column z-score normalisation; flip cost-like columns.

        After this transform, positive z-score consistently means
        "this stratum is worse than the cross-stratum mean for this
        metric" regardless of whether the metric is higher-is-better
        (AUC, PR-AUC) or higher-is-worse (cost_per_txn, total_cost).
        """
        z_data = np.full_like(raw_data, fill_value=np.nan, dtype=np.float64)
        for j, metric in enumerate(metrics):
            col = raw_data[:, j]
            finite = col[np.isfinite(col)]
            if finite.size < _MIN_FINITE_FOR_ZSCORE:
                # Not enough finite values to compute std; leave as
                # zeros so the cell colour is neutral.
                z_data[:, j] = np.where(np.isfinite(col), 0.0, np.nan)
                continue
            mean = float(np.nanmean(col))
            std = float(np.nanstd(col))
            if std == 0.0:
                z_data[:, j] = np.where(np.isfinite(col), 0.0, np.nan)
                continue
            z = (col - mean) / std
            # Flip sign for higher-is-better metrics so red = worse
            # uniformly across columns.
            if metric not in _COST_LIKE_METRICS:
                z = -z
            z_data[:, j] = z
        return z_data


__all__ = ["StratifiedEvaluator"]
