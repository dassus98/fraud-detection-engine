"""Economic-cost evaluation: threshold optimisation + sensitivity analysis.

Sprint 4 prompt 4.1: builds the threshold-optimisation surface that
Sprint 4's downstream cost-curve evaluator and Sprint 5's serving stack
read from. Wraps the per-call ``economic_cost`` primitive in
``utils/metrics.py`` (single source of truth for the cost formula) with
sweep + tie-break + sensitivity-grid plumbing.

Business rationale:
    A fraud system's operational metric is dollars, not F1 or AUC. Per
    CLAUDE.md §8 a missed fraud costs ~13× more than a blocked legit
    transaction (`fraud_cost=$450` vs `fp_cost=$35`), which means a
    symmetric metric picks the wrong threshold by construction. This
    module finds the threshold that minimises expected USD cost on
    Model A's calibrated probabilities (3.3.c → 3.3.d), and confirms
    via a ±20% sensitivity grid that the chosen threshold is robust
    against the inevitable uncertainty in the cost-model inputs
    themselves (industry-median chargeback / churn estimates have
    real error bars).

Trade-offs considered:
    - **Stateless class wrapping a stateless primitive.** Calibration's
      `PlattScaler` / `IsotonicCalibrator` carry learned state and
      enforce a `fit / transform` contract; `EconomicCostModel`
      learns nothing — costs are config, the optimum is a derivation
      from data + config. The closest analog is
      `select_calibration_method` (`evaluation/calibration.py:477`),
      not the calibrator classes. No `is_fitted_` flag, no pre-fit
      guard.
    - **Threshold sweep is `linspace(0.01, 0.99, 99)` per spec, not
      adaptive.** Reproducibility (every call walks the same grid)
      and MLflow legibility outweigh the modest waste of evaluating
      obviously-bad thresholds. A bisection / golden-section search
      would converge faster but would surface a different optimum
      across runs (the cost surface isn't strictly convex on real
      data); the fixed grid pins the answer.
    - **Tie-break favours larger τ.** When two thresholds yield
      identical total_cost — typically because they straddle a gap
      between consecutive `y_scores` and produce identical `y_pred`
      — pick the larger τ. Rationale: blocking fewer transactions on
      identical cost is the less-invasive policy. Mirrors
      `select_calibration_method`'s "ties resolve to the
      earlier-listed (so 'none' wins ties — preserves identity over
      needless transformation)" pattern (`calibration.py:548-549`).
      Implemented via stable two-key sort
      (`sort_values(["total_cost", "threshold"], ascending=[True,
      False])`), NOT `idxmin` (which would return the smallest τ on
      ties — opposite of intent).
    - **Sensitivity grid defaults to symmetric ±20% multipliers**
      (CLAUDE.md §8: "decisions are stable under ±20% variation").
      Cartesian product across the three cost axes is 5 × 5 × 5 =
      125 cells × 99 thresholds × N rows — fine in numpy at
      IEEE-CIS scale (val ≈ 83 K rows → ~1B element ops, runs in
      seconds). A larger grid (9 per axis, 729 cells) on a 500 K
      test set would push to ~36B ops; the
      ``economic.sensitivity.grid_size`` log line lets a future
      reviewer notice if a caller blew the budget.
    - **`y_scores ∈ [0, 1]` validation raises, not clips.** Model A
      passes its `Calibrator.transform` output to this module; the
      calibrator's contract guarantees `[0, 1]` (see
      `calibration.py:441-449`). Silent clipping would mask an
      upstream bug — better to fail loudly. `tn_cost` defaults to 0
      and is not in the sensitivity grid (no Settings analogue; TN
      is zero by convention in fraud-ML).
    - **Pandas DataFrame return for cost curve and sensitivity
      grid.** This is the first ``evaluation/`` module to take a
      pandas dependency — `calibration.py` is pandas-free by
      contrast. The data is naturally tabular, the consumer (Sprint
      5's reporter) wants `to_html` / `to_csv` for free, and the
      column-name contract is more legible than a `(rows, cols)`
      ndarray. Justified the dependency.

Cross-references:
    - `src/fraud_engine/utils/metrics.py:68-161` — the wrapped primitive
      ``economic_cost``; single source of truth for the cost formula.
    - `src/fraud_engine/evaluation/calibration.py:477-556` — sibling
      sweep-and-pick pattern; mirrors stable tie-break + structured
      logging.
    - `src/fraud_engine/config/settings.py:88-115` — cost defaults
      with `Field(ge=0.0)` validation; `get_settings()` is
      `lru_cache`-wrapped.
    - `CLAUDE.md` §8 — business-logic constants table + the ±20%
      sensitivity stability rule.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product
from typing import Any, Final

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from fraud_engine.config.settings import get_settings
from fraud_engine.utils.logging import get_logger
from fraud_engine.utils.metrics import economic_cost

_logger = get_logger(__name__)

# ---------------------------------------------------------------------
# Module constants.
# ---------------------------------------------------------------------

# Spec-pinned threshold sweep: linspace(0.01, 0.99, 99). Endpoints
# deliberately excluded — τ=0.0 flags every row (no useful info; max
# FP) and τ=1.0 flags none (zero TP), so sweeping them adds no signal.
_DEFAULT_N_THRESHOLDS: Final[int] = 99
_DEFAULT_THRESHOLD_LOWER: Final[float] = 0.01
_DEFAULT_THRESHOLD_UPPER: Final[float] = 0.99

# Default sensitivity grid: ±20% symmetric per axis (CLAUDE.md §8).
# Five points let the optimum cluster shape itself emerge; three
# would only show the centre vs the rails.
_DEFAULT_SENSITIVITY_MULTIPLIERS: Final[tuple[float, ...]] = (0.80, 0.90, 1.00, 1.10, 1.20)

# Probability bounds for the `y_scores` validation guard.
_PROB_LOWER: Final[float] = 0.0
_PROB_UPPER: Final[float] = 1.0

# Cost-curve column order. Mirrors `economic_cost`'s return-dict keys
# (with `threshold` prepended) so a row of the curve is the dict at
# that threshold — no rename layer.
_COST_CURVE_COLUMNS: Final[tuple[str, ...]] = (
    "threshold",
    "total_cost",
    "cost_per_txn",
    "fn",
    "fp",
    "tp",
    "tn",
)

# Sensitivity-grid column order. Costs first so a reader sees the
# scenario before the optimum it produced.
_SENSITIVITY_COLUMNS: Final[tuple[str, ...]] = (
    "fraud_cost",
    "fp_cost",
    "tp_cost",
    "optimal_threshold",
    "optimal_total_cost",
    "optimal_cost_per_txn",
)

# Cost axes the sensitivity grid sweeps. `tn_cost` excluded — it has
# no Settings field and is zero by convention.
_SENSITIVITY_AXES: Final[tuple[str, ...]] = ("fraud_cost", "fp_cost", "tp_cost")


# ---------------------------------------------------------------------
# Validation helpers.
# ---------------------------------------------------------------------


def _validate_costs(
    fraud_cost: float,
    fp_cost: float,
    tp_cost: float,
    tn_cost: float,
) -> None:
    """Enforce the same `ge=0.0` contract Settings does on its fields.

    Per-call cost overrides bypass Pydantic's validation, so we
    re-check here. Negative costs would flip the sign of the
    expected-cost objective and silently invert the threshold sweep.
    """
    for name, val in (
        ("fraud_cost", fraud_cost),
        ("fp_cost", fp_cost),
        ("tp_cost", tp_cost),
        ("tn_cost", tn_cost),
    ):
        if val < 0:
            raise ValueError(f"EconomicCostModel: {name} must be >= 0, got {val}")


def _validate_score_arrays(
    y_true: ArrayLike,
    y_scores: ArrayLike,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Coerce inputs to 1-D numpy arrays; enforce shape + range contracts.

    Returns:
        (y_true_arr, y_scores_arr) both 1-D, same length.

    Raises:
        ValueError: If shapes mismatch, or if y_scores has any value
            outside [0, 1] (calibration contract guarantees the range
            for Model A; failing loud surfaces upstream bugs).
    """
    y_true_arr = np.asarray(y_true).ravel()
    y_scores_arr = np.asarray(y_scores, dtype=np.float64).ravel()
    if y_true_arr.shape != y_scores_arr.shape:
        raise ValueError(
            f"EconomicCostModel: y_true and y_scores shape mismatch — "
            f"{y_true_arr.shape} vs {y_scores_arr.shape}"
        )
    if y_scores_arr.size > 0:
        score_min = float(y_scores_arr.min())
        score_max = float(y_scores_arr.max())
        if score_min < _PROB_LOWER or score_max > _PROB_UPPER:
            raise ValueError(
                f"EconomicCostModel: y_scores must be in [0, 1] "
                f"(calibration contract); got min={score_min}, max={score_max}"
            )
    return y_true_arr, y_scores_arr


# ---------------------------------------------------------------------
# Inner sweep (private free function — used by both `optimize_threshold`
# and `sensitivity_analysis` to avoid object churn per cell).
# ---------------------------------------------------------------------


def _sweep_thresholds(  # noqa: PLR0913 — the four cost parameters plus three array inputs are the business contract; folding into a config dict would obscure call-site semantics. Mirrors `metrics.py:68` rationale.
    y_true: np.ndarray[Any, Any],
    y_scores: np.ndarray[Any, Any],
    thresholds: np.ndarray[Any, Any],
    fraud_cost: float,
    fp_cost: float,
    tp_cost: float,
    tn_cost: float,
) -> tuple[float, pd.DataFrame]:
    """Sweep thresholds; return (optimal_τ, cost_curve DataFrame).

    The cost curve is sorted ascending by threshold. The optimum is
    chosen by stable two-key sort: lowest total_cost first, then
    largest threshold among ties (block-fewer-transactions tie-break).
    """
    rows: list[dict[str, float]] = []
    for tau in thresholds:
        y_pred = (y_scores >= tau).astype(int)
        cost_dict = economic_cost(
            y_true,
            y_pred,
            fraud_cost=fraud_cost,
            fp_cost=fp_cost,
            tp_cost=tp_cost,
            tn_cost=tn_cost,
        )
        rows.append({"threshold": float(tau), **cost_dict})
    curve = pd.DataFrame(rows, columns=list(_COST_CURVE_COLUMNS))
    curve = curve.sort_values("threshold").reset_index(drop=True)
    # Tie-break: largest τ on equal cost. `sort_values` is stable;
    # the (ascending=True, descending=False) tuple does the right thing.
    sorted_for_pick = curve.sort_values(
        ["total_cost", "threshold"],
        ascending=[True, False],
    )
    optimal_tau = float(sorted_for_pick.iloc[0]["threshold"])
    return optimal_tau, curve


# ---------------------------------------------------------------------
# Public class.
# ---------------------------------------------------------------------


class EconomicCostModel:
    """Stateless wrapper around `economic_cost` with sweep + sensitivity.

    Public API:
        - ``compute_cost(y_true, y_pred)`` — forwards to
          ``utils.metrics.economic_cost`` with the configured costs.
        - ``optimize_threshold(y_true, y_scores, thresholds=None)`` —
          sweep thresholds, return ``(optimal_τ, cost_curve)``.
        - ``sensitivity_analysis(y_true, y_scores, cost_ranges=None,
          thresholds=None)`` — sweep cost combinations + thresholds,
          return a per-scenario DataFrame.
        - ``costs`` property — snapshot of the configured costs for
          logging / manifest persistence.

    The class snapshots costs at construction time. Changing
    ``Settings`` after instantiation does NOT propagate — re-instantiate
    if you need new defaults.
    """

    def __init__(
        self,
        fraud_cost: float | None = None,
        fp_cost: float | None = None,
        tp_cost: float | None = None,
        tn_cost: float = 0.0,
    ) -> None:
        """Snapshot costs from kwargs or Settings.

        Args:
            fraud_cost: USD cost of a false negative (missed fraud).
                If None, resolved from ``Settings.fraud_cost_usd``.
            fp_cost: USD cost of a false positive (blocked legit txn).
                If None, resolved from ``Settings.fp_cost_usd``.
            tp_cost: USD cost of a true positive (analyst review).
                If None, resolved from ``Settings.tp_cost_usd``.
            tn_cost: USD cost of a true negative. Defaults to 0.0
                (no Settings field; TN cost is zero by convention).

        Raises:
            ValueError: If any cost is negative (mirrors Settings'
                ``Field(ge=0.0)`` contract for per-call overrides).
        """
        settings = get_settings()
        resolved_fraud = settings.fraud_cost_usd if fraud_cost is None else fraud_cost
        resolved_fp = settings.fp_cost_usd if fp_cost is None else fp_cost
        resolved_tp = settings.tp_cost_usd if tp_cost is None else tp_cost
        _validate_costs(resolved_fraud, resolved_fp, resolved_tp, tn_cost)
        self._fraud_cost: float = float(resolved_fraud)
        self._fp_cost: float = float(resolved_fp)
        self._tp_cost: float = float(resolved_tp)
        self._tn_cost: float = float(tn_cost)

    @property
    def costs(self) -> dict[str, float]:
        """Return a snapshot of the configured costs.

        Useful for logging the model's effective config alongside
        results, or pinning into a manifest sidecar so a downstream
        consumer can verify which costs produced a given optimum.
        """
        return {
            "fraud_cost": self._fraud_cost,
            "fp_cost": self._fp_cost,
            "tp_cost": self._tp_cost,
            "tn_cost": self._tn_cost,
        }

    # -----------------------------------------------------------------
    # compute_cost.
    # -----------------------------------------------------------------

    def compute_cost(
        self,
        y_true: ArrayLike,
        y_pred: ArrayLike,
    ) -> dict[str, float]:
        """Forward to ``utils.metrics.economic_cost`` with stored costs.

        Args:
            y_true: Ground-truth binary labels in {0, 1}.
            y_pred: Predicted binary labels in {0, 1}. Threshold
                probabilities BEFORE calling this method.

        Returns:
            Dict with keys ``total_cost``, ``cost_per_txn``, ``fn``,
            ``fp``, ``tp``, ``tn`` (per the wrapped primitive).
        """
        return economic_cost(
            y_true,
            y_pred,
            fraud_cost=self._fraud_cost,
            fp_cost=self._fp_cost,
            tp_cost=self._tp_cost,
            tn_cost=self._tn_cost,
        )

    # -----------------------------------------------------------------
    # optimize_threshold.
    # -----------------------------------------------------------------

    def optimize_threshold(
        self,
        y_true: ArrayLike,
        y_scores: ArrayLike,
        thresholds: NDArray[np.floating[Any]] | Sequence[float] | None = None,
    ) -> tuple[float, pd.DataFrame]:
        """Sweep thresholds; return (optimal_τ, cost_curve).

        Args:
            y_true: Ground-truth binary labels in {0, 1}.
            y_scores: Predicted probabilities in [0, 1] (calibrated
                output from Model A's ``Calibrator.transform``).
            thresholds: Custom threshold grid. If None, uses
                ``np.linspace(0.01, 0.99, 99)`` per spec.

        Returns:
            Tuple of:
                - ``optimal_τ``: threshold minimising ``total_cost``.
                  Tie-break favours the larger τ (block fewer txns).
                - ``cost_curve``: DataFrame with columns
                  ``threshold, total_cost, cost_per_txn, fn, fp, tp,
                  tn``, sorted ascending by ``threshold``.

        Raises:
            ValueError: If ``y_true`` and ``y_scores`` shapes mismatch,
                or if ``y_scores`` contains values outside [0, 1].
        """
        y_true_arr, y_scores_arr = _validate_score_arrays(y_true, y_scores)
        if thresholds is None:
            thr = np.linspace(
                _DEFAULT_THRESHOLD_LOWER,
                _DEFAULT_THRESHOLD_UPPER,
                _DEFAULT_N_THRESHOLDS,
            )
        else:
            thr = np.asarray(thresholds, dtype=np.float64).ravel()
        optimal_tau, curve = _sweep_thresholds(
            y_true_arr,
            y_scores_arr,
            thr,
            self._fraud_cost,
            self._fp_cost,
            self._tp_cost,
            self._tn_cost,
        )
        _logger.info(
            "economic.optimize.done",
            optimal_threshold=optimal_tau,
            n_thresholds=int(len(thr)),
            n_rows=int(len(y_true_arr)),
            costs=self.costs,
        )
        return optimal_tau, curve

    # -----------------------------------------------------------------
    # sensitivity_analysis.
    # -----------------------------------------------------------------

    def sensitivity_analysis(
        self,
        y_true: ArrayLike,
        y_scores: ArrayLike,
        cost_ranges: Mapping[str, Sequence[float]] | None = None,
        thresholds: NDArray[np.floating[Any]] | Sequence[float] | None = None,
    ) -> pd.DataFrame:
        """Sweep cost combinations × thresholds; return per-scenario optima.

        For each combination of (fraud_cost, fp_cost, tp_cost) in the
        cartesian product of ``cost_ranges``, run a threshold sweep
        and record the optimum. Confirms that the chosen threshold is
        stable under cost-input uncertainty (CLAUDE.md §8: ±20%).

        Args:
            y_true: Ground-truth binary labels in {0, 1}.
            y_scores: Predicted probabilities in [0, 1].
            cost_ranges: Per-axis cost grids. Keys must be a subset of
                ``{"fraud_cost", "fp_cost", "tp_cost"}``. Missing axes
                fall back to a single-value range at the configured
                cost. If None, defaults to ±20% symmetric grid
                (``[0.8, 0.9, 1.0, 1.1, 1.2]`` × the configured cost
                per axis).
            thresholds: Custom threshold grid (same default as
                ``optimize_threshold``).

        Returns:
            DataFrame with columns ``fraud_cost, fp_cost, tp_cost,
            optimal_threshold, optimal_total_cost,
            optimal_cost_per_txn``. One row per cost combination.

        Raises:
            ValueError: If ``cost_ranges`` contains unknown axis keys,
                or if ``y_true`` / ``y_scores`` fail validation, or if
                any range value is negative.
        """
        y_true_arr, y_scores_arr = _validate_score_arrays(y_true, y_scores)

        # Validate axes + build the full grid (single-value fallback
        # for any axis the caller didn't pass).
        if cost_ranges is None:
            full_ranges: dict[str, list[float]] = {
                "fraud_cost": [self._fraud_cost * m for m in _DEFAULT_SENSITIVITY_MULTIPLIERS],
                "fp_cost": [self._fp_cost * m for m in _DEFAULT_SENSITIVITY_MULTIPLIERS],
                "tp_cost": [self._tp_cost * m for m in _DEFAULT_SENSITIVITY_MULTIPLIERS],
            }
        else:
            unknown = set(cost_ranges) - set(_SENSITIVITY_AXES)
            if unknown:
                raise ValueError(
                    f"sensitivity_analysis: unknown cost-range axes "
                    f"{sorted(unknown)}; allowed: {list(_SENSITIVITY_AXES)}"
                )
            full_ranges = {}
            for axis in _SENSITIVITY_AXES:
                if axis in cost_ranges:
                    values = [float(v) for v in cost_ranges[axis]]
                    if any(v < 0 for v in values):
                        raise ValueError(f"sensitivity_analysis: {axis} contains negative value")
                    full_ranges[axis] = values
                else:
                    full_ranges[axis] = [getattr(self, f"_{axis}")]

        if thresholds is None:
            thr = np.linspace(
                _DEFAULT_THRESHOLD_LOWER,
                _DEFAULT_THRESHOLD_UPPER,
                _DEFAULT_N_THRESHOLDS,
            )
        else:
            thr = np.asarray(thresholds, dtype=np.float64).ravel()

        n_cells = (
            len(full_ranges["fraud_cost"])
            * len(full_ranges["fp_cost"])
            * len(full_ranges["tp_cost"])
        )
        _logger.info(
            "economic.sensitivity.grid_size",
            n_cells=n_cells,
            n_thresholds=int(len(thr)),
            n_rows=int(len(y_true_arr)),
        )

        rows: list[dict[str, float]] = []
        for fc, fpc, tpc in product(
            full_ranges["fraud_cost"],
            full_ranges["fp_cost"],
            full_ranges["tp_cost"],
        ):
            opt_tau, curve = _sweep_thresholds(
                y_true_arr,
                y_scores_arr,
                thr,
                fc,
                fpc,
                tpc,
                self._tn_cost,
            )
            # Recover the optimum row from the curve (no extra sort —
            # pandas filter is fine for a 99-row frame).
            opt_row = curve.loc[curve["threshold"] == opt_tau].iloc[0]
            rows.append(
                {
                    "fraud_cost": float(fc),
                    "fp_cost": float(fpc),
                    "tp_cost": float(tpc),
                    "optimal_threshold": opt_tau,
                    "optimal_total_cost": float(opt_row["total_cost"]),
                    "optimal_cost_per_txn": float(opt_row["cost_per_txn"]),
                }
            )
        return pd.DataFrame(rows, columns=list(_SENSITIVITY_COLUMNS))


__all__ = ["EconomicCostModel"]
