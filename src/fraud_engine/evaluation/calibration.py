"""Probability calibration: Platt scaling, isotonic regression, diagnostics.

Sprint 3 prompt 3.3.c: builds the calibration toolkit Sprint 4's
economic-cost evaluator and threshold-optimisation pass need.
LightGBM-on-fraud predictions are well-ranked (AUC ≈ 0.93+) but
miscalibrated by default — gradient-boosted models tend toward
overconfident extremes, so a "0.7 fraud probability" doesn't
correspond to ~70% empirical fraud rate. Sprint 4's threshold
sweep optimises against a probability-weighted cost surface; running
it on uncalibrated probabilities distorts the cost integral and
picks the wrong threshold.

This module is the cure: a reliability-diagram diagnostic, three
metrics (Brier score, log loss, ECE), two calibration methods
(Platt scaling, isotonic regression) wrapped behind a uniform
`fit(y, p)` / `transform(p)` contract, and an automatic chooser
(`select_calibration_method`) that picks whichever method minimises
log loss on a held-out fold (or returns "none" if neither beats the
uncalibrated baseline).

Business rationale:
    A fraud system that operates at, say, a 0.5 decision threshold
    on uncalibrated probabilities silently optimises against the
    wrong cost ratio. Calibrating once on a held-out validation set
    realigns probabilities to their empirical frequencies, so the
    threshold-sweep (Sprint 4) and the cost-curve (also Sprint 4)
    operate on a faithful representation of the model's risk
    distribution. This is the difference between a deployed model
    that minimises USD cost and one that minimises an internal
    objective the company doesn't pay for.

Trade-offs considered:
    - **Platt scaling vs isotonic regression.** Platt is a
      2-parameter logistic on the raw probability — fast, robust at
      small sample sizes, but assumes a sigmoid-shaped miscalibration
      curve. Isotonic is non-parametric monotonic — captures
      arbitrary monotone distortions but overfits at small samples.
      `select_calibration_method` empirically picks the winner on a
      held-out split rather than committing to one a priori.
    - **`select_calibration_method` returns "none" if neither wins.**
      An uncalibrated probability that's already near-perfect should
      not be perturbed; calibration on a near-perfect predictor adds
      noise. The "none" branch returns an `IdentityCalibrator` so
      downstream `transform(p)` calls work uniformly.
    - **Honest single-split holdout, not cross-validated.** Splits
      `(val_y, val_p)` 80/20 — fit on 80% calibration fold; score
      log_loss on 20% holdout. Single split is cheaper than
      `CalibratedClassifierCV`-style k-fold CV and sufficient at
      Sprint 3's val size (~83k IEEE-CIS rows × 0.20 = 16.6k
      holdout — comfortably enough to discriminate Platt vs isotonic
      vs none). Sprint 4 may revisit if the per-fold variance
      becomes operationally significant.
    - **Platt fit uses `LogisticRegression(C=1e10)`.** scikit-learn's
      default `C=1.0` applies an L2 prior that biases the slope
      toward 0; setting `C=1e10` effectively disables the
      regulariser, recovering the maximum-likelihood Platt fit
      (Platt 1999). The L2 variant is a stability hack for very
      small samples (<100); we don't need it at the scale this
      project operates.
    - **Isotonic uses `out_of_bounds="clip"`.** Test-time inputs that
      fall outside the [min, max] of the fitting range get clipped
      to the boundary. The alternative (`"raise"`) would crash on
      any val/test probability outside the calibration range; the
      `"nan"` mode would propagate NaNs that downstream code would
      have to handle. Clipping is the operational default.
    - **`log_loss` and `brier_score` live here, not in
      `utils/metrics.py`.** That file's scope is operational
      decision metrics (economic_cost, PR@K, recall@FPR, PSI);
      calibration metrics are conceptually adjacent but distinct.
      Keeping them here pins the calibration toolkit's surface area
      to one module.

Cross-references:
    - `src/fraud_engine/utils/metrics.py` — operational decision
      metrics (economic_cost, PR@K, recall@FPR, PSI).
    - `src/fraud_engine/models/lightgbm_model.py:240-410` — the
      pre-fit-raises and fitted-state-attribute pattern this
      module's calibrator classes mirror.
"""

from __future__ import annotations

from typing import Any, Final, Self

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss as _sklearn_log_loss
from sklearn.model_selection import train_test_split

from fraud_engine.utils.logging import get_logger

_logger = get_logger(__name__)

# ---------------------------------------------------------------------
# Module constants.
# ---------------------------------------------------------------------

# Default reliability-diagram bin count. 10 is the field standard:
# fine enough to surface miscalibration, coarse enough that each bin
# has enough samples to estimate the empirical fraud rate stably.
_DEFAULT_N_BINS: Final[int] = 10

# Default eps clip for `log_loss`. sklearn deprecated the `eps` kwarg
# in 1.5+ in favour of caller-side clipping; we pin our own default
# so behaviour is stable across sklearn minor versions.
_DEFAULT_LOG_LOSS_EPS: Final[float] = 1e-15

# `C` parameter for Platt scaling's logistic regression. 1e10
# effectively disables the L2 regulariser so `LogisticRegression`
# recovers the maximum-likelihood Platt fit (Platt 1999).
_PLATT_C: Final[float] = 1e10

# Default holdout fraction for `select_calibration_method`. 80/20 is
# the standard "honest split" ratio — large enough calibration fold
# (80%) to fit each method, large enough holdout (20%) for
# discriminating their out-of-sample log loss.
_DEFAULT_HOLDOUT_FRACTION: Final[float] = 0.20

# Probability bound constants. Mirrors sklearn's calibration
# clipping convention.
_PROB_LOWER: Final[float] = 0.0
_PROB_UPPER: Final[float] = 1.0


# ---------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------


def brier_score(y_true: np.ndarray[Any, Any], y_pred_proba: np.ndarray[Any, Any]) -> float:
    """Brier score = mean squared error between probabilities and labels.

    Lower is better; perfect probabilities = 0; anti-perfect = 1.
    Quadratic in error, so it penalises confident mistakes more
    aggressively than log loss.

    Args:
        y_true: 1-D array of binary labels (0 / 1).
        y_pred_proba: 1-D array of probabilities ∈ [0, 1].

    Returns:
        Brier score (float).
    """
    return float(brier_score_loss(y_true, y_pred_proba))


def log_loss(
    y_true: np.ndarray[Any, Any],
    y_pred_proba: np.ndarray[Any, Any],
    eps: float = _DEFAULT_LOG_LOSS_EPS,
) -> float:
    """Binary cross-entropy with explicit eps clipping.

    Wraps `sklearn.metrics.log_loss` with caller-side clipping (the
    sklearn `eps` kwarg was deprecated in 1.5+). Clipping prevents
    `inf` when a perfect-confidence prediction lands on the wrong
    side of a label.

    Args:
        y_true: 1-D array of binary labels (0 / 1).
        y_pred_proba: 1-D array of probabilities ∈ [0, 1].
        eps: Clip range; predictions clamped to `[eps, 1 - eps]`.

    Returns:
        Log-loss (float).
    """
    clipped = np.clip(np.asarray(y_pred_proba, dtype=np.float64), eps, 1.0 - eps)
    return float(_sklearn_log_loss(y_true, clipped, labels=[0, 1]))


def expected_calibration_error(
    y_true: np.ndarray[Any, Any],
    y_pred_proba: np.ndarray[Any, Any],
    n_bins: int = _DEFAULT_N_BINS,
) -> float:
    """Expected calibration error.

    Bins predictions into `n_bins` equal-width buckets; for each bin
    computes `|bin_avg_pred - bin_observed_rate|`; weights by bin
    size and averages. ECE is the standard scalar summary of a
    reliability diagram's miscalibration area.

    Args:
        y_true: 1-D array of binary labels (0 / 1).
        y_pred_proba: 1-D array of probabilities ∈ [0, 1].
        n_bins: Number of equal-width probability bins. Default 10.

    Returns:
        ECE (float ≥ 0). Perfect calibration → 0; pathological →
        approaches 1.
    """
    y_true_arr = np.asarray(y_true).ravel()
    y_pred_arr = np.asarray(y_pred_proba).ravel()
    n_total = len(y_pred_arr)
    if n_total == 0:
        return 0.0

    bin_edges = np.linspace(_PROB_LOWER, _PROB_UPPER, n_bins + 1)
    # Use np.digitize with right=True to match `pd.cut` half-open
    # convention (a, b]; subtract 1 for 0-indexed bins.
    bin_idx = np.clip(np.digitize(y_pred_arr, bin_edges, right=True) - 1, 0, n_bins - 1)

    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        n_bin = int(mask.sum())
        if n_bin == 0:
            continue
        bin_pred = float(y_pred_arr[mask].mean())
        bin_obs = float(y_true_arr[mask].mean())
        ece += (n_bin / n_total) * abs(bin_pred - bin_obs)
    return float(ece)


# ---------------------------------------------------------------------
# Reliability diagram.
# ---------------------------------------------------------------------


def reliability_diagram(
    y_true: np.ndarray[Any, Any],
    y_pred_proba: np.ndarray[Any, Any],
    n_bins: int = _DEFAULT_N_BINS,
    ax: Axes | None = None,
) -> Axes:
    """Plot a calibration / reliability diagram and return the Axes.

    For each non-empty bin: x = mean predicted probability in the
    bin, y = empirical positive rate. A perfectly-calibrated model
    falls on the diagonal `y = x`; deviations above the diagonal
    indicate underconfidence, below indicate overconfidence.

    Caller is responsible for `fig.savefig(...)`. The function
    returns the `Axes` so further customisation (titles, legends,
    multi-curve overlays) is possible.

    Args:
        y_true: 1-D array of binary labels (0 / 1).
        y_pred_proba: 1-D array of probabilities ∈ [0, 1].
        n_bins: Number of equal-width probability bins. Default 10.
        ax: Optional `matplotlib.axes.Axes` to plot on. If `None`,
            creates a new `(fig, ax)` via `plt.subplots()`.

    Returns:
        The `Axes` instance carrying the plot.
    """
    y_true_arr = np.asarray(y_true).ravel()
    y_pred_arr = np.asarray(y_pred_proba).ravel()

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    # Per-bin mean prediction + observed positive rate. Empty bins
    # are skipped (no marker plotted) so the curve doesn't drop to
    # zero in sparse regions.
    bin_edges = np.linspace(_PROB_LOWER, _PROB_UPPER, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_pred_arr, bin_edges, right=True) - 1, 0, n_bins - 1)
    xs: list[float] = []
    ys: list[float] = []
    sizes: list[int] = []
    for b in range(n_bins):
        mask = bin_idx == b
        n_bin = int(mask.sum())
        if n_bin == 0:
            continue
        xs.append(float(y_pred_arr[mask].mean()))
        ys.append(float(y_true_arr[mask].mean()))
        sizes.append(n_bin)

    # Perfect-calibration diagonal first so the actual curve plots on top.
    ax.plot(
        [_PROB_LOWER, _PROB_UPPER],
        [_PROB_LOWER, _PROB_UPPER],
        linestyle=":",
        color="gray",
        label="perfect calibration",
    )
    ax.plot(xs, ys, marker="o", color="#c14242", label="observed")

    ece = expected_calibration_error(y_true_arr, y_pred_arr, n_bins=n_bins)
    ax.text(
        0.02,
        0.95,
        f"ECE = {ece:.4f}",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )

    ax.set_xlim(_PROB_LOWER, _PROB_UPPER)
    ax.set_ylim(_PROB_LOWER, _PROB_UPPER)
    ax.set_xlabel("mean predicted probability (per bin)")
    ax.set_ylabel("observed positive rate (per bin)")
    ax.set_title("Reliability diagram")
    ax.legend(loc="lower right")
    return ax


# ---------------------------------------------------------------------
# Calibrators.
# ---------------------------------------------------------------------


class _IdentityCalibrator:
    """Pass-through calibrator. Returned by `select_calibration_method`
    when neither Platt nor isotonic beats the uncalibrated baseline.

    `fit` is a no-op; `transform(p) -> p`. Implemented so downstream
    code can call `.transform(...)` uniformly regardless of the
    chosen method.
    """

    def fit(
        self,
        y_true: np.ndarray[Any, Any],  # noqa: ARG002 — uniform interface
        y_pred_proba: np.ndarray[Any, Any],  # noqa: ARG002
    ) -> Self:
        """Return self (no fitting required for identity)."""
        return self

    def transform(self, y_pred_proba: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Pass-through; returns the input as a numpy array."""
        return np.asarray(y_pred_proba, dtype=np.float64).ravel()


class PlattScaler:
    """Platt scaling: 2-parameter logistic regression on the raw probability.

    Fits `P_calibrated = sigmoid(a * p + b)` against `y_true` via
    maximum likelihood (`LogisticRegression(C=1e10)`). Sigmoid-shaped
    miscalibration assumption — works well when the model's score
    is on the right manifold but the slope is wrong.

    Attributes:
        is_fitted_ (bool): True after `fit`.
        _lr (LogisticRegression): Fitted underlying classifier.
    """

    def __init__(self) -> None:
        """Construct an unfitted Platt scaler."""
        self.is_fitted_: bool = False
        self._lr: LogisticRegression | None = None

    def fit(
        self,
        y_true: np.ndarray[Any, Any],
        y_pred_proba: np.ndarray[Any, Any],
    ) -> Self:
        """Fit the underlying logistic regression to `(p, y)`.

        Args:
            y_true: 1-D array of binary labels (0 / 1).
            y_pred_proba: 1-D array of probabilities ∈ [0, 1].

        Returns:
            self (fitted in place).
        """
        x = np.asarray(y_pred_proba, dtype=np.float64).ravel().reshape(-1, 1)
        y = np.asarray(y_true).ravel()
        lr = LogisticRegression(C=_PLATT_C, solver="lbfgs", max_iter=1000)
        lr.fit(x, y)
        self._lr = lr
        self.is_fitted_ = True
        return self

    def transform(self, y_pred_proba: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Apply Platt scaling; return calibrated probabilities ∈ [0, 1].

        Args:
            y_pred_proba: 1-D array of uncalibrated probabilities.

        Returns:
            1-D array of calibrated probabilities, same shape.

        Raises:
            AttributeError: If called before `fit`.
        """
        if not self.is_fitted_ or self._lr is None:
            raise AttributeError("PlattScaler must be fit before transform")
        x = np.asarray(y_pred_proba, dtype=np.float64).ravel().reshape(-1, 1)
        return np.asarray(self._lr.predict_proba(x)[:, 1], dtype=np.float64)


def fit_platt_scaling(
    y_true: np.ndarray[Any, Any], y_pred_proba: np.ndarray[Any, Any]
) -> PlattScaler:
    """Convenience factory: instantiate + fit a `PlattScaler` in one call."""
    return PlattScaler().fit(y_true, y_pred_proba)


class IsotonicCalibrator:
    """Isotonic-regression calibrator: non-parametric monotonic mapping.

    Fits a piecewise-constant non-decreasing map from `p` to
    empirical fraud rate. More flexible than Platt — captures any
    monotonic miscalibration shape — but overfits at small sample
    sizes (the staircase has many degrees of freedom).

    Out-of-bounds inputs at `transform` time get clipped to the
    fitting range (`out_of_bounds="clip"`); paired with `y_min=0`
    and `y_max=1` to guarantee valid probabilities even at the
    extremes.

    Attributes:
        is_fitted_ (bool): True after `fit`.
        _iso (IsotonicRegression): Fitted underlying regressor.
    """

    def __init__(self) -> None:
        """Construct an unfitted isotonic calibrator."""
        self.is_fitted_: bool = False
        self._iso: IsotonicRegression | None = None

    def fit(
        self,
        y_true: np.ndarray[Any, Any],
        y_pred_proba: np.ndarray[Any, Any],
    ) -> Self:
        """Fit the underlying isotonic regression to `(p, y)`.

        Args:
            y_true: 1-D array of binary labels (0 / 1).
            y_pred_proba: 1-D array of probabilities ∈ [0, 1].

        Returns:
            self (fitted in place).
        """
        iso = IsotonicRegression(out_of_bounds="clip", y_min=_PROB_LOWER, y_max=_PROB_UPPER)
        iso.fit(
            np.asarray(y_pred_proba, dtype=np.float64).ravel(),
            np.asarray(y_true, dtype=np.float64).ravel(),
        )
        self._iso = iso
        self.is_fitted_ = True
        return self

    def transform(self, y_pred_proba: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Apply isotonic calibration; return calibrated probabilities.

        Args:
            y_pred_proba: 1-D array of uncalibrated probabilities.

        Returns:
            1-D array of calibrated probabilities, same shape.
            Out-of-range inputs are clipped to `[0, 1]`.

        Raises:
            AttributeError: If called before `fit`.
        """
        if not self.is_fitted_ or self._iso is None:
            raise AttributeError("IsotonicCalibrator must be fit before transform")
        x = np.asarray(y_pred_proba, dtype=np.float64).ravel()
        return np.asarray(self._iso.transform(x), dtype=np.float64)


def fit_isotonic_regression(
    y_true: np.ndarray[Any, Any], y_pred_proba: np.ndarray[Any, Any]
) -> IsotonicCalibrator:
    """Convenience factory: instantiate + fit an `IsotonicCalibrator`."""
    return IsotonicCalibrator().fit(y_true, y_pred_proba)


# ---------------------------------------------------------------------
# Method selection.
# ---------------------------------------------------------------------

# Type alias for any callable carrying a `transform(p) -> p_calib`
# attribute; covers `PlattScaler`, `IsotonicCalibrator`, and
# `_IdentityCalibrator`.
Calibrator = PlattScaler | IsotonicCalibrator | _IdentityCalibrator


def select_calibration_method(
    val_y: np.ndarray[Any, Any],
    val_p: np.ndarray[Any, Any],
    *,
    holdout_fraction: float = _DEFAULT_HOLDOUT_FRACTION,
    random_state: int | None = None,
) -> tuple[str, Calibrator]:
    """Pick whichever of {none, Platt, isotonic} minimises holdout log loss.

    Procedure:
        1. Stratified-split `(val_y, val_p)` into a calibration fold
           (80%) and a holdout fold (20%).
        2. Compute the uncalibrated holdout log_loss (the baseline).
        3. Fit `PlattScaler` and `IsotonicCalibrator` on the
           calibration fold; score both on the holdout.
        4. Pick the lowest log_loss. If neither beats the baseline,
           return `("none", _IdentityCalibrator())`.
        5. Otherwise refit the winning method on the FULL
           `(val_y, val_p)` and return `(name, calibrator)`.

    The honest single-split protects against the leak where fitting
    a non-parametric isotonic on `(val_y, val_p)` and evaluating its
    log_loss on the same `(val_y, val_p)` would always show a tiny
    win (it has memorised the labels).

    Args:
        val_y: 1-D array of binary labels (0 / 1) on the validation set.
        val_p: 1-D array of uncalibrated probabilities on the same set.
        holdout_fraction: Fraction of `(val_y, val_p)` reserved for
            scoring. Default 0.20.
        random_state: Seed for the train/test split. If `None`, the
            split is non-deterministic.

    Returns:
        ``(method_name, fitted_calibrator)`` where `method_name` is
        one of ``"platt"``, ``"isotonic"``, ``"none"``. The returned
        calibrator implements `transform(p) -> np.ndarray`.
    """
    val_y_arr = np.asarray(val_y).ravel()
    val_p_arr = np.asarray(val_p, dtype=np.float64).ravel()

    p_train, p_holdout, y_train, y_holdout = train_test_split(
        val_p_arr,
        val_y_arr,
        test_size=holdout_fraction,
        random_state=random_state,
        stratify=val_y_arr,
    )

    # Baseline: holdout log_loss with no calibration.
    baseline = log_loss(y_holdout, p_holdout)

    # Candidates: Platt and isotonic, fit on calibration fold only.
    platt = PlattScaler().fit(y_train, p_train)
    platt_loss = log_loss(y_holdout, platt.transform(p_holdout))

    iso = IsotonicCalibrator().fit(y_train, p_train)
    iso_loss = log_loss(y_holdout, iso.transform(p_holdout))

    _logger.info(
        "calibration.select_method.scores",
        baseline=baseline,
        platt=platt_loss,
        isotonic=iso_loss,
    )

    candidates: list[tuple[str, float]] = [
        ("none", baseline),
        ("platt", platt_loss),
        ("isotonic", iso_loss),
    ]
    # Stable ordering: ties resolve to the earlier-listed (so "none"
    # wins ties — preserves identity over needless transformation).
    best_name, best_loss = min(candidates, key=lambda nl: nl[1])

    if best_name == "none":
        return "none", _IdentityCalibrator()
    if best_name == "platt":
        return "platt", PlattScaler().fit(val_y_arr, val_p_arr)
    return "isotonic", IsotonicCalibrator().fit(val_y_arr, val_p_arr)


__all__ = [
    "Calibrator",
    "IsotonicCalibrator",
    "PlattScaler",
    "brier_score",
    "expected_calibration_error",
    "fit_isotonic_regression",
    "fit_platt_scaling",
    "log_loss",
    "reliability_diagram",
    "select_calibration_method",
]
