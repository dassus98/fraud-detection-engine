"""Unit tests for `fraud_engine.evaluation.calibration`.

Five contract surfaces:

- `TestReliabilityDiagram`: returns `Axes`; bin-count semantics; ECE
  annotation present; `fig.savefig` smoke; diagonal reference drawn.
- `TestMetrics`: `brier_score` known values; `log_loss` matches
  sklearn; eps clipping; ECE near-zero on perfect calibration; ECE
  > 0.05 on overconfident.
- `TestPlattScaler`: fit populates state; `transform` shape `(n,)`
  ∈ [0, 1]; monotonic in input; pre-fit transform raises.
- `TestIsotonicCalibrator`: same shape contract; out-of-range inputs
  clip to `[0, 1]`; pre-fit raises.
- `TestSelectCalibrationMethod`: miscalibrated → method ∈
  {"platt", "isotonic"} AND calibrated log_loss < baseline; well-
  calibrated → method == "none" OR log_loss roughly unchanged;
  deterministic with `random_state`.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # noqa: E402 — must precede any pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from scipy.special import expit  # noqa: E402
from sklearn.metrics import log_loss as sklearn_log_loss  # noqa: E402

from fraud_engine.evaluation.calibration import (  # noqa: E402
    IsotonicCalibrator,
    PlattScaler,
    brier_score,
    expected_calibration_error,
    fit_isotonic_regression,
    fit_platt_scaling,
    log_loss,
    reliability_diagram,
    select_calibration_method,
)

_SEED = 42


def _well_calibrated_pair(n_rows: int = 4000, seed: int = _SEED) -> tuple[np.ndarray, np.ndarray]:
    """Generate `(y_true, y_pred_proba)` where predictions are perfectly calibrated.

    p_true = expit(N(0, 1)); y ~ Bernoulli(p_true); predictions = p_true.
    Expected ECE ≈ 0 (up to sampling noise on n_rows).
    """
    rng = np.random.default_rng(seed)
    p_true = expit(rng.normal(0.0, 1.0, size=n_rows))
    y = (rng.uniform(0.0, 1.0, size=n_rows) < p_true).astype(np.int64)
    return y, p_true


def _overconfident_pair(n_rows: int = 4000, seed: int = _SEED) -> tuple[np.ndarray, np.ndarray]:
    """Generate `(y_true, y_pred_proba)` where predictions are overconfident.

    p_true = expit(N(0, 1)); y ~ Bernoulli(p_true); predictions =
    expit(2 * logit(p_true)) — pushes probabilities toward extremes,
    producing the classic GBM-overconfidence miscalibration shape.
    Expected ECE > 0.05.
    """
    rng = np.random.default_rng(seed)
    p_true_logit = rng.normal(0.0, 1.0, size=n_rows)
    p_true = expit(p_true_logit)
    y = (rng.uniform(0.0, 1.0, size=n_rows) < p_true).astype(np.int64)
    p_distorted = expit(2.0 * p_true_logit)
    return y, p_distorted


# ---------------------------------------------------------------------
# `TestReliabilityDiagram`.
# ---------------------------------------------------------------------


class TestReliabilityDiagram:
    """Reliability-diagram plot contract."""

    def test_returns_axes(self) -> None:
        """Function returns a matplotlib Axes for caller customisation."""
        y, p = _well_calibrated_pair()
        ax = reliability_diagram(y, p)
        assert isinstance(ax, Axes)
        plt.close(ax.figure)

    def test_uses_supplied_axes(self) -> None:
        """When `ax` is passed, the function plots on it (no new fig)."""
        y, p = _well_calibrated_pair()
        fig, ax_in = plt.subplots()
        ax_out = reliability_diagram(y, p, ax=ax_in)
        assert ax_out is ax_in
        plt.close(fig)

    def test_ece_annotation_present(self) -> None:
        """The ECE is rendered as a text annotation in the axes."""
        y, p = _well_calibrated_pair()
        ax = reliability_diagram(y, p)
        texts = [t.get_text() for t in ax.texts]
        assert any("ECE" in t for t in texts)
        plt.close(ax.figure)

    def test_diagonal_reference_drawn(self) -> None:
        """A perfect-calibration diagonal is plotted (gray dotted line)."""
        y, p = _well_calibrated_pair()
        ax = reliability_diagram(y, p)
        # The diagonal goes (0,0) → (1,1); at least one Line2D should
        # carry that pair of endpoints.
        endpoints = [(line.get_xdata()[0], line.get_xdata()[-1]) for line in ax.get_lines()]
        assert (0.0, 1.0) in endpoints
        plt.close(ax.figure)

    def test_fig_savefig_smoke(self, tmp_path) -> None:
        """`fig.savefig(...)` completes without error on the returned axes."""
        y, p = _well_calibrated_pair()
        ax = reliability_diagram(y, p)
        out = tmp_path / "rel.png"
        ax.figure.savefig(out)
        assert out.is_file()
        assert out.stat().st_size > 0
        plt.close(ax.figure)


# ---------------------------------------------------------------------
# `TestMetrics`.
# ---------------------------------------------------------------------


class TestMetrics:
    """`brier_score`, `log_loss`, `expected_calibration_error` invariants."""

    def test_brier_perfect(self) -> None:
        """`brier_score(y, y) == 0`."""
        y = np.array([0, 1, 0, 1, 1, 0], dtype=np.int64)
        # Predictions that exactly match the labels are perfect.
        p = y.astype(np.float64)
        assert brier_score(y, p) == pytest.approx(0.0, abs=1e-12)

    def test_brier_anti_perfect(self) -> None:
        """`brier_score(y, 1-y) == 1`."""
        y = np.array([0, 1, 0, 1, 1, 0], dtype=np.int64)
        p = (1 - y).astype(np.float64)
        assert brier_score(y, p) == pytest.approx(1.0, abs=1e-12)

    def test_log_loss_matches_sklearn(self) -> None:
        """Our `log_loss` agrees with sklearn's on non-extreme inputs."""
        y, p = _well_calibrated_pair()
        ours = log_loss(y, p)
        # Apply the same eps clipping sklearn would have used.
        clipped = np.clip(p, 1e-15, 1.0 - 1e-15)
        theirs = float(sklearn_log_loss(y, clipped, labels=[0, 1]))
        assert ours == pytest.approx(theirs, rel=1e-9)

    def test_log_loss_eps_clipping_prevents_inf(self) -> None:
        """Predictions exactly at 0 or 1 don't blow up log_loss to inf."""
        y = np.array([0, 1, 0, 1])
        # Adversarial predictions: opposite-of-label at full confidence.
        p = np.array([1.0, 0.0, 1.0, 0.0])
        result = log_loss(y, p)
        assert np.isfinite(result)
        assert result > 0.0

    def test_ece_perfect_calibration(self) -> None:
        """ECE ≈ 0 on a well-calibrated synthetic frame."""
        y, p = _well_calibrated_pair(n_rows=8000)
        ece = expected_calibration_error(y, p, n_bins=10)
        # 8k rows / 10 bins ≈ 800 rows per bin → standard error of
        # the bin-rate is ~sqrt(0.25/800) = 0.018; sum-weighted is
        # smaller still.
        assert ece < 0.02

    def test_ece_overconfident_miscalibration(self) -> None:
        """ECE > 0.05 on the synthetic-overconfident frame."""
        y, p = _overconfident_pair(n_rows=8000)
        ece = expected_calibration_error(y, p, n_bins=10)
        assert ece > 0.05


# ---------------------------------------------------------------------
# `TestPlattScaler`.
# ---------------------------------------------------------------------


class TestPlattScaler:
    """Platt-scaling fit/transform contract."""

    def test_fit_populates_state(self) -> None:
        """After fit, `is_fitted_` is True and underlying LR is set."""
        y, p = _overconfident_pair()
        scaler = PlattScaler().fit(y, p)
        assert scaler.is_fitted_ is True
        assert scaler._lr is not None  # noqa: SLF001

    def test_transform_shape_and_range(self) -> None:
        """`transform(p)` returns 1-D array, all values ∈ [0, 1]."""
        y, p = _overconfident_pair()
        scaler = fit_platt_scaling(y, p)
        out = scaler.transform(p)
        assert out.shape == (len(p),)
        assert (out >= 0.0).all() and (out <= 1.0).all()

    def test_transform_monotonic_in_input(self) -> None:
        """Sorting input → output is non-decreasing (Platt is monotonic)."""
        y, p = _overconfident_pair()
        scaler = fit_platt_scaling(y, p)
        sorted_p = np.linspace(0.01, 0.99, 100)
        out = scaler.transform(sorted_p)
        assert (np.diff(out) >= -1e-12).all()

    def test_pre_fit_transform_raises(self) -> None:
        """`transform` before `fit` raises `AttributeError`."""
        scaler = PlattScaler()
        with pytest.raises(AttributeError, match="fit"):
            scaler.transform(np.array([0.5, 0.5]))


# ---------------------------------------------------------------------
# `TestIsotonicCalibrator`.
# ---------------------------------------------------------------------


class TestIsotonicCalibrator:
    """Isotonic-calibrator fit/transform contract."""

    def test_fit_populates_state(self) -> None:
        """After fit, `is_fitted_` is True and underlying iso is set."""
        y, p = _overconfident_pair()
        cal = IsotonicCalibrator().fit(y, p)
        assert cal.is_fitted_ is True
        assert cal._iso is not None  # noqa: SLF001

    def test_transform_shape_and_range(self) -> None:
        """`transform(p)` returns 1-D array, all values ∈ [0, 1]."""
        y, p = _overconfident_pair()
        cal = fit_isotonic_regression(y, p)
        out = cal.transform(p)
        assert out.shape == (len(p),)
        assert (out >= 0.0).all() and (out <= 1.0).all()

    def test_transform_clips_out_of_range_inputs(self) -> None:
        """Inputs outside `[min_train, max_train]` clip to `[0, 1]`."""
        # Train on a narrow range, then transform extreme inputs.
        rng = np.random.default_rng(_SEED)
        p_train = rng.uniform(0.3, 0.7, size=400)
        y_train = (p_train > 0.5).astype(np.int64)
        cal = fit_isotonic_regression(y_train, p_train)
        # -1 below range; 2 above range. Should clip.
        out = cal.transform(np.array([-1.0, 2.0]))
        assert 0.0 <= out[0] <= 1.0
        assert 0.0 <= out[1] <= 1.0

    def test_pre_fit_transform_raises(self) -> None:
        """`transform` before `fit` raises `AttributeError`."""
        cal = IsotonicCalibrator()
        with pytest.raises(AttributeError, match="fit"):
            cal.transform(np.array([0.5, 0.5]))


# ---------------------------------------------------------------------
# `TestSelectCalibrationMethod`.
# ---------------------------------------------------------------------


class TestSelectCalibrationMethod:
    """Method-chooser improves miscalibrated probs; preserves well-calibrated."""

    def test_miscalibrated_calibration_improves_log_loss(self) -> None:
        """Spec gate: miscalibrated → method != 'none' AND calibrated log_loss < baseline."""
        y, p = _overconfident_pair(n_rows=4000)
        method, calibrator = select_calibration_method(y, p, random_state=_SEED)
        assert method in ("platt", "isotonic")
        baseline = log_loss(y, p)
        calibrated = log_loss(y, calibrator.transform(p))
        # Calibration should improve log_loss by a non-trivial margin
        # on the synthetic-overconfident frame.
        assert calibrated < baseline
        assert (baseline - calibrated) / baseline > 0.05

    def test_well_calibrated_no_regression(self) -> None:
        """Spec gate: well-calibrated → no log-loss regression.

        The chooser may pick a method (Platt is monotonic and
        near-identity on calibrated data) but it must not degrade
        log loss versus the uncalibrated baseline beyond noise.
        """
        y, p = _well_calibrated_pair(n_rows=4000)
        _method, calibrator = select_calibration_method(y, p, random_state=_SEED)
        baseline = log_loss(y, p)
        calibrated = log_loss(y, calibrator.transform(p))
        # Allow a 1% drift either way for sample-noise. The point is
        # that the chooser doesn't make a perfectly-calibrated frame
        # noticeably worse.
        assert abs(calibrated - baseline) / baseline < 0.01

    def test_deterministic_with_random_state(self) -> None:
        """Same `random_state` → same method + identical calibrated outputs."""
        y, p = _overconfident_pair(n_rows=2000)
        method_a, cal_a = select_calibration_method(y, p, random_state=_SEED)
        method_b, cal_b = select_calibration_method(y, p, random_state=_SEED)
        assert method_a == method_b
        np.testing.assert_array_equal(cal_a.transform(p), cal_b.transform(p))

    def test_returned_calibrator_transforms_shape(self) -> None:
        """Returned calibrator's `transform` produces shape `(n,)` ∈ [0, 1]."""
        y, p = _overconfident_pair(n_rows=2000)
        _method, calibrator = select_calibration_method(y, p, random_state=_SEED)
        out = calibrator.transform(np.linspace(0.01, 0.99, 50))
        assert out.shape == (50,)
        assert (out >= 0.0).all() and (out <= 1.0).all()
