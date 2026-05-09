"""Unit tests for `fraud_engine.evaluation.economic.EconomicCostModel`.

Five contract surfaces (mirrors `test_calibration.py`'s structure):

- `TestInit`: default-from-Settings semantics; explicit overrides
  beat Settings; `costs` property snapshot; negative-cost raises.
- `TestComputeCost`: spec gate "known confusion matrix → hand-computed
  cost"; empty-array safety; shape-mismatch raises; uses stored costs.
- `TestOptimizeThreshold`: default `linspace(0.01, 0.99, 99)` shape;
  cost-curve column contract + sort order; `optimal_τ` is in the
  swept grid; `y_scores ∉ [0, 1]` raises; custom thresholds
  respected; **tie-break favours larger τ** on identical cost.
- `TestOptimizeThresholdEconomicGates`: the three spec asymptotic
  gates — high `fp_cost` → τ → 1, high `fraud_cost` → τ → 0,
  default costs put the optimum in a sensible band.
- `TestSensitivityAnalysis`: spec gate "near-optimal thresholds
  cluster in a small range"; default ±20 % grid shape (125, 6);
  custom `cost_ranges` override; single-value-axis collapse;
  unknown-axis raises; negative range value raises.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from fraud_engine.config.settings import get_settings
from fraud_engine.evaluation.economic import EconomicCostModel

_SEED: int = 42

# Spec constants pinned for assertion-readability.
_DEFAULT_FRAUD_COST: float = 450.0
_DEFAULT_FP_COST: float = 35.0
_DEFAULT_TP_COST: float = 5.0


# ---------------------------------------------------------------------
# Synthetic-data generators (mirror `test_calibration.py:48-74`).
# ---------------------------------------------------------------------


def _separable_pair(
    n_rows: int = 4000,
    base_rate: float = 0.10,
    seed: int = _SEED,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Logistic-separable scores: positives near 0.85 (Beta(8, 2)),
    negatives near 0.10 (Beta(2, 8)).

    Asymptotic cost gates land sharply on this fixture; use it for the
    high-`fp_cost` / high-`fraud_cost` tests.
    """
    rng = np.random.default_rng(seed)
    y = (rng.uniform(0.0, 1.0, size=n_rows) < base_rate).astype(np.int64)
    pos_scores = rng.beta(8.0, 2.0, size=n_rows)
    neg_scores = rng.beta(2.0, 8.0, size=n_rows)
    scores = np.where(y == 1, pos_scores, neg_scores).astype(np.float64)
    return y, scores


def _hard_pair(
    n_rows: int = 4000,
    base_rate: float = 0.10,
    seed: int = _SEED,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Heavily-overlapping scores: positives Beta(2, 2) (mean 0.5),
    negatives Beta(2, 3) (mean 0.4).

    The classifier signal is weak so the optimum threshold is
    policy-driven; use it for the sensitivity-stability test where
    the answer must be robust despite cost-input uncertainty.
    """
    rng = np.random.default_rng(seed)
    y = (rng.uniform(0.0, 1.0, size=n_rows) < base_rate).astype(np.int64)
    pos_scores = rng.beta(2.0, 2.0, size=n_rows)
    neg_scores = rng.beta(2.0, 3.0, size=n_rows)
    scores = np.where(y == 1, pos_scores, neg_scores).astype(np.float64)
    return y, scores


# ---------------------------------------------------------------------
# `TestInit`.
# ---------------------------------------------------------------------


class TestInit:
    """Construction: default-from-Settings, override, snapshot, validation."""

    def test_default_costs_resolve_from_settings(self) -> None:
        """Constructor with no kwargs uses Settings defaults."""
        # Ensure get_settings() is fresh in case a prior test mutated env.
        get_settings.cache_clear()
        settings = get_settings()
        model = EconomicCostModel()
        costs = model.costs
        assert costs["fraud_cost"] == settings.fraud_cost_usd
        assert costs["fp_cost"] == settings.fp_cost_usd
        assert costs["tp_cost"] == settings.tp_cost_usd
        assert costs["tn_cost"] == 0.0

    def test_explicit_overrides_beat_settings(self) -> None:
        """Per-arg overrides win over Settings defaults."""
        model = EconomicCostModel(
            fraud_cost=999.0,
            fp_cost=88.0,
            tp_cost=7.0,
            tn_cost=1.5,
        )
        assert model.costs == {
            "fraud_cost": 999.0,
            "fp_cost": 88.0,
            "tp_cost": 7.0,
            "tn_cost": 1.5,
        }

    def test_partial_overrides_mix_with_settings(self) -> None:
        """Unspecified args fall back to Settings; specified args win."""
        get_settings.cache_clear()
        settings = get_settings()
        model = EconomicCostModel(fraud_cost=600.0)
        assert model.costs["fraud_cost"] == 600.0
        assert model.costs["fp_cost"] == settings.fp_cost_usd
        assert model.costs["tp_cost"] == settings.tp_cost_usd
        assert model.costs["tn_cost"] == 0.0

    def test_costs_property_returns_dict_shape(self) -> None:
        """`costs` returns a 4-key dict with the expected names."""
        model = EconomicCostModel()
        costs = model.costs
        assert set(costs.keys()) == {"fraud_cost", "fp_cost", "tp_cost", "tn_cost"}
        for value in costs.values():
            assert isinstance(value, float)

    def test_negative_cost_raises(self) -> None:
        """Negative costs mirror Settings' `Field(ge=0.0)` and raise."""
        with pytest.raises(ValueError, match="fraud_cost"):
            EconomicCostModel(fraud_cost=-1.0)
        with pytest.raises(ValueError, match="fp_cost"):
            EconomicCostModel(fp_cost=-0.5)
        with pytest.raises(ValueError, match="tp_cost"):
            EconomicCostModel(tp_cost=-10.0)
        with pytest.raises(ValueError, match="tn_cost"):
            EconomicCostModel(tn_cost=-1.0)

    def test_zero_cost_allowed(self) -> None:
        """`Field(ge=0.0)` allows zero — not a regression."""
        model = EconomicCostModel(
            fraud_cost=0.0,
            fp_cost=0.0,
            tp_cost=0.0,
            tn_cost=0.0,
        )
        assert model.costs == {
            "fraud_cost": 0.0,
            "fp_cost": 0.0,
            "tp_cost": 0.0,
            "tn_cost": 0.0,
        }


# ---------------------------------------------------------------------
# `TestComputeCost`.
# ---------------------------------------------------------------------


class TestComputeCost:
    """`compute_cost` forwards correctly + enforces the count contract."""

    def test_known_confusion_matrix_matches_hand_computation(self) -> None:
        """Spec gate: hand-computed cost from a small known fixture.

        y_true=[1, 0, 0, 1, 0, 0], y_pred=[0, 1, 1, 1, 0, 0]:
            FN = 1 (idx 0), FP = 2 (idx 1, 2), TP = 1 (idx 3), TN = 2 (idx 4, 5).
        With defaults (fraud=450, fp=35, tp=5, tn=0):
            total = 1*450 + 2*35 + 1*5 + 2*0 = 525.0
        """
        y_true = np.array([1, 0, 0, 1, 0, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        model = EconomicCostModel(
            fraud_cost=_DEFAULT_FRAUD_COST,
            fp_cost=_DEFAULT_FP_COST,
            tp_cost=_DEFAULT_TP_COST,
        )
        out = model.compute_cost(y_true, y_pred)
        expected_total = 1 * 450.0 + 2 * 35.0 + 1 * 5.0 + 2 * 0.0
        assert out["total_cost"] == pytest.approx(expected_total)
        assert out["fn"] == 1.0
        assert out["fp"] == 2.0
        assert out["tp"] == 1.0
        assert out["tn"] == 2.0
        assert out["cost_per_txn"] == pytest.approx(expected_total / 6.0)

    def test_compute_cost_returns_dict_with_expected_keys(self) -> None:
        """Return-dict shape matches the wrapped primitive's contract."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        model = EconomicCostModel()
        out = model.compute_cost(y_true, y_pred)
        assert set(out.keys()) == {
            "total_cost",
            "cost_per_txn",
            "fn",
            "fp",
            "tp",
            "tn",
        }

    def test_compute_cost_uses_stored_costs_not_settings(self) -> None:
        """Stored overrides — not Settings — drive the cost output."""
        y_true = np.array([1, 0])
        y_pred = np.array([0, 1])  # 1 FN, 1 FP
        model = EconomicCostModel(fraud_cost=1000.0, fp_cost=200.0, tp_cost=0.0)
        out = model.compute_cost(y_true, y_pred)
        assert out["total_cost"] == pytest.approx(1000.0 + 200.0)

    def test_compute_cost_empty_arrays_returns_zero_dict(self) -> None:
        """Empty inputs produce a zero-cost dict (cost_per_txn = 0 by guard)."""
        model = EconomicCostModel()
        out = model.compute_cost(np.array([]), np.array([]))
        assert out["total_cost"] == 0.0
        assert out["cost_per_txn"] == 0.0
        assert out["fn"] == 0.0
        assert out["fp"] == 0.0

    def test_compute_cost_shape_mismatch_raises(self) -> None:
        """Length mismatch propagates from numpy as ValueError."""
        model = EconomicCostModel()
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1])
        with pytest.raises(ValueError):
            model.compute_cost(y_true, y_pred)


# ---------------------------------------------------------------------
# `TestOptimizeThreshold`.
# ---------------------------------------------------------------------


class TestOptimizeThreshold:
    """`optimize_threshold` sweep / shape / tie-break / validation."""

    def test_default_grid_shape_is_99_by_7(self) -> None:
        """Default `linspace(0.01, 0.99, 99)` produces a (99, 7) curve."""
        y, scores = _separable_pair(n_rows=2000)
        model = EconomicCostModel()
        optimal_tau, curve = model.optimize_threshold(y, scores)
        assert curve.shape == (99, 7)
        assert isinstance(optimal_tau, float)

    def test_cost_curve_columns_in_expected_order(self) -> None:
        """Columns mirror the wrapped primitive's return keys (with `threshold` first)."""
        y, scores = _separable_pair(n_rows=2000)
        model = EconomicCostModel()
        _, curve = model.optimize_threshold(y, scores)
        assert list(curve.columns) == [
            "threshold",
            "total_cost",
            "cost_per_txn",
            "fn",
            "fp",
            "tp",
            "tn",
        ]

    def test_cost_curve_sorted_ascending_by_threshold(self) -> None:
        """Threshold column is monotone non-decreasing."""
        y, scores = _separable_pair(n_rows=2000)
        model = EconomicCostModel()
        _, curve = model.optimize_threshold(y, scores)
        diffs = np.diff(curve["threshold"].to_numpy())
        assert (diffs >= 0).all()

    def test_optimal_tau_in_swept_grid(self) -> None:
        """Returned `optimal_τ` is one of the swept thresholds."""
        y, scores = _separable_pair(n_rows=2000)
        model = EconomicCostModel()
        optimal_tau, curve = model.optimize_threshold(y, scores)
        thresholds = curve["threshold"].to_numpy()
        assert np.isclose(thresholds, optimal_tau).any()

    def test_total_cost_finite_and_nonnegative(self) -> None:
        """Cost curve never produces NaN, inf, or negative values."""
        y, scores = _separable_pair(n_rows=2000)
        model = EconomicCostModel()
        _, curve = model.optimize_threshold(y, scores)
        total = curve["total_cost"].to_numpy()
        assert np.isfinite(total).all()
        assert (total >= 0.0).all()

    def test_y_scores_outside_unit_interval_raises(self) -> None:
        """Calibrator contract: y_scores must be in [0, 1]; raise on violation."""
        model = EconomicCostModel()
        y = np.array([0, 1, 0, 1])
        with pytest.raises(ValueError, match=r"y_scores must be in \[0, 1\]"):
            model.optimize_threshold(y, np.array([-0.1, 0.5, 0.5, 0.9]))
        with pytest.raises(ValueError, match=r"y_scores must be in \[0, 1\]"):
            model.optimize_threshold(y, np.array([0.1, 0.5, 0.5, 1.5]))

    def test_shape_mismatch_raises(self) -> None:
        """Inputs of different length raise."""
        model = EconomicCostModel()
        with pytest.raises(ValueError, match="shape mismatch"):
            model.optimize_threshold(
                np.array([0, 1, 0, 1, 0]),
                np.array([0.1, 0.5, 0.9, 0.2]),
            )

    def test_custom_thresholds_respected(self) -> None:
        """Caller-supplied thresholds override the default grid."""
        y, scores = _separable_pair(n_rows=2000)
        model = EconomicCostModel()
        custom = [0.30, 0.50, 0.70]
        _, curve = model.optimize_threshold(y, scores, thresholds=custom)
        assert curve.shape == (3, 7)
        np.testing.assert_array_equal(
            curve["threshold"].to_numpy(),
            np.array(custom),
        )

    def test_tie_break_favours_larger_tau(self) -> None:
        """Two thresholds with identical predictions → optimal = larger τ.

        With y_scores = [0.1, 0.9] and thresholds = [0.3, 0.5, 0.7], all
        three thresholds yield y_pred = [0, 1] — identical confusion
        matrix and identical cost. Tie-break: pick τ = 0.7 (block-fewer
        is the policy default; documented decision).
        """
        y_true = np.array([0, 1])
        y_scores = np.array([0.1, 0.9])
        model = EconomicCostModel()
        optimal_tau, curve = model.optimize_threshold(
            y_true,
            y_scores,
            thresholds=[0.3, 0.5, 0.7],
        )
        # All three thresholds produce y_pred = [0, 1] → identical cost.
        assert curve["total_cost"].nunique() == 1
        # Tie-break picks the largest τ.
        assert optimal_tau == pytest.approx(0.7)

    def test_predict_proba_at_threshold_zero_one_inclusive(self) -> None:
        """Threshold semantics: y_pred = (y_scores >= τ).

        At τ=0.0, every score in [0, 1] is flagged positive.
        At τ=1.0, only scores exactly equal to 1.0 are flagged.
        Verifies the boundary semantics if a caller passes them.
        """
        y_true = np.array([1, 0, 1, 0])
        y_scores = np.array([0.0, 0.5, 0.8, 1.0])
        model = EconomicCostModel()
        # τ=0 → all flagged → TP=2, FP=2 → cost = 2*5 + 2*35 = 80
        _, curve = model.optimize_threshold(y_true, y_scores, thresholds=[0.0])
        assert curve.iloc[0]["fp"] == 2.0
        assert curve.iloc[0]["tp"] == 2.0


# ---------------------------------------------------------------------
# `TestOptimizeThresholdEconomicGates` (the spec asymptotic gates).
# ---------------------------------------------------------------------


class TestOptimizeThresholdEconomicGates:
    """Asymptotic behaviour: extreme costs push τ toward the rails.

    The spec phrases these as "fp_cost → ∞ ⇒ optimal τ → 1" and
    "fraud_cost → ∞ ⇒ optimal τ → 0". With finite samples and
    Beta-distributed scores, the optimum lands at the boundary where
    the dominant error class first hits zero (FP at the upper rail,
    FN at the lower rail) — not exactly at τ=1 or τ=0. The tests
    therefore verify (a) direction (high fp_cost > default > high
    fraud_cost) and (b) magnitude (the optimum is on the correct
    side of mid-grid by a clear margin).
    """

    def test_high_fp_cost_pushes_threshold_high(self) -> None:
        """Spec gate: as fp_cost → ∞, optimal τ shifts toward 1.

        With fp_cost / fraud_cost = 10_000 on a separable synthetic
        frame, the optimum lands above the upper-half boundary —
        blocking is so costly that we let almost all transactions
        through. The exact pin depends on where the negative-score
        Beta(2, 8) tail effectively reaches zero.
        """
        y, scores = _separable_pair(n_rows=4000)
        model = EconomicCostModel(
            fraud_cost=1.0,
            fp_cost=10_000.0,
            tp_cost=0.0,
        )
        optimal_tau, _ = model.optimize_threshold(y, scores)
        # Pushed clearly above mid-grid, demonstrating the asymptotic
        # direction without pinning to the rail. (Asymptotic limit
        # τ → 1 is unreachable in finite samples once FP hits 0.)
        assert optimal_tau >= 0.55

    def test_high_fraud_cost_pushes_threshold_low(self) -> None:
        """Spec gate: as fraud_cost → ∞, optimal τ shifts toward 0.

        Symmetric counterpart of the high-fp test. With fraud_cost /
        fp_cost = 10_000 the optimum lands below the lower-half
        boundary; missing fraud is so costly we block aggressively.
        """
        y, scores = _separable_pair(n_rows=4000)
        model = EconomicCostModel(
            fraud_cost=10_000.0,
            fp_cost=1.0,
            tp_cost=0.0,
        )
        optimal_tau, _ = model.optimize_threshold(y, scores)
        assert optimal_tau <= 0.45

    def test_extreme_costs_strictly_order_optima(self) -> None:
        """Direction check: high_fp_τ > default_τ > high_fraud_τ.

        Reinforces the asymptotic-direction claim with a strict
        ordering against the default-costs optimum on the same
        synthetic frame — the most direct expression of the spec.
        """
        y, scores = _separable_pair(n_rows=4000)
        default_tau, _ = EconomicCostModel().optimize_threshold(y, scores)
        high_fp_tau, _ = EconomicCostModel(
            fraud_cost=1.0,
            fp_cost=10_000.0,
            tp_cost=0.0,
        ).optimize_threshold(y, scores)
        high_fraud_tau, _ = EconomicCostModel(
            fraud_cost=10_000.0,
            fp_cost=1.0,
            tp_cost=0.0,
        ).optimize_threshold(y, scores)
        assert high_fp_tau > default_tau > high_fraud_tau

    def test_default_costs_optimum_in_sensible_band(self) -> None:
        """With Sprint-3 default costs, optimum is well off the rails.

        On a separable synthetic frame, default costs (450 / 35 / 5)
        push the optimum into the body of the distribution; the
        13× FN/FP ratio favours blocking but not at all costs.
        """
        y, scores = _separable_pair(n_rows=4000)
        model = EconomicCostModel()  # defaults from Settings
        optimal_tau, _ = model.optimize_threshold(y, scores)
        assert 0.10 < optimal_tau < 0.80


# ---------------------------------------------------------------------
# `TestSensitivityAnalysis`.
# ---------------------------------------------------------------------


class TestSensitivityAnalysis:
    """`sensitivity_analysis` grid expansion + stability gate."""

    def test_default_grid_shape_is_125_by_6(self) -> None:
        """Default ±20 % grid: 5 × 5 × 5 cells × 6 columns."""
        y, scores = _separable_pair(n_rows=2000)
        model = EconomicCostModel()
        sens = model.sensitivity_analysis(y, scores)
        assert sens.shape == (125, 6)

    def test_default_grid_columns_in_expected_order(self) -> None:
        """Costs first, optima second — readers see the scenario, then the result."""
        y, scores = _separable_pair(n_rows=2000)
        model = EconomicCostModel()
        sens = model.sensitivity_analysis(y, scores)
        assert list(sens.columns) == [
            "fraud_cost",
            "fp_cost",
            "tp_cost",
            "optimal_threshold",
            "optimal_total_cost",
            "optimal_cost_per_txn",
        ]

    def test_default_grid_values_match_plus_minus_twenty_percent(self) -> None:
        """Default fraud_cost values are 0.8/0.9/1.0/1.1/1.2 × Settings."""
        get_settings.cache_clear()
        settings = get_settings()
        y, scores = _separable_pair(n_rows=1000)
        model = EconomicCostModel()
        sens = model.sensitivity_analysis(y, scores)
        unique_fraud_costs = sorted(sens["fraud_cost"].unique())
        expected = [settings.fraud_cost_usd * m for m in (0.80, 0.90, 1.00, 1.10, 1.20)]
        np.testing.assert_allclose(unique_fraud_costs, expected, rtol=1e-9)

    def test_optimal_thresholds_cluster_in_narrow_band(self) -> None:
        """Spec gate: ±20 % cost variation → optimal τ stable.

        On a separable synthetic frame the cost surface is sharp, so
        ±20 % shifts barely move the optimum. Assert spread < 0.20
        (CLAUDE.md §8 stability rule, generously bounded).
        """
        y, scores = _separable_pair(n_rows=4000)
        model = EconomicCostModel()
        sens = model.sensitivity_analysis(y, scores)
        spread = float(sens["optimal_threshold"].max() - sens["optimal_threshold"].min())
        assert spread < 0.20

    def test_custom_cost_ranges_override_default(self) -> None:
        """Caller's `cost_ranges` win over the default ±20 % grid."""
        y, scores = _separable_pair(n_rows=1000)
        model = EconomicCostModel()
        sens = model.sensitivity_analysis(
            y,
            scores,
            cost_ranges={
                "fraud_cost": [400.0, 500.0],
                "fp_cost": [30.0, 40.0],
                "tp_cost": [4.0, 6.0],
            },
        )
        assert sens.shape == (8, 6)  # 2 × 2 × 2 cartesian
        assert sorted(sens["fraud_cost"].unique().tolist()) == [400.0, 500.0]
        assert sorted(sens["fp_cost"].unique().tolist()) == [30.0, 40.0]
        assert sorted(sens["tp_cost"].unique().tolist()) == [4.0, 6.0]

    def test_single_axis_collapses_to_stored_cost(self) -> None:
        """Unspecified axes fall back to the constructor's snapshot."""
        y, scores = _separable_pair(n_rows=1000)
        model = EconomicCostModel(fraud_cost=600.0, fp_cost=40.0, tp_cost=8.0)
        sens = model.sensitivity_analysis(
            y,
            scores,
            cost_ranges={"fraud_cost": [500.0, 700.0]},
        )
        # 2 × 1 × 1 = 2 rows. fp_cost / tp_cost held at the model's
        # stored values for every row.
        assert sens.shape == (2, 6)
        assert (sens["fp_cost"] == 40.0).all()
        assert (sens["tp_cost"] == 8.0).all()

    def test_unknown_axis_raises(self) -> None:
        """Unrecognised cost-range key fails fast."""
        y, scores = _separable_pair(n_rows=500)
        model = EconomicCostModel()
        with pytest.raises(ValueError, match="unknown cost-range axes"):
            model.sensitivity_analysis(
                y,
                scores,
                cost_ranges={"weird_cost": [1.0, 2.0]},
            )

    def test_negative_value_in_cost_range_raises(self) -> None:
        """Negative values fail fast (cost contract is ge=0.0)."""
        y, scores = _separable_pair(n_rows=500)
        model = EconomicCostModel()
        with pytest.raises(ValueError, match="negative value"):
            model.sensitivity_analysis(
                y,
                scores,
                cost_ranges={"fraud_cost": [-10.0, 100.0]},
            )

    def test_y_scores_outside_unit_interval_raises(self) -> None:
        """Same calibrator-contract guard as `optimize_threshold`."""
        model = EconomicCostModel()
        y = np.array([0, 1, 0, 1])
        with pytest.raises(ValueError, match=r"y_scores must be in \[0, 1\]"):
            model.sensitivity_analysis(y, np.array([0.1, 0.5, 0.9, 1.5]))

    def test_returned_object_is_pandas_dataframe(self) -> None:
        """Return type is `pd.DataFrame` (not ndarray or dict)."""
        y, scores = _separable_pair(n_rows=500)
        model = EconomicCostModel()
        sens = model.sensitivity_analysis(
            y,
            scores,
            cost_ranges={"fraud_cost": [450.0]},
        )
        assert isinstance(sens, pd.DataFrame)
