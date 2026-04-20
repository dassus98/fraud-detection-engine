"""Unit tests for `fraud_engine.utils.metrics`.

Covers the four metrics used by Sprint 4 evaluation and Sprint 6
monitoring: economic_cost, precision_recall_at_k, recall_at_fpr,
compute_psi. Each test pins a hand-computed number so the signature /
semantics cannot drift without an explicit code change.
"""

from __future__ import annotations

import numpy as np
import pytest

from fraud_engine.config.settings import Settings, get_settings
from fraud_engine.utils.metrics import (
    compute_psi,
    economic_cost,
    precision_recall_at_k,
    recall_at_fpr,
)


class TestEconomicCost:
    """Contract tests for `economic_cost`."""

    def test_matches_manual_with_defaults(self) -> None:
        # FN=1, FP=2, TP=1 under the defaults 450/35/5
        y_true = np.array([1, 0, 0, 1, 0, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        s = get_settings()
        expected = 1 * s.fraud_cost_usd + 2 * s.fp_cost_usd + 1 * s.tp_cost_usd
        assert economic_cost(y_true, y_pred) == pytest.approx(expected)

    def test_matches_manual_with_overrides(self) -> None:
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 1, 0, 0])  # TP=1, FP=1, FN=1, TN=1
        got = economic_cost(y_true, y_pred, fraud_cost=100.0, fp_cost=10.0, tp_cost=1.0)
        assert got == pytest.approx(1 * 100.0 + 1 * 10.0 + 1 * 1.0)

    def test_scales_linearly_with_fraud_cost(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FRAUD_COST_USD", "1000")
        get_settings.cache_clear()
        try:
            # 2 missed frauds, no FP, no TP
            y_true = np.array([1, 1, 0])
            y_pred = np.array([0, 0, 0])
            # fraud_cost_usd=1000 → 2 * 1000 = 2000
            assert economic_cost(y_true, y_pred) == pytest.approx(2000.0)
        finally:
            get_settings.cache_clear()

    def test_accepts_python_lists(self) -> None:
        """Contract: non-ndarray array-likes work too (pandas Series, list)."""
        got = economic_cost([1, 0, 1], [0, 0, 1], fraud_cost=10.0, fp_cost=1.0, tp_cost=0.5)
        # FN=1, FP=0, TP=1 → 10 + 0 + 0.5
        assert got == pytest.approx(10.5)


class TestPrecisionRecallAtK:
    """Contract tests for `precision_recall_at_k`."""

    def test_matches_manual(self) -> None:
        # Two of the top-2 scored items should be labelled positive.
        y_true = np.array([1, 0, 1, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        precision, recall = precision_recall_at_k(y_true, y_scores, k=2)
        # top-2 are items 0 and 1; item 0 is fraud, item 1 is not → TP=1
        assert precision == pytest.approx(0.5)
        # total fraud = 2, we caught 1 → recall = 0.5
        assert recall == pytest.approx(0.5)

    def test_all_positives_caught(self) -> None:
        y_true = np.array([1, 1, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.2, 0.1])
        precision, recall = precision_recall_at_k(y_true, y_scores, k=2)
        assert precision == pytest.approx(1.0)
        assert recall == pytest.approx(1.0)

    def test_invalid_k_raises(self) -> None:
        y_true = np.array([1, 0, 1])
        y_scores = np.array([0.5, 0.4, 0.3])
        with pytest.raises(ValueError):
            precision_recall_at_k(y_true, y_scores, k=0)
        with pytest.raises(ValueError):
            precision_recall_at_k(y_true, y_scores, k=10)


class TestRecallAtFPR:
    """Contract tests for `recall_at_fpr`."""

    def test_clean_separation(self) -> None:
        # Positives and negatives are linearly separable at 0.5.
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        # Any FPR budget >= 0 suffices; recall should be 1.
        assert recall_at_fpr(y_true, y_scores, target_fpr=0.0) == pytest.approx(1.0)

    def test_partial_overlap(self) -> None:
        y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        y_scores = np.array([0.9, 0.3, 0.85, 0.2, 0.1, 0.05, 0.02, 0.01, 0.0, 0.0])
        # At FPR=0 we can only take threshold > max negative score → only
        # score=0.9 passes → recall=1/2=0.5
        assert recall_at_fpr(y_true, y_scores, target_fpr=0.0) == pytest.approx(0.5)

    def test_degenerate_returns_zero(self) -> None:
        """When no threshold satisfies the FPR budget, return 0.0."""
        # With fpr=-0.01 no threshold qualifies — sklearn roc_curve
        # always reports fpr>=0 so this exercises the defensive branch.
        y_true = np.array([1, 0, 1, 0])
        y_scores = np.array([0.9, 0.8, 0.2, 0.1])
        assert recall_at_fpr(y_true, y_scores, target_fpr=-0.01) == pytest.approx(0.0)


class TestComputePSI:
    """Contract tests for `compute_psi`."""

    def test_stable_distributions(self) -> None:
        rng = np.random.default_rng(0)
        baseline = rng.normal(loc=0.0, scale=1.0, size=5_000)
        current = rng.normal(loc=0.0, scale=1.0, size=5_000)
        assert compute_psi(baseline, current) < 0.01

    def test_significant_drift(self) -> None:
        rng = np.random.default_rng(1)
        baseline = rng.normal(loc=0.0, scale=1.0, size=5_000)
        current = rng.normal(loc=2.0, scale=1.0, size=5_000)
        # Mean shift of 2 sigma → heavy re-binning → PSI well above 0.25.
        assert compute_psi(baseline, current) > 0.25

    def test_zero_bin_handled(self) -> None:
        """Empty current-bin doesn't produce inf/NaN."""
        baseline = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        current = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # all in the lowest bin
        psi = compute_psi(baseline, current, bins=5)
        assert np.isfinite(psi)
        assert psi > 0.0

    def test_identical_arrays_zero_psi(self) -> None:
        rng = np.random.default_rng(2)
        baseline = rng.normal(size=1_000)
        # Exact copy → PSI must be exactly 0.
        assert compute_psi(baseline, baseline) == pytest.approx(0.0, abs=1e-9)


def test_metrics_import_smoke(mock_settings: Settings) -> None:
    """All public metric functions are re-exported from fraud_engine.utils."""
    import fraud_engine.utils as utils

    assert utils.economic_cost is economic_cost
    assert utils.precision_recall_at_k is precision_recall_at_k
    assert utils.recall_at_fpr is recall_at_fpr
    assert utils.compute_psi is compute_psi
    _ = mock_settings
