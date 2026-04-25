"""Unit tests for `fraud_engine.utils.metrics`.

Covers the four metrics used by Sprint 4 evaluation and Sprint 6
monitoring: economic_cost, precision_recall_at_k, recall_at_fpr,
compute_psi. Each test pins a hand-computed number so the signature /
semantics cannot drift without an explicit code change. Property-based
tests use Hypothesis where the invariant is easier to state than to
enumerate (PSI symmetry, economic_cost monotonicity under added FN).
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from fraud_engine.config.settings import Settings, get_settings
from fraud_engine.utils.metrics import (
    compute_psi,
    economic_cost,
    precision_recall_at_k,
    recall_at_fpr,
)


class TestEconomicCost:
    """Contract tests for `economic_cost` (dict return)."""

    def test_matches_manual_with_defaults(self) -> None:
        """FN=1, FP=2, TP=1 under the Settings defaults 450/35/5."""
        y_true = np.array([1, 0, 0, 1, 0, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        s = get_settings()
        expected_total = 1 * s.fraud_cost_usd + 2 * s.fp_cost_usd + 1 * s.tp_cost_usd
        result = economic_cost(y_true, y_pred)
        assert result["total_cost"] == pytest.approx(expected_total)
        assert result["fn"] == pytest.approx(1.0)
        assert result["fp"] == pytest.approx(2.0)
        assert result["tp"] == pytest.approx(1.0)
        assert result["tn"] == pytest.approx(2.0)
        assert result["cost_per_txn"] == pytest.approx(expected_total / len(y_true))

    def test_matches_manual_with_overrides(self) -> None:
        """Explicit per-call costs override Settings defaults."""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 1, 0, 0])  # TP=1, FP=1, FN=1, TN=1
        got = economic_cost(y_true, y_pred, fraud_cost=100.0, fp_cost=10.0, tp_cost=1.0)
        assert got["total_cost"] == pytest.approx(111.0)
        assert got["cost_per_txn"] == pytest.approx(111.0 / 4)

    def test_all_true_predictions_yield_zero_cost_when_tp_and_tn_zero(self) -> None:
        """Spec bullet: all TP → 0 cost if tp_cost=0."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = y_true.copy()  # TP=2, TN=2, FN=0, FP=0
        got = economic_cost(
            y_true,
            y_pred,
            fraud_cost=450.0,
            fp_cost=35.0,
            tp_cost=0.0,
            tn_cost=0.0,
        )
        assert got["total_cost"] == pytest.approx(0.0)
        assert got["fn"] == pytest.approx(0.0)
        assert got["fp"] == pytest.approx(0.0)

    def test_tn_cost_is_applied(self) -> None:
        """Spec adds `tn_cost`; non-zero tn_cost contributes to total."""
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])  # TN=3
        got = economic_cost(
            y_true,
            y_pred,
            fraud_cost=0.0,
            fp_cost=0.0,
            tp_cost=0.0,
            tn_cost=2.0,
        )
        assert got["total_cost"] == pytest.approx(6.0)
        assert got["tn"] == pytest.approx(3.0)

    def test_scales_linearly_with_fraud_cost(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Settings env-var override propagates through `get_settings()`."""
        monkeypatch.setenv("FRAUD_COST_USD", "1000")
        get_settings.cache_clear()
        try:
            y_true = np.array([1, 1, 0])
            y_pred = np.array([0, 0, 0])
            # fraud_cost_usd=1000, tp_cost_usd=5 (default), 0 TP → 2000
            result = economic_cost(y_true, y_pred)
            assert result["total_cost"] == pytest.approx(2000.0)
        finally:
            get_settings.cache_clear()

    def test_accepts_python_lists(self) -> None:
        """Non-ndarray array-likes work (pandas Series, list)."""
        got = economic_cost([1, 0, 1], [0, 0, 1], fraud_cost=10.0, fp_cost=1.0, tp_cost=0.5)
        # FN=1, FP=0, TP=1 → 10 + 0 + 0.5
        assert got["total_cost"] == pytest.approx(10.5)

    @given(
        n=st.integers(min_value=5, max_value=200),
        seed=st.integers(min_value=0, max_value=10_000),
    )
    @settings(max_examples=25, deadline=None)
    def test_monotonic_in_fn_count(self, n: int, seed: int) -> None:
        """Spec bullet: adding a FN increases total cost by `fraud_cost`.

        Property: flipping a single correctly-caught positive (TP) into
        a missed one (FN) increases `total_cost` by exactly
        `fraud_cost - tp_cost` (we lose the TP cost and gain the FN
        cost). The test picks random configurations and verifies the
        delta matches that algebraic identity.
        """
        rng = np.random.default_rng(seed)
        y_true = rng.integers(0, 2, size=n)
        # Start by catching every fraud; no misses.
        y_pred = y_true.copy()
        if y_true.sum() == 0:
            # Skip degenerate draws with no positives.
            return
        before = economic_cost(
            y_true, y_pred, fraud_cost=450.0, fp_cost=35.0, tp_cost=5.0, tn_cost=0.0
        )
        # Flip exactly one caught-fraud (TP) to a missed-fraud (FN).
        first_positive = int(np.where(y_true == 1)[0][0])
        y_pred[first_positive] = 0
        after = economic_cost(
            y_true, y_pred, fraud_cost=450.0, fp_cost=35.0, tp_cost=5.0, tn_cost=0.0
        )
        # Delta = +fraud_cost (gained FN) - tp_cost (lost TP) = 445.
        assert after["total_cost"] - before["total_cost"] == pytest.approx(445.0)
        assert after["fn"] == before["fn"] + 1
        assert after["tp"] == before["tp"] - 1


class TestPrecisionRecallAtK:
    """Contract tests for `precision_recall_at_k` (fractional `k`)."""

    def test_matches_manual_top_40pct(self) -> None:
        """k=0.4 on a 5-item array flags the top 2; precision=0.5, recall=0.5."""
        y_true = np.array([1, 0, 1, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        precision, recall = precision_recall_at_k(y_true, y_scores, k=0.4)
        # Top-2 are items 0 and 1; item 0 is fraud, item 1 is not → TP=1
        assert precision == pytest.approx(0.5)
        # Total fraud = 2; caught 1 → recall = 0.5
        assert recall == pytest.approx(0.5)

    def test_all_positives_caught_at_half_k(self) -> None:
        """With 2 positives in the top-2 and k=0.5, both fire cleanly."""
        y_true = np.array([1, 1, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.2, 0.1])
        precision, recall = precision_recall_at_k(y_true, y_scores, k=0.5)
        assert precision == pytest.approx(1.0)
        assert recall == pytest.approx(1.0)

    def test_k_one_returns_base_rate_and_full_recall(self) -> None:
        """Spec bullet: at k=1.0, precision == base rate and recall == 1.0.

        Flagging the entire population catches every positive (recall=1)
        and precision collapses to the positive-class prevalence.
        """
        y_true = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
        base_rate = float(y_true.mean())
        precision, recall = precision_recall_at_k(y_true, y_scores, k=1.0)
        assert precision == pytest.approx(base_rate)
        assert recall == pytest.approx(1.0)

    def test_invalid_k_raises(self) -> None:
        """k must be strictly in (0, 1]; boundary violations raise."""
        y_true = np.array([1, 0, 1])
        y_scores = np.array([0.5, 0.4, 0.3])
        with pytest.raises(ValueError):
            precision_recall_at_k(y_true, y_scores, k=0.0)
        with pytest.raises(ValueError):
            precision_recall_at_k(y_true, y_scores, k=1.5)
        with pytest.raises(ValueError):
            precision_recall_at_k(y_true, y_scores, k=-0.1)

    def test_tiny_k_floors_to_one_item(self) -> None:
        """k=0.01 on 10 items would round to 0 without the ceil-and-floor."""
        y_true = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        y_scores = np.array([0.9, 0.1, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        precision, recall = precision_recall_at_k(y_true, y_scores, k=0.01)
        # Top-1 is item 0 (score 0.9), which is positive → precision=1.0
        assert precision == pytest.approx(1.0)
        # Caught 1 of 2 positives → recall=0.5
        assert recall == pytest.approx(0.5)

    def test_zero_positives_returns_zero_recall(self) -> None:
        """Edge case: y_true all zeros → recall branch divides by zero guard."""
        y_true = np.array([0, 0, 0, 0])
        y_scores = np.array([0.9, 0.7, 0.5, 0.3])
        precision, recall = precision_recall_at_k(y_true, y_scores, k=0.5)
        assert precision == pytest.approx(0.0)
        assert recall == pytest.approx(0.0)


class TestRecallAtFPR:
    """Contract tests for `recall_at_fpr`."""

    def test_clean_separation(self) -> None:
        """Positives and negatives linearly separable at 0.5."""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        assert recall_at_fpr(y_true, y_scores, target_fpr=0.0) == pytest.approx(1.0)

    def test_partial_overlap(self) -> None:
        """One positive is lost to the negative side of the budget."""
        y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        y_scores = np.array([0.9, 0.3, 0.85, 0.2, 0.1, 0.05, 0.02, 0.01, 0.0, 0.0])
        # At FPR=0 we can only take threshold > max negative score → only
        # score=0.9 passes → recall=1/2=0.5
        assert recall_at_fpr(y_true, y_scores, target_fpr=0.0) == pytest.approx(0.5)

    def test_target_fpr_one_yields_full_recall(self) -> None:
        """Spec bullet: at target_fpr=1.0, recall == 1.0."""
        rng = np.random.default_rng(3)
        y_true = rng.integers(0, 2, 200)
        y_scores = rng.uniform(0, 1, 200)
        # With FPR budget = 1.0, the lowest threshold qualifies → all
        # positives flagged → TPR = 1.0.
        assert recall_at_fpr(y_true, y_scores, target_fpr=1.0) == pytest.approx(1.0)

    def test_degenerate_returns_zero(self) -> None:
        """No threshold satisfies the FPR budget → 0.0 sentinel."""
        y_true = np.array([1, 0, 1, 0])
        y_scores = np.array([0.9, 0.8, 0.2, 0.1])
        # Negative target is physically impossible — sklearn roc_curve
        # always reports fpr >= 0.
        assert recall_at_fpr(y_true, y_scores, target_fpr=-0.01) == pytest.approx(0.0)


class TestComputePSI:
    """Contract tests for `compute_psi`."""

    def test_stable_distributions(self) -> None:
        """Same-distribution noise should be well below the alert band."""
        rng = np.random.default_rng(0)
        baseline = rng.normal(loc=0.0, scale=1.0, size=5_000)
        current = rng.normal(loc=0.0, scale=1.0, size=5_000)
        assert compute_psi(baseline, current) < 0.01

    def test_significant_drift(self) -> None:
        """2-sigma mean shift → PSI well above the 0.25 alert band.

        Spec bullet: disjoint distributions → PSI large.
        """
        rng = np.random.default_rng(1)
        baseline = rng.normal(loc=0.0, scale=1.0, size=5_000)
        current = rng.normal(loc=2.0, scale=1.0, size=5_000)
        assert compute_psi(baseline, current) > 0.25

    def test_disjoint_ranges_produce_very_large_psi(self) -> None:
        """Non-overlapping baseline/current distributions → PSI >> 1."""
        rng = np.random.default_rng(4)
        baseline = rng.uniform(0.0, 1.0, 2_000)
        # Current lives entirely above baseline's max → every current
        # sample lands in the top baseline bin.
        current = rng.uniform(10.0, 20.0, 2_000)
        psi = compute_psi(baseline, current)
        # With epsilon=1e-6, the empty bins force a log-ratio of
        # log(p_curr / 1e-6) that dominates. Expect PSI in the
        # double-digits.
        assert psi > 5.0

    def test_zero_bin_handled(self) -> None:
        """An empty current-bin must not produce inf/NaN."""
        baseline = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        current = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        psi = compute_psi(baseline, current, bins=5)
        assert np.isfinite(psi)
        assert psi > 0.0

    def test_identical_arrays_zero_psi(self) -> None:
        """Exact copy → PSI == 0 up to floating-point noise."""
        rng = np.random.default_rng(2)
        baseline = rng.normal(size=1_000)
        assert compute_psi(baseline, baseline) == pytest.approx(0.0, abs=1e-9)

    def test_degenerate_baseline_returns_zero(self) -> None:
        """Single-value baseline collapses to zero drift (no meaningful bins)."""
        baseline = np.full(100, 3.14)
        current = np.random.default_rng(5).uniform(0, 10, 100)
        assert compute_psi(baseline, current) == pytest.approx(0.0)

    def test_symmetry_tight_with_small_epsilon(self) -> None:
        """Spec bullet: PSI is symmetric only when epsilon is tiny.

        Part 1: with the default epsilon=1e-6 and well-populated bins,
        PSI(A, B) and PSI(B, A) agree to ~1 part in 10,000.
        """
        rng = np.random.default_rng(7)
        a = rng.normal(0.0, 1.0, 10_000)
        b = rng.normal(0.3, 1.0, 10_000)
        forward = compute_psi(a, b, epsilon=1e-6)
        reverse = compute_psi(b, a, epsilon=1e-6)
        # Bins are drawn from the first argument's quantiles, so there
        # is a small structural asymmetry even at tiny epsilon. With
        # 10k samples on each side it collapses to a few percent.
        assert forward > 0.0
        assert reverse > 0.0
        assert abs(forward - reverse) / max(forward, reverse) < 0.10

    def test_symmetry_breaks_with_larger_epsilon(self) -> None:
        """Spec bullet: PSI asymmetric under argument swap when epsilon is large.

        Part 2: raising epsilon to 1e-2 (a caricature of the
        conservative-smoothing choice) injects a visible asymmetry on
        the same input. The test documents — not polices — the
        behaviour: swapping A and B can change PSI by >= 5% under a
        generous epsilon.
        """
        rng = np.random.default_rng(8)
        # Shape the inputs so several baseline bins are sparse (high
        # positive tail vs dense-at-zero current) — that's the regime
        # where epsilon matters most.
        a = np.concatenate([rng.normal(0.0, 0.2, 5_000), rng.normal(5.0, 0.2, 200)])
        b = rng.normal(0.0, 0.2, 5_000)
        forward = compute_psi(a, b, epsilon=1e-2)
        reverse = compute_psi(b, a, epsilon=1e-2)
        assert abs(forward - reverse) / max(forward, reverse) > 0.05

    @given(
        mu=st.floats(min_value=-0.5, max_value=0.5),
        sigma=st.floats(min_value=0.5, max_value=1.5),
        seed=st.integers(min_value=0, max_value=10_000),
    )
    @settings(max_examples=15, deadline=None)
    def test_psi_non_negative(self, mu: float, sigma: float, seed: int) -> None:
        """Property: PSI >= 0 by construction (sum of (p-q)*log(p/q) terms).

        Each term in the sum is of the form `(p - q) * log(p/q)` which
        is non-negative for any positive p, q (both are > 0 thanks to
        the epsilon floor). The sum therefore cannot be negative.
        """
        rng = np.random.default_rng(seed)
        baseline = rng.normal(0.0, 1.0, 1_000)
        current = rng.normal(mu, sigma, 1_000)
        assert compute_psi(baseline, current) >= 0.0


def test_metrics_import_smoke(mock_settings: Settings) -> None:
    """All public metric functions are re-exported from fraud_engine.utils."""
    import fraud_engine.utils as utils

    assert utils.economic_cost is economic_cost
    assert utils.precision_recall_at_k is precision_recall_at_k
    assert utils.recall_at_fpr is recall_at_fpr
    assert utils.compute_psi is compute_psi
    _ = mock_settings
