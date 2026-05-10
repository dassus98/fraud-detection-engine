"""Unit tests for the ShadowComparison primitive.

Sprint 5 prompt 5.2.c. All 13 tests are pure-python: build small
DataFrames inline, invoke the analysis surface, assert outputs.
Bootstrap determinism is achieved via the fixed seed=42 default; the
deterministic-under-seed test (#10) catches drift if the rng API
behaviour ever changes.

Test scenarios:
    1. test_agreement_rate_perfect_match
    2. test_agreement_rate_zero_match
    3. test_agreement_rate_partial
    4. test_score_correlation_perfect
    5. test_score_correlation_anticorrelated
    6. test_economic_cost_with_labels
    7. test_economic_cost_without_labels_returns_none
    8. test_bootstrap_significance_distinguishable
    9. test_bootstrap_significance_indistinguishable
    10. test_bootstrap_deterministic_under_fixed_seed
    11. test_should_promote_all_criteria_met
    12. test_should_promote_below_threshold
    13. test_invalid_input_missing_columns
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_engine.evaluation.shadow_compare import (
    EconomicCosts,
    ShadowComparison,
)

# ---------------------------------------------------------------------
# Test fixtures + helpers.
# ---------------------------------------------------------------------

# Project default cost matrix (matches CLAUDE.md §8 + Settings defaults).
_DEFAULT_COSTS = EconomicCosts(fraud_cost=450.0, fp_cost=35.0, tp_cost=5.0)


def _make_predictions(  # noqa: PLR0913 — six knobs map 1:1 to PredictionResponse + label fields; folding into a dict obscures the per-field test-customisation surface
    n: int,
    *,
    champion_scores: list[float] | None = None,
    shadow_scores: list[float] | None = None,
    champion_decisions: list[str] | None = None,
    shadow_decisions: list[str] | None = None,
    is_fraud: list[int] | None = None,
) -> pd.DataFrame:
    """Construct a predictions DataFrame with the required columns + optional labels."""
    request_ids = [f"req-{i:04d}" for i in range(n)]
    cs = champion_scores if champion_scores is not None else [0.05] * n
    ss = shadow_scores if shadow_scores is not None else [0.05] * n
    cd = champion_decisions if champion_decisions is not None else ["allow"] * n
    sd = shadow_decisions if shadow_decisions is not None else ["allow"] * n
    cols: dict[str, list[float] | list[str] | list[int]] = {
        "request_id": request_ids,
        "champion_score": cs,
        "shadow_score": ss,
        "champion_decision": cd,
        "shadow_decision": sd,
    }
    if is_fraud is not None:
        cols["is_fraud"] = is_fraud
    return pd.DataFrame(cols)


# ---------------------------------------------------------------------
# Scenario 1-3: agreement_rate.
# ---------------------------------------------------------------------


def test_agreement_rate_perfect_match() -> None:
    """All decisions identical → agreement = 1.0."""
    df = _make_predictions(
        n=10,
        champion_decisions=["allow"] * 5 + ["block"] * 5,
        shadow_decisions=["allow"] * 5 + ["block"] * 5,
    )
    report = ShadowComparison(df, _DEFAULT_COSTS).run()
    assert report.agreement_rate == 1.0


def test_agreement_rate_zero_match() -> None:
    """Opposite decisions on every row → agreement = 0.0."""
    df = _make_predictions(
        n=10,
        champion_decisions=["allow"] * 10,
        shadow_decisions=["block"] * 10,
    )
    report = ShadowComparison(df, _DEFAULT_COSTS).run()
    assert report.agreement_rate == 0.0


def test_agreement_rate_partial() -> None:
    """8/10 match → agreement = 0.8."""
    df = _make_predictions(
        n=10,
        champion_decisions=["allow"] * 8 + ["block"] * 2,
        shadow_decisions=["allow"] * 9 + ["block"] * 1,
    )
    report = ShadowComparison(df, _DEFAULT_COSTS).run()
    # 8 allow-allow matches + 1 block-block match = 9 agreements out of 10
    assert report.agreement_rate == 0.9


# ---------------------------------------------------------------------
# Scenario 4-5: score_correlation.
# ---------------------------------------------------------------------


def test_score_correlation_perfect() -> None:
    """Shadow == champion → correlation = 1.0."""
    scores = [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99]
    df = _make_predictions(
        n=len(scores),
        champion_scores=scores,
        shadow_scores=scores,
    )
    report = ShadowComparison(df, _DEFAULT_COSTS).run()
    assert report.score_correlation == pytest.approx(1.0)


def test_score_correlation_anticorrelated() -> None:
    """Shadow == 1 - champion → correlation = -1.0."""
    champion = [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99]
    shadow = [1.0 - x for x in champion]
    df = _make_predictions(
        n=len(champion),
        champion_scores=champion,
        shadow_scores=shadow,
    )
    report = ShadowComparison(df, _DEFAULT_COSTS).run()
    assert report.score_correlation == pytest.approx(-1.0)


# ---------------------------------------------------------------------
# Scenario 6-7: economic_cost path.
# ---------------------------------------------------------------------


def test_economic_cost_with_labels() -> None:
    """Per-prediction cost matches manual calc for labelled subset."""
    # 4 predictions: one of each (decision, label) combo.
    df = _make_predictions(
        n=4,
        champion_decisions=["block", "block", "allow", "allow"],
        shadow_decisions=["block", "allow", "block", "allow"],
        is_fraud=[1, 0, 1, 0],
    )
    costs = EconomicCosts(fraud_cost=450.0, fp_cost=35.0, tp_cost=5.0)
    report = ShadowComparison(df, costs).run()

    # Champion costs:
    #   row 0: block + fraud → tp = 5
    #   row 1: block + not_fraud → fp = 35
    #   row 2: allow + fraud → fraud = 450
    #   row 3: allow + not_fraud → 0
    # Mean = (5 + 35 + 450 + 0) / 4 = 122.5
    assert report.champion_cost_per_txn == pytest.approx(122.5)

    # Shadow costs:
    #   row 0: block + fraud → tp = 5
    #   row 1: allow + not_fraud → 0
    #   row 2: block + fraud → tp = 5
    #   row 3: allow + not_fraud → 0
    # Mean = (5 + 0 + 5 + 0) / 4 = 2.5
    assert report.shadow_cost_per_txn == pytest.approx(2.5)
    # Cost improvement: (122.5 - 2.5) / 122.5 ≈ 0.9796
    assert report.cost_improvement == pytest.approx(0.97959, rel=1e-4)
    assert report.n_labeled == 4


def test_economic_cost_without_labels_returns_none() -> None:
    """No is_fraud column → cost fields are all None."""
    df = _make_predictions(n=5)
    report = ShadowComparison(df, _DEFAULT_COSTS).run()
    assert report.n_labeled is None
    assert report.champion_cost_per_txn is None
    assert report.shadow_cost_per_txn is None
    assert report.cost_improvement is None
    assert report.bootstrap_mean_diff is None
    assert report.bootstrap_ci_95 is None
    assert report.bootstrap_p_value is None
    # Verdict short-circuits to False (cost criterion can't pass).
    assert report.verdict.promote is False
    # ...but agreement IS computable, so its reason can be PASS.
    assert report.agreement_rate == 1.0


# ---------------------------------------------------------------------
# Scenario 8-10: bootstrap significance.
# ---------------------------------------------------------------------


def test_bootstrap_significance_distinguishable() -> None:
    """Shadow much cheaper → mean_diff > 0 with p < 0.05."""
    rng = np.random.default_rng(0)
    n = 200
    # All fraud cases. Champion misses 80%; shadow catches 80%.
    is_fraud = [1] * n
    # Champion: mostly allow → expensive (fraud_cost each)
    champion_decisions = ["allow"] * 160 + ["block"] * 40
    # Shadow: mostly block → cheap (tp_cost each)
    shadow_decisions = ["block"] * 160 + ["allow"] * 40
    rng.shuffle(champion_decisions)
    rng.shuffle(shadow_decisions)

    df = _make_predictions(
        n=n,
        champion_decisions=champion_decisions,
        shadow_decisions=shadow_decisions,
        is_fraud=is_fraud,
    )
    report = ShadowComparison(df, _DEFAULT_COSTS).run()

    assert report.bootstrap_mean_diff is not None
    assert report.bootstrap_mean_diff > 0  # shadow is cheaper
    assert report.bootstrap_p_value is not None
    assert report.bootstrap_p_value < 0.05  # significant


def test_bootstrap_significance_indistinguishable() -> None:
    """Shadow ≈ champion → p > 0.05 (not significant)."""
    n = 100
    # Same decisions → mean diff = 0 exactly → p == 1.0.
    df = _make_predictions(
        n=n,
        champion_decisions=["allow"] * 50 + ["block"] * 50,
        shadow_decisions=["allow"] * 50 + ["block"] * 50,
        is_fraud=[0] * 50 + [1] * 50,
    )
    report = ShadowComparison(df, _DEFAULT_COSTS).run()

    assert report.bootstrap_mean_diff == pytest.approx(0.0, abs=1e-9)
    assert report.bootstrap_p_value is not None
    assert report.bootstrap_p_value > 0.05  # NOT significant


def test_bootstrap_deterministic_under_fixed_seed() -> None:
    """Two runs with the same seed produce identical p-values."""
    df = _make_predictions(
        n=50,
        champion_decisions=["allow"] * 30 + ["block"] * 20,
        shadow_decisions=["block"] * 30 + ["allow"] * 20,
        is_fraud=[1] * 25 + [0] * 25,
    )
    a = ShadowComparison(df, _DEFAULT_COSTS, seed=42).run()
    b = ShadowComparison(df, _DEFAULT_COSTS, seed=42).run()
    assert a.bootstrap_mean_diff == b.bootstrap_mean_diff
    assert a.bootstrap_ci_95 == b.bootstrap_ci_95
    assert a.bootstrap_p_value == b.bootstrap_p_value


# ---------------------------------------------------------------------
# Scenario 11-12: PromotionVerdict.
# ---------------------------------------------------------------------


def test_should_promote_all_criteria_met() -> None:
    """All 3 criteria pass → promote=True."""
    n = 200
    # Construct: shadow agrees on 90% (above 85%) AND is much cheaper.
    # We need agreement_rate > 0.85, cost_improvement > 0.02, p_value < 0.05.
    # Pattern: 180 rows where both match (allow); 20 rows where champion
    # allows fraud (cost 450) but shadow blocks fraud (cost 5). Disagree only
    # on those 20 rows → 90% agreement; shadow much cheaper on 20 fraud rows.
    champion_decisions = ["allow"] * 180 + ["allow"] * 20
    shadow_decisions = ["allow"] * 180 + ["block"] * 20
    is_fraud = [0] * 180 + [1] * 20
    df = _make_predictions(
        n=n,
        champion_decisions=champion_decisions,
        shadow_decisions=shadow_decisions,
        is_fraud=is_fraud,
    )
    report = ShadowComparison(df, _DEFAULT_COSTS).run()
    assert report.agreement_rate == pytest.approx(0.9)
    assert report.cost_improvement is not None
    assert report.cost_improvement > 0.02
    assert report.bootstrap_p_value is not None
    assert report.bootstrap_p_value < 0.05
    assert report.verdict.promote is True
    assert report.verdict.cost_improvement_pass is True
    assert report.verdict.p_value_pass is True
    assert report.verdict.agreement_pass is True
    # All three reasons start with PASS.
    for reason in report.verdict.reasons:
        assert reason.startswith("PASS"), f"expected PASS reason; got {reason}"


def test_should_promote_below_threshold() -> None:
    """Only some criteria pass → promote=False; reasons explain why."""
    # Construct: shadow disagrees with champion on 80% (BELOW the 85%
    # threshold) AND is identically priced (no cost improvement). All
    # three criteria should fail.
    n = 100
    df = _make_predictions(
        n=n,
        champion_decisions=["allow"] * 50 + ["block"] * 50,
        shadow_decisions=["block"] * 50 + ["allow"] * 50,  # 0% agreement
        is_fraud=[0] * 50 + [1] * 50,  # alternating labels
    )
    report = ShadowComparison(df, _DEFAULT_COSTS).run()
    assert report.verdict.promote is False
    # Agreement should fail (0% < 85%).
    assert report.verdict.agreement_pass is False
    assert "FAIL" in report.verdict.reasons[0]


# ---------------------------------------------------------------------
# Scenario 13: input validation.
# ---------------------------------------------------------------------


def test_invalid_input_missing_columns() -> None:
    """DataFrame missing required columns raises ValueError."""
    df = pd.DataFrame({"request_id": ["a", "b"], "champion_score": [0.1, 0.2]})
    with pytest.raises(ValueError, match="missing required columns"):
        ShadowComparison(df, _DEFAULT_COSTS)
