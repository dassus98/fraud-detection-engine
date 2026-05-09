"""Unit tests for `fraud_engine.evaluation.stratified.StratifiedEvaluator`.

Five contract surfaces (mirrors `test_economic.py`'s structure):

- `TestInit`: cost-model resolution, threshold default from Settings,
  `min_stratum_size` validation, range checks.
- `TestEvaluate`: long-format DataFrame contract; per-axis n_rows
  sums; missing-column → axis skipped with warning; `month=None` →
  month axis absent.
- `TestPerAxisLogic`: spec gate "synthetic imbalanced segments
  produce expected differential metrics" — four sub-gates plus the
  consolidated headline test.
- `TestPlotHeatmap`: `Axes` return; supplied-axes; cell annotations
  carry sample size; savefig smoke; unknown-metric raises; empty
  eval_df returns a placeholder.
- `TestErrorHandling`: shape mismatch, scores ∉ [0, 1], frame /
  month length mismatch, single-class stratum returns NaN.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # noqa: E402 — must precede any pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

from fraud_engine.config.settings import get_settings  # noqa: E402
from fraud_engine.evaluation.economic import EconomicCostModel  # noqa: E402
from fraud_engine.evaluation.stratified import StratifiedEvaluator  # noqa: E402

_SEED: int = 42

_RESULT_COLUMNS: tuple[str, ...] = (
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


# ---------------------------------------------------------------------
# Synthetic-data generators.
# ---------------------------------------------------------------------


def _small_fixture(  # noqa: PLR0915 — single-pass synthetic builder; splitting fragments the data-flow
    n: int = 4000,
    seed: int = _SEED,
) -> tuple[
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    pd.DataFrame,
    pd.Series[Any],
]:
    """Build a frame with controlled per-axis biases for the spec gates.

    - Amount: TransactionAmt ~ Uniform[0, 5000]. Fraud rate is 30 %
      below $50, 1 % above $1K, 10 % otherwise (gate (a)).
    - ProductCD: random pick of {W, C, R, H, S}. W produces clean
      separable scores (Beta(8, 2) vs Beta(2, 8)); C produces
      overlapping scores (Beta(2, 2) vs Beta(2, 3)) (gate (b)).
    - DeviceType: random pick of {mobile, desktop, None}. Neutral —
      no signal bias; just exercises the (null) bucket.
    - id_01: 30 % of rows have value 1.0; 70 % NaN. has-id rows
      carry the clean-signal scores from above; no-id rows have
      scores blurred toward 0.5 (gate (c)).
    - month: 50/50 split between 5 and 6. Month 5 gets additive
      Gaussian noise on top of the base scores; month 6 is clean
      (gate (d)).
    """
    rng = np.random.default_rng(seed)

    # Stratification fields.
    amt = rng.uniform(0.0, 5000.0, size=n)
    pcd = rng.choice(np.array(["W", "C", "R", "H", "S"]), size=n)
    dev = rng.choice(np.array(["mobile", "desktop", "(none)"]), size=n)
    # Convert "(none)" to actual NaN for the DeviceType column.
    dev_obj: list[Any] = [v if v != "(none)" else None for v in dev.tolist()]
    has_id = rng.uniform(size=n) < 0.30
    id_01 = np.where(has_id, 1.0, np.nan)
    month_arr = rng.choice(np.array([5, 6]), size=n)

    # Labels: amount-driven base rate.
    base_p = np.where(
        amt < 50.0,
        0.30,
        np.where(amt > 1000.0, 0.01, 0.10),
    )
    y = (rng.uniform(size=n) < base_p).astype(np.int64)

    # Base scores: separable per the y label.
    pos_clean = rng.beta(8.0, 2.0, size=n)
    neg_clean = rng.beta(2.0, 8.0, size=n)
    scores = np.where(y == 1, pos_clean, neg_clean)

    # ProductCD = "C" → overlapping scores.
    pos_overlap = rng.beta(2.0, 2.0, size=n)
    neg_overlap = rng.beta(2.0, 3.0, size=n)
    overlap_mask = pcd == "C"
    scores = np.where(
        overlap_mask,
        np.where(y == 1, pos_overlap, neg_overlap),
        scores,
    )

    # no_identity rows → blur toward 0.5.
    blur = 0.30 * rng.uniform(size=n)
    blurred = scores * (1.0 - blur) + 0.5 * blur
    scores = np.where(~has_id, blurred, scores)

    # month = 5 → add Gaussian noise; clip to [0.001, 0.999].
    noise = 0.10 * rng.normal(size=n)
    noisy = np.clip(scores + noise, 0.001, 0.999)
    scores = np.where(month_arr == 5, noisy, scores)

    scores = scores.astype(np.float64)

    frame = pd.DataFrame(
        {
            "TransactionAmt": amt,
            "ProductCD": pcd,
            "DeviceType": pd.Series(dev_obj, dtype=object),
            "id_01": id_01,
        }
    )
    month_series = pd.Series(month_arr, name="month")

    return y, scores, frame, month_series


def _amount_only_fixture(
    n: int = 4000,
    seed: int = _SEED,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], pd.DataFrame]:
    """Frame with only TransactionAmt; fraud rate scales by bucket.

    Used by the amount-axis-only test to isolate the gate-(a) signal.
    """
    rng = np.random.default_rng(seed)
    amt = rng.uniform(0.0, 5000.0, size=n)
    base_p = np.where(amt < 50.0, 0.30, np.where(amt > 1000.0, 0.01, 0.10))
    y = (rng.uniform(size=n) < base_p).astype(np.int64)
    scores = np.where(y == 1, rng.beta(8.0, 2.0, size=n), rng.beta(2.0, 8.0, size=n)).astype(
        np.float64
    )
    frame = pd.DataFrame({"TransactionAmt": amt})
    return y, scores, frame


# ---------------------------------------------------------------------
# `TestInit`.
# ---------------------------------------------------------------------


class TestInit:
    """Construction: cost-model + threshold + min_stratum_size validation."""

    def test_default_cost_model_resolves_to_economic_default(self) -> None:
        """No-args constructor uses an `EconomicCostModel()` (Settings defaults)."""
        get_settings.cache_clear()
        settings = get_settings()
        evaluator = StratifiedEvaluator()
        assert evaluator.cost_model.costs["fraud_cost"] == settings.fraud_cost_usd
        assert evaluator.cost_model.costs["fp_cost"] == settings.fp_cost_usd

    def test_default_threshold_resolves_from_settings(self) -> None:
        """No-args threshold uses `Settings.decision_threshold`."""
        get_settings.cache_clear()
        settings = get_settings()
        evaluator = StratifiedEvaluator()
        assert evaluator.threshold == settings.decision_threshold

    def test_explicit_cost_model_overrides_default(self) -> None:
        """Custom `EconomicCostModel` is stored on the instance."""
        custom = EconomicCostModel(fraud_cost=999.0, fp_cost=88.0)
        evaluator = StratifiedEvaluator(cost_model=custom)
        assert evaluator.cost_model is custom
        assert evaluator.cost_model.costs["fraud_cost"] == 999.0

    def test_explicit_threshold_overrides_default(self) -> None:
        """Caller-supplied threshold wins over Settings."""
        evaluator = StratifiedEvaluator(threshold=0.7)
        assert evaluator.threshold == 0.7

    def test_explicit_min_stratum_size_stored(self) -> None:
        """`min_stratum_size` kwarg is stored and exposed via the property."""
        evaluator = StratifiedEvaluator(min_stratum_size=100)
        assert evaluator.min_stratum_size == 100

    def test_threshold_above_one_raises(self) -> None:
        """Threshold > 1 violates the [0, 1] contract."""
        with pytest.raises(ValueError, match="threshold"):
            StratifiedEvaluator(threshold=1.5)

    def test_threshold_below_zero_raises(self) -> None:
        """Negative threshold violates the [0, 1] contract."""
        with pytest.raises(ValueError, match="threshold"):
            StratifiedEvaluator(threshold=-0.1)

    def test_min_stratum_size_zero_raises(self) -> None:
        """`min_stratum_size` must be >= 1."""
        with pytest.raises(ValueError, match="min_stratum_size"):
            StratifiedEvaluator(min_stratum_size=0)

    def test_min_stratum_size_negative_raises(self) -> None:
        """Negative `min_stratum_size` raises."""
        with pytest.raises(ValueError, match="min_stratum_size"):
            StratifiedEvaluator(min_stratum_size=-5)


# ---------------------------------------------------------------------
# `TestEvaluate`.
# ---------------------------------------------------------------------


class TestEvaluate:
    """Top-level `evaluate(...)` contract: shape, sums, skip-with-warning."""

    def test_returns_long_format_dataframe_with_expected_columns(self) -> None:
        """Returned DataFrame has the long-format `_RESULT_COLUMNS` shape."""
        y, s, frame, month = _small_fixture()
        out = StratifiedEvaluator().evaluate(y, s, frame, month=month)
        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == list(_RESULT_COLUMNS)

    def test_stratum_axis_values_are_subset_of_known_axes(self) -> None:
        """`stratum_axis` only contains the five canonical axes."""
        y, s, frame, month = _small_fixture()
        out = StratifiedEvaluator().evaluate(y, s, frame, month=month)
        known = {"amount_bucket", "product_cd", "device_type", "identity_coverage", "month"}
        assert set(out["stratum_axis"].unique()) <= known

    def test_n_rows_per_axis_sums_to_total(self) -> None:
        """For every axis present, `n_rows` rows sum to `len(y_true)`."""
        y, s, frame, month = _small_fixture()
        out = StratifiedEvaluator().evaluate(y, s, frame, month=month)
        for axis in out["stratum_axis"].unique():
            axis_rows = out[out["stratum_axis"] == axis]
            assert int(axis_rows["n_rows"].sum()) == len(y)

    def test_fraud_rate_equals_n_pos_over_n_rows(self) -> None:
        """Identity: every non-empty stratum has `fraud_rate == n_pos / n_rows`."""
        y, s, frame, month = _small_fixture()
        out = StratifiedEvaluator().evaluate(y, s, frame, month=month)
        non_empty = out[out["n_rows"] > 0]
        expected = non_empty["n_pos"] / non_empty["n_rows"]
        np.testing.assert_allclose(
            non_empty["fraud_rate"].to_numpy(),
            expected.to_numpy(),
            rtol=1e-9,
        )

    def test_amount_bucket_emits_five_strata(self) -> None:
        """All five amount buckets appear in the output (synthetic spans 0-5K)."""
        y, s, frame = _amount_only_fixture()
        out = StratifiedEvaluator().evaluate(y, s, frame)
        amount_strata = set(out[out["stratum_axis"] == "amount_bucket"]["stratum_value"].tolist())
        assert amount_strata == {"<$50", "$50-200", "$200-500", "$500-1K", ">$1K"}

    def test_missing_column_skips_axis_with_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Frame missing `ProductCD` → no `product_cd` rows; warning logged."""
        y, s, frame, month = _small_fixture()
        frame_no_pcd = frame.drop(columns=["ProductCD"])
        with caplog.at_level("WARNING"):
            out = StratifiedEvaluator().evaluate(y, s, frame_no_pcd, month=month)
        assert "product_cd" not in set(out["stratum_axis"].unique())
        assert any(
            "axis_skipped" in rec.message or "product_cd" in rec.message for rec in caplog.records
        )

    def test_month_none_skips_month_axis(self) -> None:
        """`month=None` → no `month` rows in output."""
        y, s, frame, _ = _small_fixture()
        out = StratifiedEvaluator().evaluate(y, s, frame, month=None)
        assert "month" not in set(out["stratum_axis"].unique())

    def test_axis_order_follows_canonical(self) -> None:
        """Axes appear in the canonical `_AXES` order in the output."""
        y, s, frame, month = _small_fixture()
        out = StratifiedEvaluator().evaluate(y, s, frame, month=month)
        # Strip duplicates while preserving first-occurrence order.
        seen = list(dict.fromkeys(out["stratum_axis"].tolist()))
        canonical = ["amount_bucket", "product_cd", "device_type", "identity_coverage", "month"]
        # Output is a subset of canonical (depends on which axes ran),
        # in the same relative order.
        assert seen == [a for a in canonical if a in seen]


# ---------------------------------------------------------------------
# `TestPerAxisLogic` (the spec gate).
# ---------------------------------------------------------------------


class TestPerAxisLogic:
    """Spec gate: synthetic imbalanced segments produce expected differential metrics."""

    def test_low_amount_bucket_has_higher_fraud_rate(self) -> None:
        """Gate (a): fraud_rate(<$50) > fraud_rate(>$1K) by margin > 0.20."""
        y, s, frame = _amount_only_fixture()
        out = StratifiedEvaluator().evaluate(y, s, frame)
        by_amt = out[out["stratum_axis"] == "amount_bucket"].set_index("stratum_value")
        low = float(by_amt.loc["<$50", "fraud_rate"])
        high = float(by_amt.loc[">$1K", "fraud_rate"])
        assert low - high > 0.20

    def test_separable_product_has_higher_auc_than_overlapping(self) -> None:
        """Gate (b): ProductCD with separable signal beats overlapping by > 0.10 AUC."""
        y, s, frame, month = _small_fixture()
        out = StratifiedEvaluator().evaluate(y, s, frame, month=month)
        by_prod = out[out["stratum_axis"] == "product_cd"].set_index("stratum_value")
        # W has clean Beta(8,2)/Beta(2,8); C has overlapping Beta(2,2)/Beta(2,3).
        assert by_prod.loc["W", "auc"] > by_prod.loc["C", "auc"] + 0.10

    def test_has_identity_has_higher_auc_than_no_identity(self) -> None:
        """Gate (c): has_identity AUC > no_identity AUC (no_identity scores are blurred)."""
        y, s, frame, month = _small_fixture()
        out = StratifiedEvaluator().evaluate(y, s, frame, month=month)
        by_id = out[out["stratum_axis"] == "identity_coverage"].set_index("stratum_value")
        assert by_id.loc["has_identity", "auc"] > by_id.loc["no_identity", "auc"]

    def test_month_with_drift_has_higher_cost_per_txn(self) -> None:
        """Gate (d): month=5 (noisy scores) has higher cost_per_txn than month=6 (clean)."""
        y, s, frame, month = _small_fixture()
        out = StratifiedEvaluator().evaluate(y, s, frame, month=month)
        by_month = out[out["stratum_axis"] == "month"].set_index("stratum_value")
        assert by_month.loc["5", "cost_per_txn"] > by_month.loc["6", "cost_per_txn"]

    def test_imbalanced_segments_produce_expected_differential_metrics(self) -> None:
        """Spec headline gate consolidating gates (a), (b), (c)."""
        y, s, frame, month = _small_fixture()
        out = StratifiedEvaluator().evaluate(y, s, frame, month=month)

        by_amt = out[out["stratum_axis"] == "amount_bucket"].set_index("stratum_value")
        assert by_amt.loc["<$50", "fraud_rate"] > by_amt.loc[">$1K", "fraud_rate"] + 0.20

        by_prod = out[out["stratum_axis"] == "product_cd"].set_index("stratum_value")
        assert by_prod.loc["W", "auc"] > by_prod.loc["C", "auc"] + 0.10

        by_id = out[out["stratum_axis"] == "identity_coverage"].set_index("stratum_value")
        assert by_id.loc["has_identity", "auc"] > by_id.loc["no_identity", "auc"]

    def test_amount_bucket_edge_lands_in_upper_interval(self) -> None:
        """Boundary semantics: $50.0 lands in `$50-200`, not `<$50` (half-open)."""
        # Construct a tiny frame with one row exactly at $50.0.
        amt = np.array([50.0, 25.0, 100.0])
        y = np.array([0, 0, 0])
        s = np.array([0.5, 0.5, 0.5])
        frame = pd.DataFrame({"TransactionAmt": amt})
        out = StratifiedEvaluator(min_stratum_size=1).evaluate(y, s, frame)
        by_amt = out[out["stratum_axis"] == "amount_bucket"].set_index("stratum_value")
        # Row at $50.0 should land in `$50-200`, not `<$50`.
        assert int(by_amt.loc["<$50", "n_rows"]) == 1  # only the $25 row
        assert int(by_amt.loc["$50-200", "n_rows"]) == 2  # the $50 and $100 rows

    def test_device_type_null_group_uses_explicit_label(self) -> None:
        """DeviceType NaN rows produce a stratum labelled `(null)`."""
        y, s, frame, month = _small_fixture()
        out = StratifiedEvaluator().evaluate(y, s, frame, month=month)
        device_strata = set(out[out["stratum_axis"] == "device_type"]["stratum_value"].tolist())
        # Synthetic uses None for NaN DeviceType — should produce (null) bucket.
        assert "(null)" in device_strata

    def test_identity_coverage_emits_exactly_two_groups(self) -> None:
        """`identity_coverage` axis has exactly `has_identity` + `no_identity` rows."""
        y, s, frame, month = _small_fixture()
        out = StratifiedEvaluator().evaluate(y, s, frame, month=month)
        id_strata = set(out[out["stratum_axis"] == "identity_coverage"]["stratum_value"].tolist())
        assert id_strata == {"has_identity", "no_identity"}


# ---------------------------------------------------------------------
# `TestPlotHeatmap`.
# ---------------------------------------------------------------------


class TestPlotHeatmap:
    """`plot_heatmap` Axes return + supplied-axes + cell annotation contract."""

    def test_returns_axes(self) -> None:
        """Plot function returns a matplotlib `Axes`."""
        y, s, frame, month = _small_fixture()
        eval_df = StratifiedEvaluator().evaluate(y, s, frame, month=month)
        ax = StratifiedEvaluator().plot_heatmap(eval_df)
        assert isinstance(ax, Axes)
        plt.close(ax.figure)

    def test_uses_supplied_axes(self) -> None:
        """When `ax` is passed, the function plots on it (no new figure)."""
        y, s, frame, month = _small_fixture()
        eval_df = StratifiedEvaluator().evaluate(y, s, frame, month=month)
        fig, ax_in = plt.subplots()
        ax_out = StratifiedEvaluator().plot_heatmap(eval_df, ax=ax_in)
        assert ax_out is ax_in
        plt.close(fig)

    def test_cell_annotations_include_sample_size(self) -> None:
        """Cell annotations carry sample-size labels (`n=...`)."""
        y, s, frame, month = _small_fixture()
        eval_df = StratifiedEvaluator().evaluate(y, s, frame, month=month)
        ax = StratifiedEvaluator().plot_heatmap(eval_df)
        texts = [t.get_text() for t in ax.texts]
        assert any("n=" in t for t in texts)
        plt.close(ax.figure)

    def test_cost_cell_annotation_includes_dollar_sign(self) -> None:
        """Cost-column cells are formatted with a `$` prefix."""
        y, s, frame, month = _small_fixture()
        eval_df = StratifiedEvaluator().evaluate(y, s, frame, month=month)
        ax = StratifiedEvaluator().plot_heatmap(eval_df, metrics=("cost_per_txn",))
        texts = [t.get_text() for t in ax.texts]
        # At least one annotation has a $-prefixed numeric.
        assert any("$" in t for t in texts)
        plt.close(ax.figure)

    def test_savefig_smoke(self, tmp_path: Path) -> None:
        """`fig.savefig(...)` writes a non-empty file."""
        y, s, frame, month = _small_fixture()
        eval_df = StratifiedEvaluator().evaluate(y, s, frame, month=month)
        ax = StratifiedEvaluator().plot_heatmap(eval_df)
        out = tmp_path / "heatmap.png"
        ax.figure.savefig(out)
        assert out.is_file()
        assert out.stat().st_size > 0
        plt.close(ax.figure)

    def test_unknown_metric_raises(self) -> None:
        """Metric not present as a column in `eval_df` raises `ValueError`."""
        y, s, frame, month = _small_fixture()
        eval_df = StratifiedEvaluator().evaluate(y, s, frame, month=month)
        with pytest.raises(ValueError, match="unknown metric"):
            StratifiedEvaluator().plot_heatmap(eval_df, metrics=("nonexistent",))

    def test_empty_eval_df_returns_axes_placeholder(self) -> None:
        """Empty `eval_df` (every axis skipped) returns Axes with a message."""
        empty = pd.DataFrame(columns=list(_RESULT_COLUMNS))
        ax = StratifiedEvaluator().plot_heatmap(empty)
        assert isinstance(ax, Axes)
        plt.close(ax.figure)


# ---------------------------------------------------------------------
# `TestErrorHandling`.
# ---------------------------------------------------------------------


class TestErrorHandling:
    """Validation: shape mismatches, score-range guard, single-class strata."""

    def test_y_scores_outside_unit_interval_raises(self) -> None:
        """`y_scores ∉ [0, 1]` raises (Calibrator-contract guard)."""
        y, _, frame, month = _small_fixture(n=200)
        # Scores deliberately out of [0, 1].
        bad_scores = np.full(200, fill_value=1.5, dtype=np.float64)
        with pytest.raises(ValueError, match=r"y_scores must be in \[0, 1\]"):
            StratifiedEvaluator().evaluate(y, bad_scores, frame, month=month)

    def test_y_scores_negative_raises(self) -> None:
        """Negative scores raise the same guard."""
        y, _, frame, _ = _small_fixture(n=200)
        bad_scores = np.full(200, fill_value=-0.1, dtype=np.float64)
        with pytest.raises(ValueError, match=r"y_scores must be in \[0, 1\]"):
            StratifiedEvaluator().evaluate(y, bad_scores, frame)

    def test_y_true_y_scores_shape_mismatch_raises(self) -> None:
        """`y_true` / `y_scores` length mismatch raises."""
        evaluator = StratifiedEvaluator()
        with pytest.raises(ValueError, match="shape mismatch"):
            evaluator.evaluate(
                np.array([0, 1, 0, 1, 0]),
                np.array([0.1, 0.5, 0.9, 0.2]),
                pd.DataFrame({"TransactionAmt": [10.0, 20.0, 30.0, 40.0, 50.0]}),
            )

    def test_frame_length_mismatch_raises(self) -> None:
        """Frame with wrong length raises."""
        y, s, _, _ = _small_fixture(n=100)
        smaller_frame = pd.DataFrame({"TransactionAmt": np.zeros(50)})
        with pytest.raises(ValueError, match="frame length"):
            StratifiedEvaluator().evaluate(y, s, smaller_frame)

    def test_month_length_mismatch_raises(self) -> None:
        """`month` Series with wrong length raises."""
        y, s, frame, _ = _small_fixture(n=100)
        bad_month = pd.Series([5] * 50)
        with pytest.raises(ValueError, match="month series length"):
            StratifiedEvaluator().evaluate(y, s, frame, month=bad_month)

    def test_single_class_stratum_returns_nan_with_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Single-class stratum produces NaN AUC/PR-AUC; warning logged."""
        # Construct a frame where the `<$50` bucket has only y=0 rows.
        n = 500
        rng = np.random.default_rng(_SEED)
        amt = rng.uniform(0.0, 5000.0, size=n)
        y = (amt > 50.0).astype(np.int64)  # y=1 iff amount > 50 → <$50 bucket has all y=0
        # Force <$50 to have at least 50 rows so it isn't skipped for size.
        small_idx = rng.choice(np.where(amt > 50.0)[0], size=60, replace=False)
        amt[small_idx] = 25.0
        y[small_idx] = 0
        scores = np.where(y == 1, rng.beta(8.0, 2.0, size=n), rng.beta(2.0, 8.0, size=n)).astype(
            np.float64
        )
        frame = pd.DataFrame({"TransactionAmt": amt})

        with caplog.at_level("WARNING"):
            out = StratifiedEvaluator(min_stratum_size=10).evaluate(y, scores, frame)
        by_amt = out[out["stratum_axis"] == "amount_bucket"].set_index("stratum_value")
        # `<$50` is single-class (all y=0) → NaN AUC, NaN PR-AUC.
        assert pd.isna(by_amt.loc["<$50", "auc"])
        assert pd.isna(by_amt.loc["<$50", "pr_auc"])
        # Cost is still computed (well-defined on single class).
        assert pd.notna(by_amt.loc["<$50", "cost_per_txn"])
        # Degenerate-stratum warning logged.
        assert any("degenerate_stratum" in rec.message for rec in caplog.records)
