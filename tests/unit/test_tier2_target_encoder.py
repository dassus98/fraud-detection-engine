"""Unit tests for `fraud_engine.features.tier2_aggregations.TargetEncoder`.

Five contract surfaces:

- `TestSmoothingFormula`: hand-computed full-encoder values; α=0
  reduces to raw rates; α → ∞ collapses to the global rate.
- `TestOOFCorrectness`: each training row's encoded value is
  derivable from a fold that does NOT contain the row; full-train
  encoder is fit alongside the OOF pass; unseen-category fallback.
- `TestPipelineIntegration`: `FeaturePipeline.fit_transform` engages
  the OOF override (post-2.2.d 1-line fix); existing generators'
  default behaviour is unchanged.
- `TestConfigLoad`: default YAML loads correctly; explicit kwargs
  override.
- `TestErrorHandling`: `transform` before `fit` raises `AttributeError`.
"""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from fraud_engine.features import BaseFeatureGenerator, FeaturePipeline
from fraud_engine.features.tier2_aggregations import TargetEncoder

# ---------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------


def _make_synthetic_frame(n: int = 100, seed: int = 0) -> pd.DataFrame:
    """Build a balanced synthetic frame with cat ∈ {A, B, C, None}."""
    rng = np.random.default_rng(seed)
    cats: list[str | None] = list(rng.choice(["A", "B", "C", None], size=n))
    fraud = rng.integers(0, 2, size=n)
    # Force at least 5 fraud + 5 non-fraud rows so StratifiedKFold(n=5) works.
    fraud[:5] = 1
    fraud[5:10] = 0
    return pd.DataFrame(
        {
            "TransactionDT": np.arange(n, dtype=np.int64),
            "cat": cats,
            "isFraud": fraud,
        }
    )


# ---------------------------------------------------------------------
# Smoothing formula.
# ---------------------------------------------------------------------


class TestSmoothingFormula:
    """Smoothing math: hand-computed values; α=0 / α→∞ limits."""

    def test_hand_computed_full_encoder(self) -> None:
        """Full-train encoded value matches the hand-computed formula."""
        df = pd.DataFrame(
            {
                "cat": ["A", "A", "A", "B", "B", "B"],
                "isFraud": [1, 1, 0, 0, 0, 0],
            }
        )
        # α=2, global_rate = 2/6 = 0.3333
        # 'A': sum=2, count=3 → (2 + 2 × 0.3333) / (3 + 2) = 2.6667 / 5 = 0.5333
        # 'B': sum=0, count=3 → (0 + 2 × 0.3333) / (3 + 2) = 0.6667 / 5 = 0.1333
        gen = TargetEncoder(cat_cols=["cat"], target_col="isFraud", alpha=2.0, n_splits=2)
        gen.fit(df)
        assert gen.mappings_ is not None
        global_rate = float(df["isFraud"].mean())
        expected_a = (2 + 2.0 * global_rate) / (3 + 2.0)
        expected_b = (0 + 2.0 * global_rate) / (3 + 2.0)
        assert gen.mappings_["cat"]["A"] == pytest.approx(expected_a)
        assert gen.mappings_["cat"]["B"] == pytest.approx(expected_b)

    def test_alpha_zero_yields_raw_rate(self) -> None:
        """With α=0, encoded value == raw fraud rate per category."""
        df = pd.DataFrame(
            {
                "cat": ["A", "A", "A", "B", "B", "B"],
                "isFraud": [1, 1, 0, 0, 1, 0],
            }
        )
        gen = TargetEncoder(cat_cols=["cat"], target_col="isFraud", alpha=0.0, n_splits=2)
        gen.fit(df)
        assert gen.mappings_ is not None
        # Raw rates: A → 2/3, B → 1/3.
        assert gen.mappings_["cat"]["A"] == pytest.approx(2 / 3)
        assert gen.mappings_["cat"]["B"] == pytest.approx(1 / 3)

    def test_alpha_infinity_yields_global_rate(self) -> None:
        """α → ∞ collapses the full-train encoded value to the global rate.

        Tests the full-train encoder (`fit` + `transform`) rather than
        OOF (`fit_transform`) because OOF folds each carry their own
        fold-specific global_rate that differs from the full-train rate
        by `O(1/sqrt(n_per_fold))` — a real and intentional difference.
        The α → ∞ limit is a property of the encoder math, cleanest to
        verify on the single-rate full-train path.
        """
        df = _make_synthetic_frame(n=80, seed=1)
        global_rate = float(df["isFraud"].mean())
        gen = TargetEncoder(
            cat_cols=["cat"],
            target_col="isFraud",
            alpha=1e9,
            n_splits=5,
            random_state=0,
        )
        gen.fit(df)
        out = gen.transform(df)
        for value in out["cat_target_enc"]:
            assert value == pytest.approx(global_rate, abs=1e-6)


# ---------------------------------------------------------------------
# OOF correctness — the strict-correctness suite.
# ---------------------------------------------------------------------


class TestOOFCorrectness:
    """Each training row's encoded value uses a fold that does NOT contain it."""

    def test_oof_excludes_self_fold(self) -> None:
        """`fit_transform` row-by-row matches a hand-rolled OOF re-derivation."""
        df = _make_synthetic_frame(n=60, seed=2)
        seed = 7
        alpha = 5.0
        n_splits = 5

        gen = TargetEncoder(
            cat_cols=["cat"],
            target_col="isFraud",
            alpha=alpha,
            n_splits=n_splits,
            random_state=seed,
        )
        out = gen.fit_transform(df)

        # Re-derive expected OOF values using the same StratifiedKFold.
        expected = np.full(len(df), np.nan)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        targets = df["isFraud"].to_numpy()
        for other_idx, oof_idx in skf.split(np.zeros(len(df)), targets):
            other_df = df.iloc[other_idx]
            fold_global_rate = float(other_df["isFraud"].mean())
            grouped = other_df.groupby("cat", dropna=False)["isFraud"]
            counts = grouped.count()
            sums = grouped.sum()
            mapping_series = (sums + alpha * fold_global_rate) / (counts + alpha)
            mapping = {k: float(v) for k, v in mapping_series.items()}

            for original_pos, cat_val in zip(
                oof_idx, df.iloc[oof_idx]["cat"].to_numpy(), strict=True
            ):
                if pd.isna(cat_val):
                    found = False
                    for k, v in mapping.items():
                        if pd.isna(k):
                            expected[int(original_pos)] = v
                            found = True
                            break
                    if not found:
                        expected[int(original_pos)] = fold_global_rate
                else:
                    expected[int(original_pos)] = mapping.get(cat_val, fold_global_rate)

        actual = out["cat_target_enc"].to_numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12)

    def test_full_encoder_fit_after_oof(self) -> None:
        """`fit_transform(df)` populates `mappings_` matching `fit(df)`."""
        df = _make_synthetic_frame(n=60, seed=3)

        gen_ft = TargetEncoder(
            cat_cols=["cat"], target_col="isFraud", alpha=5.0, n_splits=5, random_state=0
        )
        gen_ft.fit_transform(df)

        gen_fit = TargetEncoder(
            cat_cols=["cat"], target_col="isFraud", alpha=5.0, n_splits=5, random_state=0
        )
        gen_fit.fit(df)

        assert gen_ft.mappings_ is not None
        assert gen_fit.mappings_ is not None
        assert gen_ft.global_rates_ == gen_fit.global_rates_
        # Mappings agree on every key (NaN keys handled via pd.isna pairing).
        for col in ("cat",):
            ft_map = gen_ft.mappings_[col]
            fit_map = gen_fit.mappings_[col]
            assert set(str(k) for k in ft_map) == set(str(k) for k in fit_map)
            for key in ft_map:
                if pd.isna(key):
                    nan_value_ft = next(v for k, v in ft_map.items() if pd.isna(k))
                    nan_value_fit = next(v for k, v in fit_map.items() if pd.isna(k))
                    assert nan_value_ft == pytest.approx(nan_value_fit)
                else:
                    assert ft_map[key] == pytest.approx(fit_map[key])

    def test_unseen_category_at_transform_yields_global_rate(self) -> None:
        """An unseen category at `transform` time encodes to the global rate."""
        train = pd.DataFrame({"cat": ["A", "A", "B", "B", "A", "B"], "isFraud": [1, 0, 0, 1, 0, 0]})
        unseen = pd.DataFrame({"cat": ["UNKNOWN_X"]})
        gen = TargetEncoder(
            cat_cols=["cat"], target_col="isFraud", alpha=10.0, n_splits=2, random_state=0
        )
        gen.fit(train)
        out = gen.transform(unseen)

        assert gen.global_rates_ is not None
        assert out["cat_target_enc"].iloc[0] == pytest.approx(gen.global_rates_["cat"])

    def test_nan_category_treated_as_own_group(self) -> None:
        """NaN cat is its own group with its own encoded value, distinct from other categories."""
        # 4 NaN-cat with 3/4 fraud; 4 'A'-cat with 1/4 fraud.
        df = pd.DataFrame(
            {
                "cat": [None, None, None, None, "A", "A", "A", "A"],
                "isFraud": [1, 1, 0, 1, 0, 0, 0, 1],
            }
        )
        gen = TargetEncoder(
            cat_cols=["cat"], target_col="isFraud", alpha=10.0, n_splits=2, random_state=0
        )
        gen.fit(df)
        assert gen.mappings_ is not None

        # Fetch the NaN-key encoding.
        nan_value = next(v for k, v in gen.mappings_["cat"].items() if pd.isna(k))
        a_value = gen.mappings_["cat"]["A"]
        # NaN group has higher fraud rate than 'A', and they're different.
        assert nan_value > a_value
        assert nan_value != pytest.approx(a_value, abs=1e-6)


# ---------------------------------------------------------------------
# Pipeline integration — verifies the 2.2.d `pipeline.py` 1-line fix.
# ---------------------------------------------------------------------


class _MeanCenter(BaseFeatureGenerator):
    """Stub generator (mean-centering); used to assert the pipeline change is identity-preserving."""

    def __init__(self, col: str) -> None:
        self.col = col
        self._mean: float | None = None

    def fit(self, df: pd.DataFrame) -> Self:
        self._mean = float(df[self.col].mean())
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._mean is None:
            raise AttributeError("_MeanCenter must be fit before transform")
        out = df.copy()
        out[f"{self.col}_centred"] = df[self.col] - self._mean
        return out

    def get_feature_names(self) -> list[str]:
        return [f"{self.col}_centred"]

    def get_business_rationale(self) -> str:
        return "Mean-centred stub for pipeline integration tests."


class TestPipelineIntegration:
    """`FeaturePipeline.fit_transform` engages generator polymorphism."""

    def test_pipeline_fit_transform_engages_oof_override(self) -> None:
        """Running TargetEncoder inside a pipeline produces OOF (not full-train) encoding."""
        df = _make_synthetic_frame(n=60, seed=4)
        seed = 11

        # Direct call.
        direct = TargetEncoder(
            cat_cols=["cat"], target_col="isFraud", alpha=5.0, n_splits=5, random_state=seed
        )
        direct_out = direct.fit_transform(df)

        # Via pipeline.
        pipe = FeaturePipeline(
            generators=[
                TargetEncoder(
                    cat_cols=["cat"],
                    target_col="isFraud",
                    alpha=5.0,
                    n_splits=5,
                    random_state=seed,
                )
            ]
        )
        pipe_out = pipe.fit_transform(df)

        # OOF is deterministic given seed → identical outputs.
        np.testing.assert_allclose(
            pipe_out["cat_target_enc"].to_numpy(),
            direct_out["cat_target_enc"].to_numpy(),
            rtol=1e-12,
        )

    def test_existing_generators_unchanged_under_pipeline_fix(self) -> None:
        """Stub generator's pipeline output equals direct `fit().transform()` chain."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})

        # Direct chain (the OLD pipeline semantics).
        gen = _MeanCenter("x")
        direct_out = gen.fit(df).transform(df)

        # Pipeline (NEW semantics — calls `gen.fit_transform`).
        pipe = FeaturePipeline(generators=[_MeanCenter("x")])
        pipe_out = pipe.fit_transform(df)

        pd.testing.assert_frame_equal(direct_out, pipe_out)


# ---------------------------------------------------------------------
# Config loading.
# ---------------------------------------------------------------------


class TestConfigLoad:
    """Default YAML loads; explicit kwargs override."""

    def test_default_config_loads(self) -> None:
        gen = TargetEncoder()
        assert gen.cat_cols == ("card4", "addr1", "P_emaildomain")
        assert gen.target_col == "isFraud"
        assert gen.alpha == pytest.approx(10.0)
        assert gen.n_splits == 5
        assert gen.get_feature_names() == [
            "card4_target_enc",
            "addr1_target_enc",
            "P_emaildomain_target_enc",
        ]

    def test_constructor_overrides_config(self) -> None:
        gen = TargetEncoder(
            cat_cols=["card4"],
            target_col="isFraud",
            alpha=3.0,
            n_splits=3,
            random_state=99,
        )
        assert gen.cat_cols == ("card4",)
        assert gen.alpha == pytest.approx(3.0)
        assert gen.n_splits == 3
        assert gen.random_state == 99
        assert gen.get_feature_names() == ["card4_target_enc"]


# ---------------------------------------------------------------------
# Error handling.
# ---------------------------------------------------------------------


class TestErrorHandling:
    """`transform` before `fit` raises with a clear message."""

    def test_transform_before_fit_raises(self) -> None:
        gen = TargetEncoder(cat_cols=["cat"], target_col="isFraud", alpha=10.0, n_splits=2)
        df = pd.DataFrame({"cat": ["A", "B"]})
        with pytest.raises(AttributeError, match="fit before transform"):
            gen.transform(df)
