"""Unit tests for `fraud_engine.features.tier1_basic`.

Five test surfaces:

- `TestAmountTransformerSpec`: spec-mandated assertions
  (`log1p(0)==0`, `log1p(e-1)≈1`, negative-amount rejection,
  decile range, fit-required guard).
- `TestTimeFeatureGeneratorSpec`: spec-mandated assertions
  (hour / dow ranges, `is_weekend` matches dow ∈ {5, 6},
  `is_business_hours` boundary cases).
- `TestPropertyBased`: hypothesis property `hour_sin² + hour_cos² ≈ 1`
  for every integer hour 0..23.
- `TestContractCompliance`: both generators satisfy
  `BaseFeatureGenerator` (feature names, non-empty rationale).
- `TestPipelineIntegration`: both generators slot into
  `FeaturePipeline.fit_transform` cleanly.

All inputs are inline synthetic frames; no real-data dependency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st

from fraud_engine.features import FeaturePipeline
from fraud_engine.features.tier1_basic import AmountTransformer, TimeFeatureGenerator

_SYNTHETIC_SEED: int = 42


def _synthetic_timestamps(n_rows: int = 100) -> pd.DataFrame:
    """Return a frame with one tz-aware `timestamp` column.

    Spans roughly a week of UTC datetimes so every hour and every
    weekday appears in the sample. Deterministic seed so tests are
    reproducible.
    """
    rng = np.random.default_rng(_SYNTHETIC_SEED)
    base = pd.Timestamp("2026-01-05 00:00:00+00:00")  # Monday
    minutes = rng.integers(0, 7 * 24 * 60, size=n_rows)
    timestamps = [base + pd.Timedelta(minutes=int(m)) for m in minutes]
    return pd.DataFrame({"timestamp": pd.to_datetime(timestamps, utc=True)})


def _synthetic_merged(n_rows: int = 200) -> pd.DataFrame:
    """Combine timestamps + a `TransactionAmt` column for pipeline tests."""
    rng = np.random.default_rng(_SYNTHETIC_SEED)
    base = _synthetic_timestamps(n_rows)
    base["TransactionAmt"] = rng.uniform(0.5, 1000.0, size=n_rows)
    return base


# ---------------- AmountTransformer spec ---------------- #


class TestAmountTransformerSpec:
    """Spec-mandated assertions for `AmountTransformer`."""

    def test_log_zero_is_zero(self) -> None:
        """`log1p(0) == 0` per spec."""
        gen = AmountTransformer().fit(pd.DataFrame({"TransactionAmt": [0.0, 1.0]}))
        out = gen.transform(pd.DataFrame({"TransactionAmt": [0.0]}))
        assert out["log_amount"].iloc[0] == 0.0

    def test_log_e_minus_one_is_one(self) -> None:
        """`log1p(e - 1) ≈ 1` per spec."""
        x = float(np.exp(1.0) - 1.0)
        gen = AmountTransformer().fit(pd.DataFrame({"TransactionAmt": np.linspace(0.1, 10.0, 30)}))
        out = gen.transform(pd.DataFrame({"TransactionAmt": [x]}))
        assert out["log_amount"].iloc[0] == pytest.approx(1.0, abs=1e-9)

    def test_negative_amount_raises_in_fit(self) -> None:
        """`fit` rejects negative amounts with a clear `ValueError`."""
        with pytest.raises(ValueError, match="negative amount"):
            AmountTransformer().fit(pd.DataFrame({"TransactionAmt": [1.0, -2.0, 3.0]}))

    def test_negative_amount_raises_in_transform(self) -> None:
        """`transform` re-checks negativity (defence-in-depth)."""
        gen = AmountTransformer().fit(pd.DataFrame({"TransactionAmt": [1.0, 2.0, 3.0]}))
        with pytest.raises(ValueError, match="negative amount"):
            gen.transform(pd.DataFrame({"TransactionAmt": [-1.0]}))

    def test_decile_in_zero_to_nine(self) -> None:
        """Output `amount_decile` always lies in 0..9."""
        df = pd.DataFrame({"TransactionAmt": np.linspace(0.0, 1000.0, 1000)})
        out = AmountTransformer().fit_transform(df)
        assert out["amount_decile"].between(0, 9).all()

    def test_transform_before_fit_raises(self) -> None:
        """Transform without prior fit raises `AttributeError`."""
        gen = AmountTransformer()
        with pytest.raises(AttributeError, match="must be fit"):
            gen.transform(pd.DataFrame({"TransactionAmt": [1.0]}))

    def test_decile_edges_persist_after_fit(self) -> None:
        """`fit` populates `decile_edges`; was None before."""
        gen = AmountTransformer()
        assert gen.decile_edges is None
        gen.fit(pd.DataFrame({"TransactionAmt": np.arange(1.0, 101.0)}))
        assert gen.decile_edges is not None
        assert len(gen.decile_edges) >= 2  # at least one bucket


# ---------------- TimeFeatureGenerator spec ---------------- #


class TestTimeFeatureGeneratorSpec:
    """Spec-mandated assertions for `TimeFeatureGenerator`."""

    def test_hour_of_day_in_range(self) -> None:
        """`hour_of_day` always lies in 0..23."""
        df = _synthetic_timestamps()
        out = TimeFeatureGenerator().fit_transform(df)
        assert out["hour_of_day"].between(0, 23).all()

    def test_day_of_week_in_range(self) -> None:
        """`day_of_week` always lies in 0..6 (Monday=0)."""
        df = _synthetic_timestamps()
        out = TimeFeatureGenerator().fit_transform(df)
        assert out["day_of_week"].between(0, 6).all()

    def test_is_weekend_matches_day_of_week(self) -> None:
        """`is_weekend == (day_of_week in {5, 6})` per spec."""
        df = _synthetic_timestamps()
        out = TimeFeatureGenerator().fit_transform(df)
        expected = out["day_of_week"].isin([5, 6]).astype(int)
        pd.testing.assert_series_equal(out["is_weekend"], expected, check_names=False)

    def test_is_business_hours_definition(self) -> None:
        """`is_business_hours` fires for 9 ≤ hour < 17 UTC."""
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2026-01-05 08:00:00+00:00",  # before
                        "2026-01-05 09:00:00+00:00",  # boundary, inside
                        "2026-01-05 16:59:00+00:00",  # boundary, inside
                        "2026-01-05 17:00:00+00:00",  # boundary, outside
                        "2026-01-05 23:59:00+00:00",  # after
                    ],
                    utc=True,
                )
            }
        )
        out = TimeFeatureGenerator().fit_transform(df)
        assert list(out["is_business_hours"]) == [0, 1, 1, 0, 0]


# ---------------- Property-based: unit-circle invariant ---------------- #


class TestPropertyBased:
    """Hypothesis-driven property tests for the cyclical encoding."""

    @given(hour=st.integers(min_value=0, max_value=23))
    def test_hour_sin_cos_unit_circle(self, hour: int) -> None:
        """`hour_sin² + hour_cos² ≈ 1` for every integer hour 0..23."""
        ts = pd.Timestamp(f"2026-01-01 {hour:02d}:00:00+00:00")
        df = pd.DataFrame({"timestamp": [ts]})
        out = TimeFeatureGenerator().fit_transform(df)
        squared_sum = out["hour_sin"].iloc[0] ** 2 + out["hour_cos"].iloc[0] ** 2
        assert squared_sum == pytest.approx(1.0, abs=1e-9)


# ---------------- Contract compliance ---------------- #


class TestContractCompliance:
    """Both generators satisfy `BaseFeatureGenerator` introspection."""

    def test_amount_feature_names(self) -> None:
        gen = AmountTransformer()
        assert gen.get_feature_names() == ["log_amount", "amount_decile"]

    def test_amount_rationale_non_empty(self) -> None:
        # 50-char floor catches a future "TODO" stub.
        assert len(AmountTransformer().get_business_rationale()) > 50

    def test_time_feature_names(self) -> None:
        names = TimeFeatureGenerator().get_feature_names()
        for expected in (
            "hour_of_day",
            "day_of_week",
            "is_weekend",
            "is_business_hours",
            "hour_sin",
            "hour_cos",
        ):
            assert expected in names
        assert len(names) == 6

    def test_time_rationale_non_empty(self) -> None:
        assert len(TimeFeatureGenerator().get_business_rationale()) > 50


# ---------------- Pipeline integration ---------------- #


class TestPipelineIntegration:
    """Both generators slot into `FeaturePipeline.fit_transform` cleanly."""

    def test_pipeline_fit_transform_chains(self) -> None:
        """A pipeline with both generators produces all 8 feature columns."""
        df = _synthetic_merged()
        pipe = FeaturePipeline(generators=[AmountTransformer(), TimeFeatureGenerator()])
        out = pipe.fit_transform(df)
        for col in (
            "log_amount",
            "amount_decile",
            "hour_of_day",
            "day_of_week",
            "is_weekend",
            "is_business_hours",
            "hour_sin",
            "hour_cos",
        ):
            assert col in out.columns
        # Original input columns also survive.
        assert "TransactionAmt" in out.columns
        assert "timestamp" in out.columns
