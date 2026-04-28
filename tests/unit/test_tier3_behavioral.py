"""Unit tests for `fraud_engine.features.tier3_behavioral`.

Four contract surfaces:

- `TestBehavioralDeviation`: hand-computed z-scores / flags / hour
  deviation; first-event / single-prior fallbacks; tied-timestamp
  semantics; NaN-input defensiveness.
- `TestColdStartHandler`: first-event coldstart; warm after
  `min_history`; multi-entity output.
- `TestTemporalSafety`: `assert_no_future_leak` passes for both
  classes' headline columns.
- `TestConfigLoad`: default YAMLs load; no-arg constructors get the
  spec-correct defaults.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from fraud_engine.features.temporal_guards import assert_no_future_leak
from fraud_engine.features.tier3_behavioral import (
    BehavioralDeviation,
    ColdStartHandler,
)

# ---------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------


def _build_frame(  # noqa: PLR0913 — six explicit kwargs mirror the cleaner-output column set.
    *,
    timestamps: list[int],
    cards: list[Any],
    amounts: list[float],
    addrs: list[Any],
    devices: list[Any],
    hours: list[int],
) -> pd.DataFrame:
    """Build a minimal cleaner-shaped frame for the deviation tests."""
    return pd.DataFrame(
        {
            "TransactionDT": timestamps,
            "card1": cards,
            "TransactionAmt": amounts,
            "addr1": addrs,
            "DeviceInfo": devices,
            "hour_of_day": hours,
        }
    )


# ---------------------------------------------------------------------
# `BehavioralDeviation`.
# ---------------------------------------------------------------------


class TestBehavioralDeviation:
    """Per-card1 deviation features behave per the spec."""

    def test_first_event_returns_zero_fallbacks(self) -> None:
        """A single-row card → every deviation feature is 0."""
        df = _build_frame(
            timestamps=[0],
            cards=["A"],
            amounts=[100.0],
            addrs=["X"],
            devices=["D1"],
            hours=[5],
        )
        gen = BehavioralDeviation()
        out = gen.fit_transform(df)
        for col in gen.get_feature_names():
            assert out[col].iloc[0] == 0

    def test_amt_zscore_with_history(self) -> None:
        """Hand-computed: prior amounts [10,20,30,40] → mean=25, std≈12.91; z(100)≈5.81."""
        df = _build_frame(
            timestamps=[0, 100, 200, 300, 400],
            cards=["A"] * 5,
            amounts=[10.0, 20.0, 30.0, 40.0, 100.0],
            addrs=["X"] * 5,
            devices=["D"] * 5,
            hours=[5] * 5,
        )
        gen = BehavioralDeviation()
        out = gen.fit_transform(df)
        ref = pd.Series([10.0, 20.0, 30.0, 40.0])
        expected = (100.0 - float(ref.mean())) / (float(ref.std(ddof=1)) + 1e-9)
        assert out["amt_zscore_vs_card1_history"].iloc[4] == pytest.approx(expected, rel=1e-6)

    def test_amt_zscore_with_one_prior_is_zero(self) -> None:
        """n=1 prior → sample std undefined → z falls back to 0."""
        df = _build_frame(
            timestamps=[0, 100],
            cards=["A", "A"],
            amounts=[10.0, 100.0],
            addrs=["X", "X"],
            devices=["D", "D"],
            hours=[5, 5],
        )
        gen = BehavioralDeviation()
        out = gen.fit_transform(df)
        assert out["amt_zscore_vs_card1_history"].iloc[1] == 0

    def test_time_zscore_with_history(self) -> None:
        """ts=[0,100,250,450,700]; prior deltas=[100,150,200]; current_delta=250."""
        df = _build_frame(
            timestamps=[0, 100, 250, 450, 700],
            cards=["A"] * 5,
            amounts=[100.0] * 5,
            addrs=["X"] * 5,
            devices=["D"] * 5,
            hours=[5] * 5,
        )
        gen = BehavioralDeviation()
        out = gen.fit_transform(df)
        # mean of [100, 150, 200] = 150; std = 50; z = (250 - 150) / (50 + ε) ≈ 2.
        expected = (250.0 - 150.0) / (50.0 + 1e-9)
        assert out["time_since_last_txn_zscore"].iloc[4] == pytest.approx(expected, rel=1e-6)

    def test_addr_change_flag_works(self) -> None:
        """Card switches addr → flag=1; same as mode → flag=0."""
        df = _build_frame(
            timestamps=[0, 100, 200, 300],
            cards=["A"] * 4,
            amounts=[1.0] * 4,
            addrs=["X", "X", "Y", "Z"],
            devices=["D"] * 4,
            hours=[5] * 4,
        )
        gen = BehavioralDeviation()
        out = gen.fit_transform(df)
        # Row 0: first event → 0.
        # Row 1 (X): mode prior is X; same → 0.
        # Row 2 (Y): mode prior is X; different → 1.
        # Row 3 (Z): mode prior is X (count 2 vs Y count 1); different → 1.
        assert out["addr_change_flag"].tolist() == [0, 0, 1, 1]

    def test_device_change_flag_new_device(self) -> None:
        """Device not in past set → flag=1; in past set → 0."""
        df = _build_frame(
            timestamps=[0, 100, 200, 300],
            cards=["A"] * 4,
            amounts=[1.0] * 4,
            addrs=["X"] * 4,
            devices=["D_a", "D_b", "D_a", "D_c"],
            hours=[5] * 4,
        )
        gen = BehavioralDeviation()
        out = gen.fit_transform(df)
        # Row 0: first event → 0.
        # Row 1 (D_b): prior={D_a}; D_b not in → 1.
        # Row 2 (D_a): prior={D_a, D_b}; D_a in → 0.
        # Row 3 (D_c): prior={D_a, D_b}; D_c not in → 1.
        assert out["device_change_flag"].tolist() == [0, 1, 0, 1]

    def test_hour_deviation_correct(self) -> None:
        """Hour deviation = abs(current_hour − prior_mean_hour)."""
        df = _build_frame(
            timestamps=[0, 100, 200, 300],
            cards=["A"] * 4,
            amounts=[1.0] * 4,
            addrs=["X"] * 4,
            devices=["D"] * 4,
            hours=[3, 5, 7, 15],
        )
        gen = BehavioralDeviation()
        out = gen.fit_transform(df)
        # Row 3: prior hours [3, 5, 7] → mean = 5; |15 − 5| = 10.
        assert out["hour_deviation"].iloc[3] == pytest.approx(10.0)

    def test_ties_excluded(self) -> None:
        """Tied-timestamp rows for the same card see neither each other."""
        df = _build_frame(
            timestamps=[0, 100, 100, 200],
            cards=["A"] * 4,
            amounts=[10.0, 20.0, 30.0, 40.0],
            addrs=["X"] * 4,
            devices=["D"] * 4,
            hours=[5] * 4,
        )
        gen = BehavioralDeviation()
        out = gen.fit_transform(df)
        # Rows 1 and 2 are tied at 100. Each sees only row 0 (n=1 prior → z=0).
        assert out["amt_zscore_vs_card1_history"].iloc[1] == 0
        assert out["amt_zscore_vs_card1_history"].iloc[2] == 0

    def test_input_columns_preserved(self) -> None:
        """All input columns survive `transform`; 5 new feature columns added."""
        df = _build_frame(
            timestamps=[0, 100],
            cards=["A", "A"],
            amounts=[10.0, 20.0],
            addrs=["X", "X"],
            devices=["D", "D"],
            hours=[5, 5],
        )
        df["extra_col"] = ["x", "y"]
        gen = BehavioralDeviation()
        out = gen.fit_transform(df)
        for col in df.columns:
            assert col in out.columns
        for col in gen.get_feature_names():
            assert col in out.columns

    def test_nan_amount_does_not_crash(self) -> None:
        """NaN amount is skipped in pass 2; pass 1 falls back to 0."""
        df = _build_frame(
            timestamps=[0, 100, 200],
            cards=["A", "A", "A"],
            amounts=[10.0, float("nan"), 20.0],
            addrs=["X"] * 3,
            devices=["D"] * 3,
            hours=[5] * 3,
        )
        gen = BehavioralDeviation()
        out = gen.fit_transform(df)
        # Row 1 (NaN amt): amt_z stays at 0; the NaN is NOT pushed,
        # so row 2 still sees only row 0's amount in priors.
        assert out["amt_zscore_vs_card1_history"].iloc[1] == 0


# ---------------------------------------------------------------------
# `ColdStartHandler`.
# ---------------------------------------------------------------------


class TestColdStartHandler:
    """`is_coldstart_{entity}` flag set per spec."""

    def test_first_event_is_coldstart(self) -> None:
        """First event for a card → flag=1 across the first `min_history` rows."""
        df = pd.DataFrame(
            {
                "TransactionDT": [0, 100, 200],
                "card1": ["A", "A", "A"],
            }
        )
        gen = ColdStartHandler(min_history=3)
        out = gen.fit_transform(df)
        # Past counts: 0, 1, 2 — all < 3.
        assert out["is_coldstart_card1"].tolist() == [1, 1, 1]

    def test_warm_event_after_min_history(self) -> None:
        """After `min_history` prior events the flag flips to 0."""
        df = pd.DataFrame(
            {
                "TransactionDT": [0, 100, 200, 300, 400],
                "card1": ["A"] * 5,
            }
        )
        gen = ColdStartHandler(min_history=3)
        out = gen.fit_transform(df)
        # Past counts: 0, 1, 2, 3, 4 — first three < 3, last two ≥ 3.
        assert out["is_coldstart_card1"].tolist() == [1, 1, 1, 0, 0]

    def test_multiple_entities(self) -> None:
        """`entity_cols=["card1", "addr1"]` emits both flag columns."""
        df = pd.DataFrame(
            {
                "TransactionDT": [0, 100, 200, 300],
                "card1": ["A", "A", "A", "A"],
                "addr1": [10, 10, 20, 20],
            }
        )
        gen = ColdStartHandler(entity_cols=["card1", "addr1"], min_history=2)
        out = gen.fit_transform(df)
        assert "is_coldstart_card1" in out.columns
        assert "is_coldstart_addr1" in out.columns
        # card1 past counts: 0, 1, 2, 3 → [1, 1, 0, 0]
        assert out["is_coldstart_card1"].tolist() == [1, 1, 0, 0]
        # addr1=10 past counts: 0, 1, 2, 2 → [1, 1, 0, 0] (row 2 is first
        # event for addr1=20 → coldstart=1; row 3 is past_count=1 for addr1=20 → coldstart=1)
        assert out["is_coldstart_addr1"].tolist() == [1, 1, 1, 1]


# ---------------------------------------------------------------------
# Temporal safety.
# ---------------------------------------------------------------------


def _make_synthetic_real_frame(n: int = 60, seed: int = 0) -> pd.DataFrame:
    """Build a multi-card synthetic frame for the leak walks."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "TransactionDT": np.arange(n, dtype=np.int64) * 60,
            "card1": rng.choice(["A", "B", "C"], size=n),
            "TransactionAmt": rng.uniform(1.0, 500.0, size=n),
            "addr1": rng.choice([10, 20, 30], size=n),
            "DeviceInfo": rng.choice(["D1", "D2", "D3"], size=n),
            "hour_of_day": rng.integers(0, 24, size=n),
        }
    )


class TestTemporalSafety:
    """Both classes' headline outputs pass `assert_no_future_leak`."""

    def test_assert_no_future_leak_amt_zscore_passes(self) -> None:
        df = _make_synthetic_real_frame()
        gen = BehavioralDeviation()
        out = gen.fit_transform(df)

        def _recompute(slice_df: pd.DataFrame) -> pd.Series[Any]:
            recomputed = BehavioralDeviation().fit_transform(slice_df)
            series = recomputed["amt_zscore_vs_card1_history"]
            series.name = "amt_zscore_vs_card1_history"
            return series

        assert_no_future_leak(out, df, _recompute)

    def test_assert_no_future_leak_coldstart_passes(self) -> None:
        df = _make_synthetic_real_frame()
        gen = ColdStartHandler()
        out = gen.fit_transform(df)

        def _recompute(slice_df: pd.DataFrame) -> pd.Series[Any]:
            recomputed = ColdStartHandler().fit_transform(slice_df)
            series = recomputed["is_coldstart_card1"]
            series.name = "is_coldstart_card1"
            return series

        assert_no_future_leak(out, df, _recompute)


# ---------------------------------------------------------------------
# Config loading.
# ---------------------------------------------------------------------


class TestConfigLoad:
    """Default constructors read the YAML config files."""

    def test_default_config_loads_behavioral(self) -> None:
        gen = BehavioralDeviation()
        assert gen.entity_col == "card1"
        assert gen.amount_col == "TransactionAmt"
        assert gen.addr_col == "addr1"
        assert gen.device_col == "DeviceInfo"
        assert gen.hour_col == "hour_of_day"
        assert gen.timestamp_col == "TransactionDT"
        assert gen.epsilon == pytest.approx(1.0e-9)
        assert gen.get_feature_names() == [
            "amt_zscore_vs_card1_history",
            "time_since_last_txn_zscore",
            "addr_change_flag",
            "device_change_flag",
            "hour_deviation",
        ]

    def test_default_config_loads_coldstart(self) -> None:
        gen = ColdStartHandler()
        assert gen.entity_cols == ("card1",)
        assert gen.min_history == 3
        assert gen.timestamp_col == "TransactionDT"
        assert gen.get_feature_names() == ["is_coldstart_card1"]
