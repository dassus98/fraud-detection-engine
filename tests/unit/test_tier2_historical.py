"""Unit tests for `fraud_engine.features.tier2_aggregations.HistoricalStats`.

Three contract surfaces:

- `TestHistoricalStats`: hand-computed mean / std / max on a small
  frame; NaN entity → NaN stats; empty window → NaN; single-event
  std=NaN; strict-past semantics; tied-timestamp handling;
  unsupported-stat constructor rejection.
- `TestTemporalSafety`: `assert_no_future_leak` passes; a hypothesis
  property test verifies the optimised deque sweep agrees with a
  `TemporalSafeGenerator`-based naive reference on random small frames.
- `TestConfigLoad`: default constructor reads `configs/historical_stats.yaml`;
  explicit kwargs override the YAML.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings

from fraud_engine.features.temporal_guards import (
    TemporalSafeGenerator,
    assert_no_future_leak,
)
from fraud_engine.features.tier2_aggregations import HistoricalStats

# ---------------------------------------------------------------------
# Reference implementation: naive O(n²) historical-stats for property test.
# ---------------------------------------------------------------------


class _NaiveHistoricalStats(TemporalSafeGenerator):
    """Reference implementation of HistoricalStats — O(n²) row-iterating.

    Built on top of `TemporalSafeGenerator` so `past_df` is guaranteed
    strict-past by the base class. For each row, filters `past_df` by
    `(entity_value, ts >= window_start)` and computes stats with
    `pd.Series` methods (which use sample std by default, matching
    the optimised impl's `ddof=1`).
    """

    def __init__(
        self,
        entity_stats: Mapping[str, Sequence[str]] | Sequence[tuple[str, Sequence[str]]],
        windows: Mapping[str, int] | Sequence[tuple[str, int]],
        amount_col: str = "TransactionAmt",
    ) -> None:
        super().__init__()
        if isinstance(entity_stats, Mapping):
            self.entity_stats: tuple[tuple[str, tuple[str, ...]], ...] = tuple(
                (e, tuple(s)) for e, s in entity_stats.items()
            )
        else:
            self.entity_stats = tuple((e, tuple(s)) for e, s in entity_stats)
        if isinstance(windows, Mapping):
            self.windows: tuple[tuple[str, int], ...] = tuple(windows.items())
        else:
            self.windows = tuple(windows)
        self.amount_col = amount_col

    def _compute_for_row(self, row: pd.Series, past_df: pd.DataFrame) -> dict[str, Any]:
        ts = row[self.timestamp_col]
        result: dict[str, Any] = {}
        for entity, stats in self.entity_stats:
            entity_val = row[entity]
            for label, secs in self.windows:
                window_start = ts - secs
                if pd.isna(entity_val):
                    matches = pd.Series(dtype=float)
                else:
                    matches = past_df[
                        (past_df[entity] == entity_val)
                        & (past_df[self.timestamp_col] >= window_start)
                    ][self.amount_col]
                for stat in stats:
                    col = f"{entity}_amt_{stat}_{label}"
                    if matches.empty:
                        result[col] = float("nan")
                    elif stat == "mean":
                        result[col] = float(matches.mean())
                    elif stat == "max":
                        result[col] = float(matches.max())
                    elif stat == "std":
                        if len(matches) < 2:
                            result[col] = float("nan")
                        else:
                            result[col] = float(matches.std(ddof=1))
        return result

    def get_feature_names(self) -> list[str]:
        return [
            f"{e}_amt_{stat}_{label}"
            for e, stats in self.entity_stats
            for stat in stats
            for label, _ in self.windows
        ]

    def get_business_rationale(self) -> str:
        return "Reference: naive O(n²) historical-stats for property-test."


# ---------------------------------------------------------------------
# Tier-2 historical-stats contract tests.
# ---------------------------------------------------------------------


_BIG_WINDOW: int = 1_000_000


class TestHistoricalStats:
    """Contract tests for the deque-based optimised implementation."""

    def test_hand_computed_stats(self) -> None:
        """Mean / std / max on a 5-row hand-computed frame match expected values."""
        df = pd.DataFrame(
            {
                "TransactionDT": [0, 100, 200, 300, 400],
                "entity": ["A", "A", "A", "A", "A"],
                "TransactionAmt": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )
        gen = HistoricalStats(
            entity_stats={"entity": ["mean", "std", "max"]},
            windows={"big": _BIG_WINDOW},
            amount_col="TransactionAmt",
        )
        out = gen.fit_transform(df)

        # Row 0: empty deque → all NaN.
        assert pd.isna(out["entity_amt_mean_big"].iloc[0])
        assert pd.isna(out["entity_amt_std_big"].iloc[0])
        assert pd.isna(out["entity_amt_max_big"].iloc[0])

        # Row 1: deque=[10] → mean=10, std=NaN, max=10.
        assert out["entity_amt_mean_big"].iloc[1] == pytest.approx(10.0)
        assert pd.isna(out["entity_amt_std_big"].iloc[1])
        assert out["entity_amt_max_big"].iloc[1] == pytest.approx(10.0)

        # Row 4: deque=[10, 20, 30, 40] → mean=25, std=pd.Series.std (sample), max=40.
        ref = pd.Series([10.0, 20.0, 30.0, 40.0])
        assert out["entity_amt_mean_big"].iloc[4] == pytest.approx(float(ref.mean()))
        assert out["entity_amt_std_big"].iloc[4] == pytest.approx(float(ref.std(ddof=1)))
        assert out["entity_amt_max_big"].iloc[4] == pytest.approx(40.0)

    def test_nan_entity_yields_nan(self) -> None:
        """NaN entity row produces NaN across every stat column for that row."""
        df = pd.DataFrame(
            {
                "TransactionDT": [0, 100, 200],
                "entity": ["A", None, "A"],
                "TransactionAmt": [10.0, 20.0, 30.0],
            }
        )
        gen = HistoricalStats(
            entity_stats={"entity": ["mean", "max"]},
            windows={"big": _BIG_WINDOW},
            amount_col="TransactionAmt",
        )
        out = gen.fit_transform(df)

        # Row 1 (NaN entity): both stats NaN.
        assert pd.isna(out["entity_amt_mean_big"].iloc[1])
        assert pd.isna(out["entity_amt_max_big"].iloc[1])
        # Row 2 (entity=A): row 1's amount was NOT pushed (NaN entity);
        # so deque only contains row 0 (amt=10).
        assert out["entity_amt_mean_big"].iloc[2] == pytest.approx(10.0)
        assert out["entity_amt_max_big"].iloc[2] == pytest.approx(10.0)

    def test_empty_window_yields_nan(self) -> None:
        """First-row stats for an entity (empty deque) are NaN across the board."""
        df = pd.DataFrame(
            {
                "TransactionDT": [0, 100],
                "entity": ["A", "B"],
                "TransactionAmt": [10.0, 20.0],
            }
        )
        gen = HistoricalStats(
            entity_stats={"entity": ["mean", "std", "max"]},
            windows={"big": _BIG_WINDOW},
            amount_col="TransactionAmt",
        )
        out = gen.fit_transform(df)
        # Both rows are first-of-entity → all stats NaN.
        for col in ("entity_amt_mean_big", "entity_amt_std_big", "entity_amt_max_big"):
            assert pd.isna(out[col].iloc[0])
            assert pd.isna(out[col].iloc[1])

    def test_single_event_std_nan(self) -> None:
        """n=1 deque: mean=max=that value; std=NaN (sample std needs ≥ 2)."""
        df = pd.DataFrame(
            {
                "TransactionDT": [0, 100],
                "entity": ["A", "A"],
                "TransactionAmt": [42.0, 99.0],
            }
        )
        gen = HistoricalStats(
            entity_stats={"entity": ["mean", "std", "max"]},
            windows={"big": _BIG_WINDOW},
            amount_col="TransactionAmt",
        )
        out = gen.fit_transform(df)
        # Row 1: deque=[(0, 42)] → mean=42, max=42, std=NaN.
        assert out["entity_amt_mean_big"].iloc[1] == pytest.approx(42.0)
        assert out["entity_amt_max_big"].iloc[1] == pytest.approx(42.0)
        assert pd.isna(out["entity_amt_std_big"].iloc[1])

    def test_strict_past_no_self_count(self) -> None:
        """Row at T does not include itself in its own stats."""
        # If self-counting leaked, row 1's mean would be (10+20)/2=15.
        # Strict-past gives mean=10.
        df = pd.DataFrame(
            {
                "TransactionDT": [0, 100],
                "entity": ["A", "A"],
                "TransactionAmt": [10.0, 20.0],
            }
        )
        gen = HistoricalStats(
            entity_stats={"entity": ["mean"]},
            windows={"big": _BIG_WINDOW},
            amount_col="TransactionAmt",
        )
        out = gen.fit_transform(df)
        assert out["entity_amt_mean_big"].iloc[1] == pytest.approx(10.0)

    def test_ties_excluded(self) -> None:
        """Tied timestamps see neither each other nor the future."""
        df = pd.DataFrame(
            {
                "TransactionDT": [0, 100, 100, 200],
                "entity": ["A", "A", "A", "A"],
                "TransactionAmt": [10.0, 20.0, 30.0, 40.0],
            }
        )
        gen = HistoricalStats(
            entity_stats={"entity": ["mean"]},
            windows={"big": _BIG_WINDOW},
            amount_col="TransactionAmt",
        )
        out = gen.fit_transform(df)
        # Row 0 (ts=0): empty → NaN.
        # Rows 1, 2 (tied at ts=100): each sees only ts=0 (amt=10) → mean=10.
        # Row 3 (ts=200): sees ts=0,100,100 → mean = (10+20+30)/3 = 20.
        assert pd.isna(out["entity_amt_mean_big"].iloc[0])
        assert out["entity_amt_mean_big"].iloc[1] == pytest.approx(10.0)
        assert out["entity_amt_mean_big"].iloc[2] == pytest.approx(10.0)
        assert out["entity_amt_mean_big"].iloc[3] == pytest.approx(20.0)

    def test_unsupported_stat_raises(self) -> None:
        """Constructor rejects unsupported stat names with a clear message."""
        with pytest.raises(ValueError, match="median"):
            HistoricalStats(
                entity_stats={"card1": ["median"]},
                windows={"30d": 2_592_000},
                amount_col="TransactionAmt",
            )


# ---------------------------------------------------------------------
# Temporal-safety property tests.
# ---------------------------------------------------------------------


@st.composite
def _frame_strategy(draw: st.DrawFn) -> pd.DataFrame:
    """Generate a small DataFrame with TransactionDT, entity, amount."""
    n = draw(st.integers(min_value=2, max_value=15))
    timestamps = draw(
        st.lists(
            st.integers(min_value=0, max_value=500),
            min_size=n,
            max_size=n,
        )
    )
    entities = draw(
        st.lists(
            st.one_of(st.integers(min_value=0, max_value=3), st.none()),
            min_size=n,
            max_size=n,
        )
    )
    amounts = draw(
        st.lists(
            st.floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False),
            min_size=n,
            max_size=n,
        )
    )
    return pd.DataFrame(
        {"TransactionDT": timestamps, "entity": entities, "TransactionAmt": amounts}
    )


@st.composite
def _windows_strategy(draw: st.DrawFn) -> dict[str, int]:
    """Generate 1-2 distinct windows in 1..500 seconds."""
    n_windows = draw(st.integers(min_value=1, max_value=2))
    secs = draw(
        st.lists(
            st.integers(min_value=1, max_value=500),
            min_size=n_windows,
            max_size=n_windows,
            unique=True,
        )
    )
    return {f"w{i}": s for i, s in enumerate(secs)}


class TestTemporalSafety:
    """The optimised generator's output must be leak-free and match the naive reference."""

    def test_assert_no_future_leak_passes(self) -> None:
        """`assert_no_future_leak` passes on the historical-stats output."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "TransactionDT": np.arange(50, dtype=np.int64) * 60,
                "entity": ["A" if i % 3 == 0 else "B" for i in range(50)],
                "TransactionAmt": rng.uniform(1.0, 200.0, size=50),
            }
        )
        gen = HistoricalStats(
            entity_stats={"entity": ["mean"]},
            windows={"5m": 300},
            amount_col="TransactionAmt",
        )
        out = gen.fit_transform(df)

        def _recompute(slice_df: pd.DataFrame) -> pd.Series[Any]:
            recomputed = HistoricalStats(
                entity_stats={"entity": ["mean"]},
                windows={"5m": 300},
                amount_col="TransactionAmt",
            ).fit_transform(slice_df)
            series = recomputed["entity_amt_mean_5m"]
            series.name = "entity_amt_mean_5m"
            return series

        assert_no_future_leak(out, df, _recompute)

    @given(df=_frame_strategy(), windows=_windows_strategy())
    @settings(
        max_examples=50,
        deadline=2000,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_optimized_matches_naive(self, df: pd.DataFrame, windows: dict[str, int]) -> None:
        """Optimised deque sweep matches the naive O(n²) reference column-for-column."""
        entity_stats = {"entity": ["mean", "std", "max"]}
        optimized = HistoricalStats(
            entity_stats=entity_stats,
            windows=windows,
            amount_col="TransactionAmt",
        )
        naive = _NaiveHistoricalStats(
            entity_stats=entity_stats,
            windows=windows,
            amount_col="TransactionAmt",
        )

        out_opt = optimized.fit_transform(df)
        out_naive = naive.fit_transform(df)

        feature_cols = optimized.get_feature_names()
        # Use float comparison; NaNs match NaNs under pd.testing.
        opt_view = out_opt[feature_cols].reset_index(drop=True).astype(float)
        naive_view = out_naive[feature_cols].reset_index(drop=True).astype(float)
        pd.testing.assert_frame_equal(opt_view, naive_view, check_dtype=False)


# ---------------------------------------------------------------------
# Config loading.
# ---------------------------------------------------------------------


class TestConfigLoad:
    """`HistoricalStats()` (no kwargs) loads `configs/historical_stats.yaml`."""

    def test_default_config_loads(self) -> None:
        """Default constructor reads YAML; entity_stats + windows + amount_col match the file."""
        gen = HistoricalStats()
        # YAML default: card1 -> [mean, std, max]; addr1 -> [mean, std]; 1 window.
        assert gen.entity_stats == (
            ("card1", ("mean", "std", "max")),
            ("addr1", ("mean", "std")),
        )
        assert dict(gen.windows) == {"30d": 2_592_000}
        assert gen.amount_col == "TransactionAmt"
        # Total feature columns: 3 (card1) + 2 (addr1) = 5.
        assert len(gen.get_feature_names()) == 5

    def test_constructor_overrides_config(self) -> None:
        """Explicit kwargs ignore YAML defaults."""
        gen = HistoricalStats(
            entity_stats={"card1": ["mean"]},
            windows={"5s": 5},
            amount_col="TransactionAmt",
        )
        assert gen.entity_stats == (("card1", ("mean",)),)
        assert dict(gen.windows) == {"5s": 5}
        assert gen.get_feature_names() == ["card1_amt_mean_5s"]
