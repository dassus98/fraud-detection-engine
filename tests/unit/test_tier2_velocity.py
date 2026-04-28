"""Unit tests for `fraud_engine.features.tier2_aggregations.VelocityCounter`.

Four contract surfaces:

- `TestVelocityCounter`: hand-computed counts on a 5-row frame; NaN
  entity → 0; strict-past semantics; tied-timestamp handling; input
  columns survive.
- `TestTemporalSafety`: `assert_no_future_leak` passes; a hypothesis
  property test verifies the optimised deque sweep agrees with a
  `TemporalSafeGenerator`-based naive reference on random small frames.
- `TestPerformance` (`@pytest.mark.slow`): 100k-row benchmark must
  complete in well under 30 s.
- `TestConfigLoad`: default constructor reads `configs/velocity.yaml`;
  explicit kwargs override the YAML.
"""

from __future__ import annotations

import time
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
from fraud_engine.features.tier2_aggregations import VelocityCounter

# ---------------------------------------------------------------------
# Reference implementation: naive O(n²) velocity counter for property test.
# ---------------------------------------------------------------------


class _NaiveVelocityCounter(TemporalSafeGenerator):
    """Reference implementation of VelocityCounter — O(n²) row-iterating.

    Built on top of `TemporalSafeGenerator` so `past_df` is guaranteed
    strict-past by the base class. For each row, filters `past_df` by
    `(entity_value, ts >= window_start)` and counts. Slow but provably
    correct; used as the property-test oracle.
    """

    def __init__(
        self,
        entity_cols: Sequence[str],
        windows: Mapping[str, int] | Sequence[tuple[str, int]],
    ) -> None:
        super().__init__()
        self.entity_cols: tuple[str, ...] = tuple(entity_cols)
        if isinstance(windows, Mapping):
            self.windows: tuple[tuple[str, int], ...] = tuple(windows.items())
        else:
            self.windows = tuple(windows)

    def _compute_for_row(self, row: pd.Series, past_df: pd.DataFrame) -> dict[str, Any]:
        ts = row[self.timestamp_col]
        result: dict[str, Any] = {}
        for entity in self.entity_cols:
            entity_val = row[entity]
            for label, secs in self.windows:
                col = f"{entity}_velocity_{label}"
                if pd.isna(entity_val):
                    result[col] = 0
                    continue
                window_start = ts - secs
                matches = past_df[
                    (past_df[entity] == entity_val) & (past_df[self.timestamp_col] >= window_start)
                ]
                result[col] = int(len(matches))
        return result

    def get_feature_names(self) -> list[str]:
        return [f"{e}_velocity_{label}" for e in self.entity_cols for label, _ in self.windows]

    def get_business_rationale(self) -> str:
        return "Reference: naive O(n²) velocity counter for property-test."


# ---------------------------------------------------------------------
# Tier-2 velocity contract tests.
# ---------------------------------------------------------------------


class TestVelocityCounter:
    """Contract tests for the deque-based optimized implementation."""

    def test_hand_computed_counts(self) -> None:
        """Counts on a 5-row hand-computed frame match expected values."""
        df = pd.DataFrame(
            {
                "TransactionDT": [0, 100, 200, 7000, 50_000],
                "entity": ["A", "A", "A", "B", "A"],
            }
        )
        gen = VelocityCounter(entity_cols=["entity"], windows={"1h": 3600, "24h": 86400})
        out = gen.fit_transform(df)

        # Strict-past, same-entity counts within each window.
        # 1h (3600s): rows 1,2 see only nearby A's; row 4 (ts=50000)
        # has 1h window_start = 46400 — none of A's prior ts qualify.
        assert out["entity_velocity_1h"].tolist() == [0, 1, 2, 0, 0]
        # 24h (86400s): row 4's window_start = -36400 — all 3 prior A's
        # qualify.
        assert out["entity_velocity_24h"].tolist() == [0, 1, 2, 0, 3]

    def test_nan_entity_yields_zero(self) -> None:
        """A NaN entity value produces 0 across every velocity column for that row."""
        df = pd.DataFrame(
            {
                "TransactionDT": [0, 100, 200],
                "entity": ["A", None, "A"],
            }
        )
        gen = VelocityCounter(entity_cols=["entity"], windows={"1h": 3600})
        out = gen.fit_transform(df)

        # Row 1 (NaN entity) gets 0; row 2 sees only row 0 (A), so 1.
        # NaN row's timestamp is NOT pushed to any deque — row 2 still
        # only counts row 0.
        assert out["entity_velocity_1h"].tolist() == [0, 0, 1]

    def test_strict_past_no_self_count(self) -> None:
        """The row at T does not count itself in its own velocity."""
        df = pd.DataFrame(
            {
                "TransactionDT": [0, 1, 2, 3],
                "entity": ["A", "A", "A", "A"],
            }
        )
        gen = VelocityCounter(entity_cols=["entity"], windows={"big": 1_000_000})
        out = gen.fit_transform(df)
        # If self-counting leaked in, every row would be off by 1.
        assert out["entity_velocity_big"].tolist() == [0, 1, 2, 3]

    def test_ties_on_timestamp_excluded(self) -> None:
        """Tied timestamps see neither each other nor the future."""
        df = pd.DataFrame(
            {
                "TransactionDT": [0, 100, 100, 200],
                "entity": ["A", "A", "A", "A"],
            }
        )
        gen = VelocityCounter(entity_cols=["entity"], windows={"big": 1_000_000})
        out = gen.fit_transform(df)
        # Row 0 (ts=0): 0 prior. Rows 1, 2 (tied at ts=100): each sees
        # only the row at ts=0, NOT each other. Row 3 (ts=200): sees
        # all three earlier rows.
        assert out["entity_velocity_big"].tolist() == [0, 1, 1, 3]

    def test_input_columns_preserved(self) -> None:
        """All input columns survive `transform`; full set of velocity columns is added."""
        df = pd.DataFrame(
            {
                "TransactionDT": [0, 100],
                "card1": [1, 2],
                "addr1": [10.0, 20.0],
                "extra_col": ["x", "y"],
            }
        )
        gen = VelocityCounter(entity_cols=["card1", "addr1"], windows={"1h": 3600, "24h": 86400})
        out = gen.fit_transform(df)

        # All input columns survive.
        for col in df.columns:
            assert col in out.columns
        # 2 entities × 2 windows = 4 new columns.
        expected_velocity_cols = {
            "card1_velocity_1h",
            "card1_velocity_24h",
            "addr1_velocity_1h",
            "addr1_velocity_24h",
        }
        assert expected_velocity_cols.issubset(set(out.columns))


# ---------------------------------------------------------------------
# Temporal-safety property tests.
# ---------------------------------------------------------------------


@st.composite
def _frame_strategy(draw: st.DrawFn) -> pd.DataFrame:
    """Generate a small DataFrame with TransactionDT + entity columns."""
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
    return pd.DataFrame({"TransactionDT": timestamps, "entity": entities})


@st.composite
def _windows_strategy(draw: st.DrawFn) -> dict[str, int]:
    """Generate 1-3 distinct windows in 1..500 seconds."""
    n_windows = draw(st.integers(min_value=1, max_value=3))
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
        """`assert_no_future_leak` passes on the velocity output."""
        df = pd.DataFrame(
            {
                "TransactionDT": np.arange(50, dtype=np.int64) * 60,
                "entity": ["A" if i % 3 == 0 else "B" for i in range(50)],
            }
        )
        gen = VelocityCounter(entity_cols=["entity"], windows={"5m": 300})
        out = gen.fit_transform(df)

        def _recompute(slice_df: pd.DataFrame) -> pd.Series[Any]:
            recomputed = VelocityCounter(entity_cols=["entity"], windows={"5m": 300}).fit_transform(
                slice_df
            )
            series = recomputed["entity_velocity_5m"]
            series.name = "entity_velocity_5m"
            return series

        assert_no_future_leak(out, df, _recompute)

    @given(df=_frame_strategy(), windows=_windows_strategy())
    @settings(
        max_examples=50,
        deadline=2000,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_optimized_matches_naive(self, df: pd.DataFrame, windows: dict[str, int]) -> None:
        """Optimized deque sweep matches the naive O(n²) reference column-for-column."""
        optimized = VelocityCounter(entity_cols=["entity"], windows=windows)
        naive = _NaiveVelocityCounter(entity_cols=["entity"], windows=windows)

        out_opt = optimized.fit_transform(df)
        out_naive = naive.fit_transform(df)

        feature_cols = optimized.get_feature_names()
        # Compare by value; dtype may differ trivially (int vs int64).
        opt_view = out_opt[feature_cols].reset_index(drop=True).astype(int)
        naive_view = out_naive[feature_cols].reset_index(drop=True).astype(int)
        pd.testing.assert_frame_equal(opt_view, naive_view, check_dtype=False)


# ---------------------------------------------------------------------
# Performance benchmark.
# ---------------------------------------------------------------------


class TestPerformance:
    """Spec contract: 100k rows × 4 entities × 3 windows in <30 s."""

    @pytest.mark.slow
    def test_100k_rows_under_30s(self) -> None:
        """Wall-clock benchmark: 100k rows finishes well under the 30 s ceiling."""
        n = 100_000
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "TransactionDT": np.arange(n, dtype=np.int64) * 60,
                "card1": rng.integers(0, 10_000, size=n, dtype=np.int64),
                "addr1": rng.integers(0, 1_000, size=n, dtype=np.int64),
                "DeviceInfo": rng.choice(["A", "B", "C", "D"], size=n),
                "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", "hotmail.com"], size=n),
            }
        )
        gen = VelocityCounter(
            entity_cols=["card1", "addr1", "DeviceInfo", "P_emaildomain"],
            windows={"1h": 3600, "24h": 86400, "7d": 604800},
        )

        start = time.perf_counter()
        out = gen.fit_transform(df)
        elapsed = time.perf_counter() - start

        # Sanity: every velocity column landed in the output.
        for col in gen.get_feature_names():
            assert col in out.columns
        assert len(out) == n
        # Headline gate: under the 30 s spec ceiling.
        assert elapsed < 30.0, f"100k rows took {elapsed:.2f}s, ceiling 30s"


# ---------------------------------------------------------------------
# Config loading.
# ---------------------------------------------------------------------


class TestConfigLoad:
    """`VelocityCounter()` (no kwargs) loads `configs/velocity.yaml`."""

    def test_default_config_loads(self) -> None:
        """Default constructor reads YAML; entity_cols + windows match the file."""
        gen = VelocityCounter()
        # YAML default: 4 entities × 3 windows = 12 columns.
        assert gen.entity_cols == ("card1", "addr1", "DeviceInfo", "P_emaildomain")
        assert dict(gen.windows) == {"1h": 3600, "24h": 86400, "7d": 604800}
        assert len(gen.get_feature_names()) == 12

    def test_constructor_overrides_config(self) -> None:
        """Explicit kwargs ignore the YAML defaults."""
        gen = VelocityCounter(entity_cols=["card1"], windows={"30s": 30})
        assert gen.entity_cols == ("card1",)
        assert dict(gen.windows) == {"30s": 30}
        assert gen.get_feature_names() == ["card1_velocity_30s"]
