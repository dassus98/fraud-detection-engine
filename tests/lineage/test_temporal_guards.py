"""Lineage-contract tests for `fraud_engine.features.temporal_guards`.

Three contract surfaces:

- `assert_no_future_leak`: a safe (past-only) feature passes; a leaky
  (look-ahead) feature raises `AssertionError`; an unnamed Series
  raises `ValueError`; first-row NaN is handled; same seed reproduces
  the same failure; ``n_samples`` is clamped to frame size.
- `TemporalSafeGenerator`: a concrete subclass produces correct
  outputs end-to-end and round-trips through `assert_no_future_leak`;
  the ABC enforces `_compute_for_row`; the slice handed to
  `_compute_for_row` is strictly past.
- Tie-on-timestamp edge case: rows sharing a timestamp do NOT see
  each other in the past slice (strict ``<``).

All tests use synthetic in-memory frames — no `MANIFEST.json` gate —
so they always run under both `make test-fast` and `make test-lineage`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from fraud_engine.features.temporal_guards import (
    TemporalSafeGenerator,
    assert_no_future_leak,
)

pytestmark = pytest.mark.lineage


# ---------------------------------------------------------------------
# Synthetic helpers.
# ---------------------------------------------------------------------


def _synthetic_frame(n: int = 100, seed: int = 0) -> pd.DataFrame:
    """Toy frame: monotone TransactionDT (seconds) + a numeric amount column."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "TransactionDT": np.arange(n, dtype=np.int64) * 60,
            "amount": rng.uniform(1.0, 500.0, size=n),
        }
    )


def _safe_lag1(df: pd.DataFrame) -> pd.Series:
    """Past-only feature: previous row's amount; first row → NaN."""
    out = df["amount"].shift(1)
    out.name = "amount_lag1"
    return out


def _leaky_lead1(df: pd.DataFrame) -> pd.Series:
    """LEAKING feature: next row's amount."""
    out = df["amount"].shift(-1)
    out.name = "amount_lead1"
    return out


# ---------------------------------------------------------------------
# `assert_no_future_leak` tests.
# ---------------------------------------------------------------------


class TestAssertNoFutureLeak:
    """Contract tests for `assert_no_future_leak`."""

    def test_safe_feature_passes(self) -> None:
        """A past-only feature passes the assertion."""
        source = _synthetic_frame(n=100)
        feature_df = source.copy()
        feature_df["amount_lag1"] = _safe_lag1(source)
        assert_no_future_leak(feature_df, source, _safe_lag1)

    def test_leaky_feature_raises(self) -> None:
        """A look-ahead feature trips the assertion with an informative message."""
        source = _synthetic_frame(n=100)
        feature_df = source.copy()
        feature_df["amount_lead1"] = _leaky_lead1(source)
        with pytest.raises(AssertionError, match="amount_lead1"):
            assert_no_future_leak(feature_df, source, _leaky_lead1)

    def test_unnamed_series_raises_value_error(self) -> None:
        """A feature_func returning an unnamed Series raises ValueError."""
        source = _synthetic_frame(n=20)
        feature_df = source.copy()
        feature_df["unnamed"] = source["amount"].shift(1)

        def _unnamed(df: pd.DataFrame) -> pd.Series:
            # `shift` propagates the column name; we must explicitly clear
            # it so the Series is genuinely unnamed.
            out = df["amount"].shift(1)
            out.name = None
            return out

        with pytest.raises(ValueError, match="unnamed"):
            assert_no_future_leak(feature_df, source, _unnamed)

    def test_first_row_nan_handled(self) -> None:
        """First-row NaN compares as a match against itself; no false alarm."""
        source = _synthetic_frame(n=10)
        feature_df = source.copy()
        feature_df["amount_lag1"] = _safe_lag1(source)
        # Force the sampled set to include the first row by sampling all of them.
        assert_no_future_leak(feature_df, source, _safe_lag1, n_samples=10)

    def test_seed_reproducible(self) -> None:
        """Same seed picks the same rows; same first failure idx."""
        source = _synthetic_frame(n=100)
        feature_df = source.copy()
        feature_df["amount_lead1"] = _leaky_lead1(source)

        with pytest.raises(AssertionError) as exc1:
            assert_no_future_leak(feature_df, source, _leaky_lead1, seed=123)
        with pytest.raises(AssertionError) as exc2:
            assert_no_future_leak(feature_df, source, _leaky_lead1, seed=123)
        assert str(exc1.value) == str(exc2.value)

    def test_n_samples_clamped_to_frame_size(self) -> None:
        """`n_samples` larger than frame size is clamped without raising."""
        source = _synthetic_frame(n=100)
        feature_df = source.copy()
        feature_df["amount_lag1"] = _safe_lag1(source)
        # 10_000 > 100 — must clamp to 100, not raise.
        assert_no_future_leak(feature_df, source, _safe_lag1, n_samples=10_000)


# ---------------------------------------------------------------------
# `TemporalSafeGenerator` tests.
# ---------------------------------------------------------------------


class _RunningCount(TemporalSafeGenerator):
    """Stub: emits `prior_count` = number of strictly-earlier rows.

    By construction `prior_count` at row T equals
    ``len(source[source.TransactionDT < T])`` — i.e. the strict-past
    count.  Used as the smallest possible concrete subclass that
    still exercises every contract surface.
    """

    def _compute_for_row(self, row: pd.Series, past_df: pd.DataFrame) -> dict[str, Any]:
        return {"prior_count": int(len(past_df))}

    def get_feature_names(self) -> list[str]:
        return ["prior_count"]

    def get_business_rationale(self) -> str:
        return "Stub: count of rows with timestamp strictly before this row's."


class _StrictPastChecker(TemporalSafeGenerator):
    """Stub that ASSERTS its `past_df` is strictly-past.

    Increments `checks_passed` on each non-empty `past_df`. If the
    base `transform` ever passed a non-strict-past slice, the
    `assert` inside `_compute_for_row` would fire.
    """

    def __init__(self) -> None:
        super().__init__()
        self.checks_passed: int = 0

    def _compute_for_row(self, row: pd.Series, past_df: pd.DataFrame) -> dict[str, Any]:
        if not past_df.empty:
            assert (past_df["TransactionDT"] < row["TransactionDT"]).all(), (
                f"past_df contains rows with TransactionDT >= " f"{row['TransactionDT']}"
            )
            self.checks_passed += 1
        return {"dummy": 0}

    def get_feature_names(self) -> list[str]:
        return ["dummy"]

    def get_business_rationale(self) -> str:
        return "Stub: asserts past_df is strictly past."


class TestTemporalSafeGenerator:
    """Contract tests for the row-iterating reference generator."""

    def test_concrete_subclass_runs_end_to_end(self) -> None:
        """First row's prior_count == 0; last row's prior_count == n-1; columns survive."""
        df = _synthetic_frame(n=50)
        gen = _RunningCount()
        out = gen.transform(df)

        assert len(out) == len(df)
        assert "prior_count" in out.columns
        # Every input column survives.
        for col in df.columns:
            assert col in out.columns
        assert out["prior_count"].iloc[0] == 0
        assert out["prior_count"].iloc[-1] == 49
        # Monotone non-decreasing because TransactionDT is monotone.
        assert (out["prior_count"].diff().dropna() >= 0).all()

    def test_temporal_safe_generator_passes_assert_no_future_leak(self) -> None:
        """`_RunningCount` output round-trips through `assert_no_future_leak`."""
        source = _synthetic_frame(n=80)
        gen = _RunningCount()
        feature_df = gen.transform(source)

        def _recompute(slice_df: pd.DataFrame) -> pd.Series:
            recomputed = _RunningCount().transform(slice_df)
            out = recomputed["prior_count"]
            out.name = "prior_count"
            return out

        assert_no_future_leak(feature_df, source, _recompute)

    def test_subclass_must_implement_compute_for_row(self) -> None:
        """Missing `_compute_for_row` → TypeError at instantiation (ABC enforcement)."""

        class _Incomplete(TemporalSafeGenerator):
            def get_feature_names(self) -> list[str]:
                return ["foo"]

            def get_business_rationale(self) -> str:
                return "Incomplete subclass."

        with pytest.raises(TypeError, match="_compute_for_row"):
            _Incomplete()  # type: ignore[abstract]

    def test_strict_past_passed_to_compute_for_row(self) -> None:
        """`past_df` handed to `_compute_for_row` is strictly past."""
        df = _synthetic_frame(n=20)
        gen = _StrictPastChecker()
        gen.transform(df)
        # 19 non-empty past_dfs (rows 1..19); row 0 has empty past.
        assert gen.checks_passed == 19


# ---------------------------------------------------------------------
# Edge case: ties on timestamp.
# ---------------------------------------------------------------------


class TestTimestampTies:
    """Rows sharing a timestamp do NOT see each other (strict `<`)."""

    def test_ties_on_timestamp_excluded_from_past(self) -> None:
        """Two rows tied at T=60: each sees only the row at T=0, not the other tied row."""
        df = pd.DataFrame(
            {
                "TransactionDT": [0, 60, 60, 120],
                "amount": [10.0, 20.0, 30.0, 40.0],
            }
        )
        gen = _RunningCount()
        out = gen.transform(df)
        # Row 0 (T=0):   0 strictly-earlier rows.
        # Row 1 (T=60):  1 strictly-earlier row (T=0).
        # Row 2 (T=60):  1 strictly-earlier row (T=0) — the OTHER row at
        #                T=60 is NOT counted because `<` is strict.
        # Row 3 (T=120): 3 strictly-earlier rows (T=0, 60, 60).
        assert out["prior_count"].tolist() == [0, 1, 1, 3]
