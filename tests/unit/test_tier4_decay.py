"""Unit tests for `fraud_engine.features.tier4_decay.ExponentialDecayVelocity`.

Five contract surfaces:

- `TestExponentialDecayVelocity`: hand-computed v_ewm / fraud_v_ewm
  values; first-event / single-prior / NaN-entity fallbacks; tied-
  timestamp semantics; the single-row OOF leak gate.
- `TestTransformVal`: fit-then-transform behaviour (state decays
  forward; unseen entities → 0; pre-fit raises; backward time raises).
- `TestTemporalSafety`: `assert_no_future_leak` passes; hypothesis
  property test verifies the optimised running-state algorithm
  matches the naive O(n²) reference column-for-column on random
  small frames (50 examples).
- `TestPerformance` (`@pytest.mark.slow`): 100k-row benchmark must
  complete well under the 30 s spec ceiling.
- `TestConfigLoad`: default constructor reads `tier4_config.yaml`;
  explicit kwargs override the YAML; duplicate λ values raise.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
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
from fraud_engine.features.tier4_decay import ExponentialDecayVelocity

# ---------------------------------------------------------------------
# Reference implementation: naive O(n²) row-iterating EWM.
# ---------------------------------------------------------------------


class _NaiveExponentialDecayVelocity(TemporalSafeGenerator):
    """Reference O(n²) row-iterating EWM for the property test.

    Built on top of `TemporalSafeGenerator` so `past_df` is guaranteed
    strictly past by the base class (the row at T cannot see itself).
    For each row, filters `past_df` by entity equality and computes
    `Σ exp(-λ · Δt_hours)` directly — provably correct, slow.

    Mirrors `ExponentialDecayVelocity` semantics:
        - NaN entity → all 0.
        - NaN `is_fraud` in past rows → treated as 0 (matches the
          optimised impl's `math.isnan(...) → 0.0` policy).
        - Column names use `f"{λ:g}"` format (matches optimised).
        - Output column order: outer entity_cols, inner lambdas,
          v_ewm before fraud_v_ewm per (entity, λ).
    """

    def __init__(
        self,
        entity_cols: Sequence[str],
        lambdas: Sequence[float],
        fraud_weighted: bool = False,
        target_col: str = "isFraud",
    ) -> None:
        super().__init__()
        self.entity_cols: tuple[str, ...] = tuple(entity_cols)
        self.lambdas: tuple[float, ...] = tuple(lambdas)
        self.fraud_weighted: bool = fraud_weighted
        self.target_col: str = target_col

    def _compute_for_row(self, row: pd.Series, past_df: pd.DataFrame) -> dict[str, Any]:
        ts = row[self.timestamp_col]
        result: dict[str, Any] = {}
        for ec in self.entity_cols:
            entity_val = row[ec]
            if pd.isna(entity_val) or past_df.empty:
                same_entity = past_df.iloc[0:0]
            else:
                same_entity = past_df[past_df[ec] == entity_val]
            for lam in self.lambdas:
                lam_str = f"{lam:g}"
                v_col = f"{ec}_v_ewm_lambda_{lam_str}"
                f_col = f"{ec}_fraud_v_ewm_lambda_{lam_str}"
                if same_entity.empty:
                    result[v_col] = 0.0
                    if self.fraud_weighted:
                        result[f_col] = 0.0
                    continue
                dts_h = (ts - same_entity[self.timestamp_col].to_numpy()) / 3600.0
                decays = np.exp(-lam * dts_h)
                result[v_col] = float(decays.sum())
                if self.fraud_weighted:
                    fraud = same_entity[self.target_col].to_numpy().astype(float)
                    fraud = np.nan_to_num(fraud, nan=0.0)
                    result[f_col] = float((decays * fraud).sum())
        return result

    def get_feature_names(self) -> list[str]:
        names: list[str] = []
        for ec in self.entity_cols:
            for lam in self.lambdas:
                lam_str = f"{lam:g}"
                names.append(f"{ec}_v_ewm_lambda_{lam_str}")
                if self.fraud_weighted:
                    names.append(f"{ec}_fraud_v_ewm_lambda_{lam_str}")
        return names

    def get_business_rationale(self) -> str:
        return "Reference: naive O(n²) EWM for the property test."


# ---------------------------------------------------------------------
# Frame builders.
# ---------------------------------------------------------------------


def _build_decay_frame(
    timestamps: list[int],
    cards: list[Any],
    is_fraud: list[int] | None = None,
) -> pd.DataFrame:
    """Build a minimal cleaner-shaped frame for the decay tests."""
    data: dict[str, Any] = {
        "TransactionDT": timestamps,
        "card1": cards,
    }
    if is_fraud is not None:
        data["isFraud"] = is_fraud
    return pd.DataFrame(data)


# ---------------------------------------------------------------------
# `TestExponentialDecayVelocity`: hand-computed correctness.
# ---------------------------------------------------------------------


class TestExponentialDecayVelocity:
    """Per-event EWM math behaves per the spec."""

    def test_empty_history_yields_zero(self) -> None:
        """Single-row frame: every output column is 0.

        First event for any entity has no priors, so every EWM (and
        fraud-EWM) is 0 by definition. This is a trivial case but
        confirms that fit_transform on n=1 doesn't crash on empty
        state and produces the expected zero defaults.
        """
        df = _build_decay_frame(timestamps=[0], cards=["A"], is_fraud=[1])
        gen = ExponentialDecayVelocity(
            entity_cols=["card1"],
            lambdas=[0.05],
            fraud_weighted=True,
            target_col="isFraud",
        )
        out = gen.fit_transform(df)
        assert out["card1_v_ewm_lambda_0.05"].iloc[0] == 0.0
        assert out["card1_fraud_v_ewm_lambda_0.05"].iloc[0] == 0.0

    def test_single_event_end_state_v_equals_one(self) -> None:
        """After fit_transform on a single event, end state has v=1.

        Inspects `_end_state_` directly (not the output frame) because
        the Δt=0 increment can't be observed via `out[col].iloc[0]`
        on a tied-row group — pass-1 reads happen before pass-2 pushes,
        so the public output for the lone row reads 0 (no priors).
        End-state inspection is the cleanest way to verify the push.
        """
        df = _build_decay_frame(timestamps=[0], cards=["A"], is_fraud=[0])
        gen = ExponentialDecayVelocity(
            entity_cols=["card1"],
            lambdas=[0.05],
            fraud_weighted=False,
        )
        gen.fit_transform(df)
        assert gen._end_state_ is not None
        state = gen._end_state_[("card1", 0.05)]["A"]
        assert state.v == pytest.approx(1.0)
        assert state.fraud_v == pytest.approx(0.0)
        assert state.last_t == 0

    def test_dt_half_life_yields_half(self) -> None:
        """λ=0.05/h: at Δt = ln(2)/0.05 hours, v_ewm decays by 0.5.

        Frame: row 0 at T=0; row 1 at T = round(ln(2)/0.05 × 3600)
        seconds = 49906. Row 1's read happens before its push, so
        only row 0 contributes; row 0's weight is exp(-0.05 · 49906/3600)
        ≈ exp(-ln 2) = 0.5 (within float precision and integer-second
        rounding).
        """
        # ln(2) / 0.05 ≈ 13.8629 hours = 49906.4 seconds; round down.
        half_life_seconds = 49906
        df = _build_decay_frame(
            timestamps=[0, half_life_seconds],
            cards=["A", "A"],
            is_fraud=[0, 0],
        )
        gen = ExponentialDecayVelocity(entity_cols=["card1"], lambdas=[0.05], fraud_weighted=False)
        out = gen.fit_transform(df)
        # Hand-computed expected (exact, given integer-second rounding).
        expected_exact = math.exp(-0.05 * half_life_seconds / 3600.0)
        assert out["card1_v_ewm_lambda_0.05"].iloc[1] == pytest.approx(expected_exact, abs=1e-12)
        # And it's within 1e-3 of the analytic 0.5 target — confirming
        # the half-life identity within the integer-second resolution.
        assert out["card1_v_ewm_lambda_0.05"].iloc[1] == pytest.approx(0.5, abs=1e-3)

    def test_nan_entity_yields_zero(self) -> None:
        """NaN entity → output 0; NaN row's contribution NOT pushed.

        Three rows: row 0 entity A, row 1 entity NaN, row 2 entity A.
        Row 1's output is 0 (no entity to compute for). Row 2 sees
        only row 0 (the NaN row's contribution was never pushed), so
        row 2's v_ewm is exp(-0.05 · 2/3600) ≈ 0.99997 (tiny decay
        over 2 seconds), NOT exp(-0.05 · 2/3600) + something from row 1.
        """
        df = _build_decay_frame(
            timestamps=[0, 1, 2],
            cards=["A", None, "A"],
            is_fraud=[0, 0, 0],
        )
        gen = ExponentialDecayVelocity(entity_cols=["card1"], lambdas=[0.05], fraud_weighted=False)
        out = gen.fit_transform(df)
        # Row 1: NaN entity → 0.
        assert out["card1_v_ewm_lambda_0.05"].iloc[1] == 0.0
        # Row 2: sees only row 0; expected v_ewm = exp(-0.05 · 2/3600).
        expected = math.exp(-0.05 * 2 / 3600.0)
        assert out["card1_v_ewm_lambda_0.05"].iloc[2] == pytest.approx(expected, abs=1e-12)

    def test_ties_on_timestamp_excluded(self) -> None:
        """Tied-timestamp rows: each sees pre-tie state, not each other.

        Three rows: T=0 (entity A), T=3600 (entity A), T=3600 (entity A).
        Rows 1 and 2 are tied at T=3600. Pass-1 invariant: both read
        the same pre-tie state, which contains only row 0. Both should
        output exp(-0.05 · 1) ≈ 0.9512294 — neither sees the other.
        """
        df = _build_decay_frame(
            timestamps=[0, 3600, 3600],
            cards=["A", "A", "A"],
            is_fraud=[0, 0, 0],
        )
        gen = ExponentialDecayVelocity(entity_cols=["card1"], lambdas=[0.05], fraud_weighted=False)
        out = gen.fit_transform(df)
        expected = math.exp(-0.05 * 1.0)
        assert out["card1_v_ewm_lambda_0.05"].iloc[1] == pytest.approx(expected, abs=1e-12)
        assert out["card1_v_ewm_lambda_0.05"].iloc[2] == pytest.approx(expected, abs=1e-12)

    def test_input_columns_preserved(self) -> None:
        """All input columns survive transform; expected new columns added."""
        df = _build_decay_frame(
            timestamps=[0, 100],
            cards=["A", "A"],
            is_fraud=[0, 1],
        )
        df["extra_col"] = ["x", "y"]
        gen = ExponentialDecayVelocity(
            entity_cols=["card1"],
            lambdas=[0.05, 0.1],
            fraud_weighted=True,
            target_col="isFraud",
        )
        out = gen.fit_transform(df)
        # Every input column survives.
        for col in df.columns:
            assert col in out.columns
        # 1 entity × 2 λ × 2 signals = 4 new columns.
        expected_new = {
            "card1_v_ewm_lambda_0.05",
            "card1_fraud_v_ewm_lambda_0.05",
            "card1_v_ewm_lambda_0.1",
            "card1_fraud_v_ewm_lambda_0.1",
        }
        assert expected_new.issubset(set(out.columns))

    def test_oof_safety_with_fraud_label(self) -> None:
        """**The OOF gate.** Single-row frame with is_fraud=1.

        The row's own `is_fraud` MUST NOT factor into its own
        `fraud_v_ewm`. Pass-1 reads state before pass-2 pushes, so
        the lone row's pass-1 read finds an empty state and emits
        0.0 exactly. If this test fails, the read-before-push
        ordering has been broken — the most catastrophic possible
        bug for this generator (silent training-time leakage).
        """
        df = _build_decay_frame(timestamps=[0], cards=["A"], is_fraud=[1])
        gen = ExponentialDecayVelocity(
            entity_cols=["card1"],
            lambdas=[0.05],
            fraud_weighted=True,
            target_col="isFraud",
        )
        out = gen.fit_transform(df)
        # Exact 0.0, not approximate — there's no decay or numerical
        # noise involved (the read returns 0 from an empty state).
        assert out["card1_fraud_v_ewm_lambda_0.05"].iloc[0] == 0.0


# ---------------------------------------------------------------------
# `TestTransformVal`: fit-then-transform behaviour.
# ---------------------------------------------------------------------


class TestTransformVal:
    """`transform(val)` decays end-state forward, never pushes labels."""

    def test_fit_then_transform_decays_end_state(self) -> None:
        """End-state decays cleanly to a val row's timestamp.

        Train: 5 rows at T=0, 3600, 7200, 10800, 14400 (1-hour apart),
        all entity A, all not-fraud. End state.v at T=14400 is the
        sum over all 5 events of exp(-0.05 · (14400 - T_i)/3600).
        Val frame: 1 row at T=18000. Expected v_ewm: end_state.v ·
        exp(-0.05 · 1) — the closed-form sum over all 5 train events
        of exp(-0.05 · (18000 - T_i)/3600).
        """
        train = _build_decay_frame(
            timestamps=[0, 3600, 7200, 10800, 14400],
            cards=["A"] * 5,
            is_fraud=[0] * 5,
        )
        gen = ExponentialDecayVelocity(entity_cols=["card1"], lambdas=[0.05], fraud_weighted=False)
        gen.fit(train)
        val = _build_decay_frame(timestamps=[18000], cards=["A"])
        out = gen.transform(val)
        # Closed-form expected value: at T_val = 18000, each train
        # event contributes exp(-0.05 · (18000 - T_i)/3600).
        expected = sum(
            math.exp(-0.05 * (18000 - t) / 3600.0) for t in (0, 3600, 7200, 10800, 14400)
        )
        assert out["card1_v_ewm_lambda_0.05"].iloc[0] == pytest.approx(expected, abs=1e-9)

    def test_transform_unseen_entity_yields_zero(self) -> None:
        """Val rows with entity values not seen at fit time → 0."""
        train = _build_decay_frame(timestamps=[0, 3600], cards=["A", "A"], is_fraud=[0, 0])
        gen = ExponentialDecayVelocity(entity_cols=["card1"], lambdas=[0.05], fraud_weighted=False)
        gen.fit(train)
        val = _build_decay_frame(timestamps=[7200], cards=["B"])
        out = gen.transform(val)
        assert out["card1_v_ewm_lambda_0.05"].iloc[0] == 0.0

    def test_transform_before_fit_raises(self) -> None:
        """`transform` on un-fit instance raises `AttributeError`."""
        gen = ExponentialDecayVelocity(entity_cols=["card1"], lambdas=[0.05], fraud_weighted=False)
        df = _build_decay_frame(timestamps=[0], cards=["A"])
        with pytest.raises(AttributeError, match="must be fit before transform"):
            gen.transform(df)

    def test_transform_backward_time_raises(self) -> None:
        """`transform(val)` with a row earlier than end-state's last_t raises.

        Documents the "fail loudly on broken upstream invariants"
        contract. `temporal_split` should never produce overlapping
        train/val splits, but if a user does pass a backward-time
        frame, the assertion fires immediately rather than silently
        inflating the state.
        """
        train = _build_decay_frame(timestamps=[10000], cards=["A"], is_fraud=[0])
        gen = ExponentialDecayVelocity(entity_cols=["card1"], lambdas=[0.05], fraud_weighted=False)
        gen.fit(train)
        # Val row at T=5000 — earlier than train's last_t=10000.
        val = _build_decay_frame(timestamps=[5000], cards=["A"])
        with pytest.raises(ValueError, match="backward time"):
            gen.transform(val)


# ---------------------------------------------------------------------
# `TestTemporalSafety`: leak gate + naive-vs-optimised property test.
# ---------------------------------------------------------------------


@st.composite
def _frame_strategy_with_fraud(draw: st.DrawFn) -> pd.DataFrame:
    """Generate a small DataFrame with TransactionDT, entity, isFraud.

    Mirrors `test_tier2_velocity._frame_strategy` shape but adds the
    isFraud column needed for `fraud_weighted=True` testing.
    """
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
    fraud = draw(
        st.lists(
            st.integers(min_value=0, max_value=1),
            min_size=n,
            max_size=n,
        )
    )
    return pd.DataFrame({"TransactionDT": timestamps, "entity": entities, "isFraud": fraud})


@st.composite
def _lambdas_strategy(draw: st.DrawFn) -> list[float]:
    """Generate 1-2 distinct λ values in [0.01, 1.0]."""
    n = draw(st.integers(min_value=1, max_value=2))
    return draw(
        st.lists(
            st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=n,
            max_size=n,
            unique=True,
        )
    )


class TestTemporalSafety:
    """Optimised generator's output must be leak-free + match the naive ref."""

    def test_assert_no_future_leak_passes(self) -> None:
        """`assert_no_future_leak` passes on the optimised v_ewm output."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "TransactionDT": np.arange(50, dtype=np.int64) * 60,
                "entity": rng.choice(["A", "B", "C"], size=50),
                "isFraud": rng.integers(0, 2, size=50),
            }
        )
        gen = ExponentialDecayVelocity(
            entity_cols=["entity"],
            lambdas=[0.1],
            fraud_weighted=True,
            target_col="isFraud",
        )
        out = gen.fit_transform(df)

        def _recompute(slice_df: pd.DataFrame) -> pd.Series[Any]:
            recomputed = ExponentialDecayVelocity(
                entity_cols=["entity"],
                lambdas=[0.1],
                fraud_weighted=True,
                target_col="isFraud",
            ).fit_transform(slice_df)
            series = recomputed["entity_v_ewm_lambda_0.1"]
            series.name = "entity_v_ewm_lambda_0.1"
            return series

        assert_no_future_leak(out, df, _recompute)

    @given(df=_frame_strategy_with_fraud(), lambdas=_lambdas_strategy())
    @settings(
        max_examples=50,
        deadline=2000,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_optimized_matches_naive(self, df: pd.DataFrame, lambdas: list[float]) -> None:
        """Optimised running-state EWM matches naive O(n²) ref column-for-column."""
        optimized = ExponentialDecayVelocity(
            entity_cols=["entity"],
            lambdas=lambdas,
            fraud_weighted=True,
            target_col="isFraud",
        )
        naive = _NaiveExponentialDecayVelocity(
            entity_cols=["entity"],
            lambdas=lambdas,
            fraud_weighted=True,
            target_col="isFraud",
        )
        out_opt = optimized.fit_transform(df)
        out_naive = naive.fit_transform(df)
        feature_cols = optimized.get_feature_names()
        # Tolerance: incremental and naive use different summation
        # orders → expect float drift at high precision.
        pd.testing.assert_frame_equal(
            out_opt[feature_cols].reset_index(drop=True).astype(float),
            out_naive[feature_cols].reset_index(drop=True).astype(float),
            check_dtype=False,
            atol=1e-9,
            rtol=1e-9,
        )


# ---------------------------------------------------------------------
# `TestPerformance`: spec contract on 100k rows.
# ---------------------------------------------------------------------


class TestPerformance:
    """Spec contract: 100k × 4 entities × 3 λ × fraud_weighted in <30 s."""

    @pytest.mark.slow
    def test_100k_rows_under_30s(self) -> None:
        """Wall-clock benchmark: 100k rows finishes well under the spec ceiling."""
        n = 100_000
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "TransactionDT": np.arange(n, dtype=np.int64) * 60,
                "card1": rng.integers(0, 10_000, size=n, dtype=np.int64),
                "addr1": rng.integers(0, 1_000, size=n, dtype=np.int64),
                "DeviceInfo": rng.choice(["A", "B", "C", "D"], size=n),
                "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", "hotmail.com"], size=n),
                "isFraud": rng.integers(0, 2, size=n),
            }
        )
        gen = ExponentialDecayVelocity(
            entity_cols=["card1", "addr1", "DeviceInfo", "P_emaildomain"],
            lambdas=[0.05, 0.1, 0.5],
            fraud_weighted=True,
            target_col="isFraud",
        )
        start = time.perf_counter()
        out = gen.fit_transform(df)
        elapsed = time.perf_counter() - start

        # Sanity: every output column landed.
        for col in gen.get_feature_names():
            assert col in out.columns
        assert len(out) == n
        # Headline gate: under the 30 s spec ceiling.
        assert elapsed < 30.0, f"100k rows took {elapsed:.2f}s, ceiling 30s"


# ---------------------------------------------------------------------
# `TestConfigLoad`: YAML defaults + constructor overrides.
# ---------------------------------------------------------------------


class TestConfigLoad:
    """`ExponentialDecayVelocity()` with no args reads `tier4_config.yaml`."""

    def test_default_config_loads(self) -> None:
        """Default constructor reads the YAML; all four keys honoured."""
        gen = ExponentialDecayVelocity()
        assert gen.entity_cols == ("card1", "addr1", "DeviceInfo", "P_emaildomain")
        assert gen.lambdas == (0.05, 0.1, 0.5)
        assert gen.fraud_weighted is True
        assert gen.target_col == "isFraud"
        # 4 entities × 3 λ × 2 signals = 24 columns.
        assert len(gen.get_feature_names()) == 24

    def test_constructor_overrides_config(self) -> None:
        """Explicit kwargs ignore YAML defaults."""
        gen = ExponentialDecayVelocity(
            entity_cols=["card1"],
            lambdas=[0.1],
            fraud_weighted=False,
            target_col="isFraud",
        )
        assert gen.entity_cols == ("card1",)
        assert gen.lambdas == (0.1,)
        assert gen.fraud_weighted is False
        assert gen.get_feature_names() == ["card1_v_ewm_lambda_0.1"]

    def test_duplicate_lambdas_raises(self) -> None:
        """Constructor rejects duplicate λ values fail-fast.

        Duplicates would silently produce duplicate column names
        (the second overwriting the first), wasting the feature
        budget. Documented as Trade-off 9 in the module docstring.
        """
        with pytest.raises(ValueError, match="lambdas must be unique"):
            ExponentialDecayVelocity(
                entity_cols=["card1"],
                lambdas=[0.05, 0.05],
                fraud_weighted=False,
                target_col="isFraud",
            )
