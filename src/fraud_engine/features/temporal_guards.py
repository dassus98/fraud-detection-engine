"""Temporal-safety guards: enforcement against look-ahead leakage.

Look-ahead leakage — a feature for a transaction at time T accidentally
incorporates information from time > T — is the most common bug in
fraud ML pipelines and the subtlest, because the offending feature
typically *does* improve validation AUC (since the validation rows
have look-ahead access to their own labels' near-neighbours). The
model that ships then craters in production. This module is the
enforcement layer that catches such bugs at test time.

Two primitives:

- `assert_no_future_leak(feature_df, source_df, feature_func, ...)` —
  sample-based assertion. Picks `n_samples` random rows in
  `feature_df`, recomputes each row's feature value from
  `source_df` rows with timestamp ≤ that row's timestamp, and
  asserts the recomputed value matches what `feature_df` already
  carries. Used in lineage tests to validate Tier 2-5 generators.
- `TemporalSafeGenerator` — abstract subclass of
  `BaseFeatureGenerator` whose default `transform` iterates rows
  in temporal order and dispatches to a subclass-implemented
  `_compute_for_row(row, past_df)`. `past_df` is sliced with strict
  `<` so the row at T cannot see itself; by construction the
  subclass cannot leak. Slow (O(n²) reference shape), used as a
  development scaffold and as a correctness oracle for vectorized
  optimisations.

Business rationale:
    Without an enforcement layer, leakage bugs surface only after
    the model is deployed — a senior reviewer flagged this exact
    failure mode on the previous iteration of this project. Catching
    leakage at test time, before any feature ships, is cheap;
    catching it in production is catastrophic. The sample-based
    assertion is the project's universal lineage check for any
    time-windowed feature.

Trade-offs considered:
    - **Sample-based vs exhaustive recompute.** Exhaustive
      recomputation (every row, every feature) is O(N²) for an
      N-row frame, infeasible at 590k. Sampling 50 rows uniformly
      catches any *systematic* leak with overwhelming probability —
      a generator that leaks for some rows but not others is
      vanishingly rare in practice (the bug is in the formula, not
      the data).
    - **Strict `<` in the row-iterating generator vs `≤` in the
      assertion.** `TemporalSafeGenerator.transform` uses strict
      `<` — the row at T does NOT see itself, which matches how
      a real-time serving system would work (you compute features
      for transaction T using only transactions strictly before T).
      `assert_no_future_leak` recomputes with `≤` because some
      features include the current row in their definition (e.g.
      `log(amount)` of the current transaction); the assertion's
      contract is "the recomputed value when given all data up to
      and including T must match the original." The two conventions
      are intentionally different.
    - **Iterative O(n²) `transform` over a vectorized version.**
      Vectorized rolling-window operations in pandas / numpy require
      careful indexing to avoid look-ahead; the iterative version
      is provably correct because `past_df` is a plain slice. This
      module's `TemporalSafeGenerator` is the reference; tier 2+
      generators may ship vectorized versions and validate against
      this one via `assert_no_future_leak`.
    - **Float comparison via `np.isclose`** rather than exact
      equality. Recomputed values can differ from the original by
      floating-point noise (different summation order, etc.); a
      strict `==` would generate false positives. Tolerances
      (`rtol=1e-9`, `atol=1e-12`) are tight enough that genuine
      drift surfaces but loose enough to absorb noise.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Final, Self

import numpy as np
import pandas as pd

from fraud_engine.features.base import BaseFeatureGenerator
from fraud_engine.utils.logging import get_logger

# Default temporal column. Mirrors `data/splits.py:_TIME_COLUMN`; the
# project orders all temporal operations by `TransactionDT` (int
# seconds), which is the only stable ordering signal in IEEE-CIS.
_DEFAULT_TIMESTAMP_COL: Final[str] = "TransactionDT"

# Default sample count for `assert_no_future_leak`. 50 catches any
# systematic leak with overwhelming probability while keeping
# per-test runtime <1 s on synthetic data and <10 s on a 10k frame.
_DEFAULT_N_SAMPLES: Final[int] = 50

# Default RNG seed for sample selection. Same value as the project's
# canonical `Settings.seed` default so failures are reproducible
# without extra plumbing.
_DEFAULT_SEED: Final[int] = 42

# Float tolerance for comparing recomputed vs original feature values.
# `np.isclose` defaults are looser; we tighten them slightly so genuine
# leakage (which produces meaningful numerical drift) surfaces while
# floating-point noise (different summation order, etc.) is absorbed.
_RTOL: Final[float] = 1e-9
_ATOL: Final[float] = 1e-12

_logger = get_logger(__name__)


def _values_match(expected: Any, actual: Any) -> bool:
    """Return True if `expected` and `actual` are numerically / value equal.

    NaN-NaN compares as a match (both genuinely missing). Numeric
    values use `np.isclose` with the module's tolerances; non-numeric
    values fall back to `==`.
    """
    expected_is_nan = isinstance(expected, float) and np.isnan(expected)
    actual_is_nan = isinstance(actual, float) and np.isnan(actual)
    if expected_is_nan and actual_is_nan:
        return True
    if expected_is_nan != actual_is_nan:
        return False
    if isinstance(expected, int | float | np.integer | np.floating) and isinstance(
        actual, int | float | np.integer | np.floating
    ):
        return bool(np.isclose(expected, actual, rtol=_RTOL, atol=_ATOL, equal_nan=True))
    return bool(expected == actual)


def assert_no_future_leak(  # noqa: PLR0913 — public API; six explicit params keep call sites readable.
    feature_df: pd.DataFrame,
    source_df: pd.DataFrame,
    feature_func: Callable[[pd.DataFrame], pd.Series[Any]],
    timestamp_col: str = _DEFAULT_TIMESTAMP_COL,
    n_samples: int = _DEFAULT_N_SAMPLES,
    seed: int = _DEFAULT_SEED,
) -> None:
    """Sample n_samples rows; recompute each from past-only source_df; assert match.

    For each sampled row at index ``idx`` with timestamp ``T``,
    restricts ``source_df`` to rows with ``timestamp <= T`` and calls
    ``feature_func`` on that subset. The recomputed value at ``idx``
    must match the original feature value in ``feature_df``; a
    mismatch indicates the original computation peeked at data with
    timestamp > T (look-ahead leakage).

    Business rationale:
        Look-ahead bugs typically appear silently — the offending
        feature improves validation AUC (because val rows have access
        to their own near-neighbours' labels) but the deployed model
        cratters. This assertion catches the bug at test time,
        before deployment. Universal lineage check for any
        time-windowed feature added in Sprint 2 onwards.

    Trade-offs considered:
        - Sample-based (50 rows) vs exhaustive (every row). Exhaustive
          is O(N²) and infeasible on 590k rows; sample-based catches
          any *systematic* leak (a bug in the feature formula) with
          overwhelming probability.
        - `<=` in the recomputation vs `<` strict. Some features
          include the current row by definition (`log(amount)`);
          using `<` would force every test to special-case "the
          row at T isn't in the recomputed slice." `<=` keeps the
          assertion semantics simple: "the feature value at T must
          be recomputable from data up to and including T."

    Args:
        feature_df: DataFrame containing the already-computed feature
            column. Index must align with ``source_df.index``.
        source_df: Source data the feature was derived from. Must
            contain ``timestamp_col``.
        feature_func: Callable that takes a DataFrame slice and
            returns a `pd.Series` of feature values. The Series MUST
            have ``.name`` set to a column name in ``feature_df``;
            this is how the assertion knows which column to compare
            against.
        timestamp_col: Name of the temporal ordering column. Default
            ``"TransactionDT"`` (the project's canonical ordering
            signal).
        n_samples: How many rows to sample. Clamped to
            ``len(feature_df)`` if smaller. Default 50.
        seed: RNG seed for sample selection. Same value across runs
            picks the same rows.

    Raises:
        ValueError: ``feature_func`` returns an unnamed Series, or
            ``timestamp_col`` is missing from either frame.
        AssertionError: a sampled row's recomputed value disagrees
            with the original. Message includes row index, timestamp,
            feature name, expected and actual values.
    """
    if timestamp_col not in feature_df.columns:
        raise ValueError(f"timestamp_col {timestamp_col!r} not in feature_df.columns")
    if timestamp_col not in source_df.columns:
        raise ValueError(f"timestamp_col {timestamp_col!r} not in source_df.columns")

    n_actual = min(n_samples, len(feature_df))
    if n_actual == 0:
        return
    sample = feature_df.sample(n=n_actual, random_state=seed)

    for raw_idx, row in sample.iterrows():
        # `iterrows` yields `Hashable` for the index; pandas-stubs `.loc`
        # overloads want a narrower type, so coerce once at top-of-loop.
        idx: Any = raw_idx
        ts_value = row[timestamp_col]
        past_inclusive = source_df[source_df[timestamp_col] <= ts_value]
        if past_inclusive.empty:
            _logger.debug(
                "assert_no_future_leak.skip_no_past",
                idx=idx,
                timestamp=ts_value,
            )
            continue

        recomputed = feature_func(past_inclusive)
        if not isinstance(recomputed, pd.Series):
            raise ValueError(f"feature_func must return pd.Series; got {type(recomputed).__name__}")
        if recomputed.name is None:
            raise ValueError(
                "feature_func returned an unnamed pd.Series; set Series.name "
                "to the matching column in feature_df."
            )
        # Cast to Any: pandas column labels are Hashable, but `.loc[row, col]`
        # stubs only accept narrower types.
        feature_name: Any = recomputed.name

        expected = recomputed.loc[idx] if idx in recomputed.index else recomputed.iloc[-1]
        actual = feature_df.loc[idx, feature_name]

        if not _values_match(expected, actual):
            raise AssertionError(
                f"Look-ahead leak at idx={idx}, {timestamp_col}={ts_value}, "
                f"feature={feature_name}: expected={expected!r}, got={actual!r}"
            )


class TemporalSafeGenerator(BaseFeatureGenerator, ABC):
    """Row-iterating `BaseFeatureGenerator` that is leak-free by construction.

    Concrete subclasses implement ``_compute_for_row(row, past_df)``;
    the base provides a concrete ``transform`` that iterates rows in
    temporal order and slices ``past_df`` with strict ``<`` so the
    row at T cannot see itself. ``fit`` defaults to a stateless no-op;
    subclasses override for stateful generators (e.g. learning a
    per-entity decay table on the train fold).

    Contract:
        - ``_compute_for_row`` MUST return a dict whose keys equal
          ``set(self.get_feature_names())``. Mismatches surface as
          ``KeyError`` when the transform writes back results.
        - Subclasses MUST still implement ``get_feature_names`` and
          ``get_business_rationale`` (inherited from
          `BaseFeatureGenerator` as abstract).
        - ``timestamp_col`` defaults to ``"TransactionDT"``;
          subclasses override the class attribute if they need a
          different column.

    Business rationale:
        Provides a "slow but provably correct" reference for any
        new tier 2-5 generator. A vectorized implementation can be
        validated against this reference via ``assert_no_future_leak``;
        if the two disagree, the vectorized version has a bug.

    Trade-offs considered:
        - O(n²) iteration vs O(n) vectorized. Iteration is the price
          of mechanical leak-freedom — every call passes only a
          ``past_df`` slice, never the full frame, so even a buggy
          subclass cannot leak. For the 10k-sample lineage tests
          this is fast enough; for full-data feature builds the
          subclass should ship a vectorized override.
        - `Self` return on ``fit`` (mirrors `BaseFeatureGenerator`'s
          contract).

    Attributes:
        timestamp_col: Name of the temporal ordering column.
    """

    timestamp_col: str = _DEFAULT_TIMESTAMP_COL

    @abstractmethod
    def _compute_for_row(self, row: pd.Series[Any], past_df: pd.DataFrame) -> dict[str, Any]:
        """Compute the feature(s) for one row.

        ``past_df`` is guaranteed to contain only rows with timestamp
        STRICTLY less than ``row[self.timestamp_col]`` — the row at T
        cannot see itself. May be empty (first row in temporal order).

        Args:
            row: The row whose features we're producing. Includes
                the timestamp column.
            past_df: Strictly-past slice of the source frame.

        Returns:
            Dict mapping feature_name → value. Keys MUST equal
            ``set(self.get_feature_names())``.
        """

    def fit(self, _df: pd.DataFrame) -> Self:
        """Default: stateless. Subclasses override to learn fitted state."""
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Iterate rows in temporal order; emit feature columns.

        Sorts indirectly so the original row order is preserved in
        the output. Each row's features are computed from a strictly
        earlier slice of ``df``; ties on timestamp are excluded from
        the past slice (strict ``<``).

        Args:
            df: Frame to transform. Must contain ``self.timestamp_col``.

        Returns:
            ``df.copy()`` with one column per name in
            ``self.get_feature_names()``.

        Raises:
            KeyError: ``self.timestamp_col`` not in ``df.columns``.
        """
        if self.timestamp_col not in df.columns:
            raise KeyError(
                f"{self.timestamp_col!r} not in df.columns; "
                f"TemporalSafeGenerator needs the timestamp column."
            )
        feature_names = self.get_feature_names()

        ts = df[self.timestamp_col].to_numpy()
        sort_idx = np.argsort(ts, kind="stable")
        sorted_ts = ts[sort_idx]

        # Pre-allocate result lists positionally; we write back at the
        # original row index so the output preserves df's row order.
        results: dict[str, list[Any]] = {name: [None] * len(df) for name in feature_names}
        df_sorted = df.iloc[sort_idx]

        for sorted_pos in range(len(df_sorted)):
            row = df_sorted.iloc[sorted_pos]
            ts_value = sorted_ts[sorted_pos]
            past_count = int(np.searchsorted(sorted_ts, ts_value, side="left"))
            past_df = df_sorted.iloc[:past_count]
            features = self._compute_for_row(row, past_df)
            orig_pos = int(sort_idx[sorted_pos])
            for name in feature_names:
                results[name][orig_pos] = features[name]

        out = df.copy()
        for name in feature_names:
            out[name] = results[name]
        return out


__all__ = ["TemporalSafeGenerator", "assert_no_future_leak"]
