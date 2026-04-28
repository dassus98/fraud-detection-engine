"""Tier-2 aggregations: per-entity, time-windowed transaction counts.

Sprint 2 first-Tier-2 generator: `VelocityCounter` produces
`{entity}_velocity_{window_label}` columns counting how many *strictly
earlier* transactions for the same entity fell within a fixed
lookback window. Velocity is the canonical fraud signal — a card that
ran 1 transaction yesterday and 50 in the past hour is overwhelmingly
likely to be compromised, irrespective of any static feature.

Business rationale:
    Every production fraud system in the literature keys on velocity.
    Sprint 3's LightGBM expects velocity columns as headline numeric
    features. Sprint 5's serving layer will re-implement these counts
    against Redis state with the same column names; matching the
    naming contract here lets a Sprint 5 inference call drop into the
    same feature vector slot Sprint 3 trained on.

Trade-offs considered:
    - **Deque-per-entity vs `groupby + rolling`.** A `groupby` with
      `rolling("24h").count()` is `O(n²)` worst-case on this dataset
      (590k rows × per-group sort, with pandas' `rolling` semantics
      that re-walk the window for every row). The deque-based sweep
      is `O(n)` amortised after the initial `O(n log n)` sort: each
      timestamp is pushed and popped at most once per (entity, window)
      pair. Spec mandates the deque approach; benchmark in the
      completion report confirms <30 s on 100k rows.
    - **Strict-`<` ties handling.** Two rows sharing `TransactionDT`
      see neither each other nor the future row; both count only
      strictly-earlier events. Implemented by *batching* tied rows:
      first compute counts for every tied row (each sees only what
      was pushed before the tie), THEN push all tied timestamps to
      the per-entity deques. Without this batching, the second tied
      row would erroneously count the first.
    - **NaN entity values → count = 0.** A row whose entity is null
      contributes no signal, AND its timestamp is not pushed to any
      deque (no entity to key on). This matches what production
      serving would do — a transaction without device info doesn't
      update the device's velocity state. LightGBM splits cleanly on
      0 vs >0; a NaN would force imputation.
    - **Stateless `fit` for batch use.** The batch generator learns
      nothing from training data — every count is computed from the
      current frame's history within the call to `transform`.
      `pipeline.fit_transform(train); pipeline.transform(val)`
      therefore produces val velocity counts that span only val's
      own events. Real serving (Sprint 5) carries train's tail
      forward via Redis state; the *transform* logic transplants
      directly. Documented in the completion report.

Performance contract:
    - 100k rows × 4 entities × 3 windows ≈ 1.2 M deque ops, well
      under the spec's 30 s ceiling on any reasonable machine.
      `test_100k_rows_under_30s` enforces this.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Self

import numpy as np
import pandas as pd
import yaml

from fraud_engine.features.base import BaseFeatureGenerator

# Default temporal column. Mirrors `data/splits.py:_TIME_COLUMN` and
# `temporal_guards.py:_DEFAULT_TIMESTAMP_COL`; the project orders all
# temporal operations by `TransactionDT` (int seconds).
_DEFAULT_TIMESTAMP_COL: Final[str] = "TransactionDT"

# Default amount column for `HistoricalStats`. Same value used by
# `tier1_basic.AmountTransformer`; pinned constant so a future schema
# rename (e.g. `transaction_amount`) updates exactly one place.
_TRANSACTION_AMT_COLUMN: Final[str] = "TransactionAmt"

# Per-class config filenames. Both classes share the resolver helper
# below; the constants are defined here so a rename touches one place.
_VELOCITY_CONFIG_FILENAME: Final[str] = "velocity.yaml"
_HISTORICAL_STATS_CONFIG_FILENAME: Final[str] = "historical_stats.yaml"

# Sample-vs-population std: pandas defaults to ddof=1 (sample); numpy
# defaults to ddof=0 (population). We follow pandas because every
# downstream consumer (Sprint 3 LightGBM, Sprint 4 evaluation) reads
# stats produced by `pd.Series.std()` and the contract should match.
_STD_DDOF: Final[int] = 1

# Stats supported by `HistoricalStats`. Adding more (e.g. `min`,
# `median`, `count_distinct`) requires extending the dispatch table
# inside `HistoricalStats.transform`.
_SUPPORTED_STATS: Final[frozenset[str]] = frozenset({"mean", "std", "max"})

# Sample std requires at least this many observations. Below the
# threshold `pd.Series.std(ddof=1)` returns NaN, so `HistoricalStats`
# leaves the default NaN in place to match.
_MIN_SAMPLES_FOR_STD: Final[int] = 2


def _resolve_config_path(filename: str) -> Path:
    """Resolve `configs/{filename}` relative to the repo root.

    Mirrors `tier1_basic._resolve_default_config_path` — this file
    lives at `src/fraud_engine/features/tier2_aggregations.py`, so
    the repo root is three parents up. Filename-parameterised so
    every Tier-2 generator can share one resolver.

    Args:
        filename: Bare filename under `configs/` (e.g. `velocity.yaml`).

    Returns:
        Absolute path to the requested YAML.
    """
    project_root = Path(__file__).resolve().parents[3]
    return project_root / "configs" / filename


def _load_yaml(path: Path) -> dict[str, Any]:
    """Read and parse a YAML config file.

    Args:
        path: Absolute path to a YAML file.

    Returns:
        The parsed top-level mapping.

    Raises:
        FileNotFoundError: If `path` does not exist.
        TypeError: If the YAML root is not a mapping.
    """
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected top-level mapping in {path}, got {type(loaded).__name__}")
    return loaded


class VelocityCounter(BaseFeatureGenerator):
    """Per-entity transaction counts over fixed lookback windows.

    For each `(entity, window)` pair, emits one feature column named
    `{entity}_velocity_{window_label}` whose value at row R is the
    number of *strictly earlier* rows with the same entity value
    that fall within `window_seconds` of R's timestamp.

    Business rationale:
        Velocity is the strongest single fraud signal in production
        systems. A card that ran 1 transaction yesterday and 50 in
        the past hour is overwhelmingly likely to be compromised
        irrespective of amount, hour, or static features.

    Trade-offs considered:
        - Deque-per-entity sweep (O(n) amortised) vs `groupby+rolling`
          (O(n²) on this dataset). See module docstring.
        - Strict-`<` semantics: tied timestamps see neither each other
          nor the future. Batched-per-tied-group counting handles it.
        - NaN entity → count = 0 (no signal, no state mutation).
        - Stateless `fit`: the batch generator reconstructs state in
          every `transform` call. Online serving (Sprint 5) reuses the
          *transform* logic against Redis; same column-name contract.

    Attributes:
        entity_cols: tuple of column names to aggregate over.
        windows: ordered tuple of `(label, seconds)` pairs. The pair
            list (rather than dict) preserves declaration order so
            `get_feature_names()` is deterministic.
        timestamp_col: temporal ordering column. Default
            `"TransactionDT"`.
    """

    def __init__(
        self,
        entity_cols: Sequence[str] | None = None,
        windows: Mapping[str, int] | None = None,
        timestamp_col: str = _DEFAULT_TIMESTAMP_COL,
        config_path: Path | None = None,
    ) -> None:
        """Construct the velocity counter.

        Args:
            entity_cols: Column names to aggregate over. If `None`,
                read from the YAML's `entities` list.
            windows: Mapping of window label -> seconds. If `None`,
                read from the YAML's `windows` mapping. Iteration
                order of the supplied mapping is preserved.
            timestamp_col: Temporal ordering column. Default
                `"TransactionDT"`.
            config_path: Override the default YAML path. Useful for
                tests; production code uses the default.
        """
        if entity_cols is None or windows is None:
            cfg = _load_yaml(config_path or _resolve_config_path(_VELOCITY_CONFIG_FILENAME))
            yaml_entities: Sequence[str] = cfg.get("entities", [])
            yaml_windows: Mapping[str, int] = cfg.get("windows", {})
        else:
            yaml_entities = []
            yaml_windows = {}

        chosen_entities: Sequence[str] = entity_cols if entity_cols is not None else yaml_entities
        chosen_windows: Mapping[str, int] = windows if windows is not None else yaml_windows

        self.entity_cols: tuple[str, ...] = tuple(chosen_entities)
        self.windows: tuple[tuple[str, int], ...] = tuple(chosen_windows.items())
        self.timestamp_col: str = timestamp_col

    def fit(self, _df: pd.DataFrame) -> Self:
        """Stateless: returns self.

        Velocity has no fitted parameters — the lookback windows and
        entity list are pinned at construction. Sprint 5's online
        serving will reuse `transform` over a Redis-backed timeline
        without changing this batch contract.
        """
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:  # noqa: PLR0912 — tied-group batching is a single algorithm; splitting across helpers would lose locality.
        """Compute per-entity, per-window strictly-past counts.

        Algorithm:
            1. Stable-sort indices by `timestamp_col`.
            2. Iterate sorted positions, batching ties on timestamp.
            3. For each tied group at timestamp T:
               a. *First* compute the count for every tied row using
                  the current per-entity deques (each row sees only
                  events strictly before T).
               b. *Then* push every tied row's timestamp into the
                  appropriate deques.
            4. Per-(entity, window) deque eviction is lazy: pop-left
               while the head's timestamp is older than `T - window`.

        Args:
            df: Frame to transform. Must contain `self.timestamp_col`
                and every column in `self.entity_cols`.

        Returns:
            `df.copy()` with one new column per `(entity, window)`
            pair, named `{entity}_velocity_{window_label}`.

        Raises:
            KeyError: If `self.timestamp_col` or any entity column
                is missing from `df`.
        """
        missing = [c for c in (self.timestamp_col, *self.entity_cols) if c not in df.columns]
        if missing:
            raise KeyError(f"VelocityCounter.transform: missing required column(s) {missing}")

        feature_names = self.get_feature_names()
        n = len(df)

        timestamps = df[self.timestamp_col].to_numpy()
        sort_idx = np.argsort(timestamps, kind="stable")
        sorted_timestamps = timestamps[sort_idx]

        # Pre-extract entity columns as numpy arrays in sorted order;
        # avoids per-row `.iloc` lookups (≈10x faster on a 100k frame).
        sorted_entities: dict[str, np.ndarray[Any, Any]] = {
            entity: df[entity].to_numpy()[sort_idx] for entity in self.entity_cols
        }

        # Pre-allocate result lists positionally; written back at the
        # original row index so output preserves df's row order.
        results: dict[str, list[int]] = {name: [0] * n for name in feature_names}

        # Per-(entity, window_seconds) deque keyed by entity value.
        state: dict[tuple[str, int], dict[Any, deque[int]]] = {
            (entity, secs): defaultdict(deque)
            for entity in self.entity_cols
            for _, secs in self.windows
        }

        i = 0
        while i < n:
            # Identify the run of tied timestamps starting at i.
            j = i
            tie_value = sorted_timestamps[i]
            while j < n and sorted_timestamps[j] == tie_value:
                j += 1
            ts_value = int(tie_value)

            # Pass 1: count for each tied row. The deques hold only
            # events strictly before the current tie group, so no row
            # in [i, j) can see another row in [i, j).
            for k in range(i, j):
                orig_pos = int(sort_idx[k])
                for entity in self.entity_cols:
                    entity_val = sorted_entities[entity][k]
                    if pd.isna(entity_val):
                        continue  # default 0 stays in place
                    for label, secs in self.windows:
                        d = state[(entity, secs)][entity_val]
                        window_start = ts_value - secs
                        while d and d[0] < window_start:
                            d.popleft()
                        col = f"{entity}_velocity_{label}"
                        results[col][orig_pos] = len(d)

            # Pass 2: push every tied row's timestamp into the deques.
            for k in range(i, j):
                for entity in self.entity_cols:
                    entity_val = sorted_entities[entity][k]
                    if pd.isna(entity_val):
                        continue
                    for _, secs in self.windows:
                        state[(entity, secs)][entity_val].append(ts_value)

            i = j

        out = df.copy()
        for name, vals in results.items():
            out[name] = vals
        return out

    def get_feature_names(self) -> list[str]:
        """Return the deterministic list of generated column names.

        Order: outer loop entity_cols; inner loop windows. Both
        orderings are pinned at construction so a manifest written
        at `pipeline.save` time stays stable across re-runs.
        """
        return [
            f"{entity}_velocity_{label}" for entity in self.entity_cols for label, _ in self.windows
        ]

    def get_business_rationale(self) -> str:
        """Return the manifest-rendered business rationale."""
        return (
            "Per-entity transaction counts over fixed lookback "
            "windows. Velocity is the canonical fraud signal — a card "
            "running N transactions in the past hour is far more likely "
            "to be fraudulent than one running 1, irrespective of "
            "amount or hour. Strictly-past semantics avoid look-ahead "
            "leakage; tied timestamps do not see each other."
        )


class HistoricalStats(BaseFeatureGenerator):
    """Per-entity rolling mean / std / max of an amount column.

    For each `(entity, window, stat)` triple, emits one column named
    `{entity}_amt_{stat}_{window_label}` whose value at row R is the
    statistic computed over the amount column for *strictly earlier*
    same-entity rows whose timestamp falls within `window_seconds`
    of R's timestamp.

    Business rationale:
        Mean / std / max summarise an entity's recent spending shape
        in a way velocity counts cannot. A card whose past-30-day
        mean is $40 suddenly seeing a $2000 transaction is far more
        suspicious than the same card if its 30-day mean is $1500.
        Sprint 3's LightGBM splits on these "expected behaviour"
        features; Sprint 5's serving layer reuses the same
        column-name contract over a Redis-backed timeline.

    Trade-offs considered:
        - **Recompute-from-deque vs running-state.** A running-mean +
          running-sum-of-squares + monotonic-max-deque approach is
          O(1) per push/pop, but the bookkeeping is fragile under
          eviction (especially the max — needs a strict-monotonic
          deque). The simpler approach: store `(timestamp, amount)`
          in a per-entity deque and recompute statistics from a
          numpy array on every read. With a 30 d window and typical
          entity activity, deques stay small; numpy vectorises the
          per-row stat call.
        - **Sample std (ddof=1).** Matches `pd.Series.std()` so the
          property test's pandas-based reference and the optimised
          impl agree.
        - **n=1 deque → std = NaN.** Sample std requires ≥ 2 points;
          mean and max still return the single value.
        - **NaN entity → NaN stats.** Different from
          VelocityCounter's "0" — count of 0 is meaningful, mean of
          0 over zero observations is misleading. NaN is the clean
          "no data" indicator for LightGBM.
        - **NaN amount → row not pushed.** Defensive against Sprint
          5's serving layer ingesting an unvalidated payload; the
          cleaner forbids NaN amounts in the train pipeline.
        - **Tied-timestamp batching.** Identical two-pass pattern to
          `VelocityCounter`: pass 1 reads from the deque (no tied
          row sees another); pass 2 pushes every tied row's
          `(ts, amt)` tuple.

    Attributes:
        entity_stats: ordered tuple of `(entity_name, tuple_of_stats)`.
            Order pinned at construction so `get_feature_names()`
            is deterministic.
        windows: ordered tuple of `(label, seconds)` pairs.
        amount_col: column to aggregate.
        timestamp_col: temporal ordering column.
    """

    def __init__(
        self,
        entity_stats: Mapping[str, Sequence[str]] | None = None,
        windows: Mapping[str, int] | None = None,
        amount_col: str | None = None,
        timestamp_col: str = _DEFAULT_TIMESTAMP_COL,
        config_path: Path | None = None,
    ) -> None:
        """Construct the historical-stats generator.

        Args:
            entity_stats: Mapping of entity column -> stat names.
                If `None`, read from the YAML's `entities` mapping.
            windows: Mapping of window label -> seconds. If `None`,
                read from the YAML's `windows`.
            amount_col: Column to aggregate. If `None`, read from
                YAML's `amount_col` (defaults to `TransactionAmt`).
            timestamp_col: Temporal ordering column. Default
                `"TransactionDT"`.
            config_path: Override the default YAML path. Useful for
                tests; production code uses the default.

        Raises:
            ValueError: If any declared stat is not in
                `_SUPPORTED_STATS`.
        """
        if entity_stats is None or windows is None or amount_col is None:
            cfg = _load_yaml(config_path or _resolve_config_path(_HISTORICAL_STATS_CONFIG_FILENAME))
            yaml_entities: Mapping[str, Mapping[str, Any]] = cfg.get("entities", {})
            yaml_windows: Mapping[str, int] = cfg.get("windows", {})
            yaml_amount_col: str = cfg.get("amount_col", _TRANSACTION_AMT_COLUMN)
        else:
            yaml_entities = {}
            yaml_windows = {}
            yaml_amount_col = _TRANSACTION_AMT_COLUMN

        if entity_stats is None:
            chosen_entity_stats: list[tuple[str, tuple[str, ...]]] = [
                (entity, tuple(spec.get("stats", []))) for entity, spec in yaml_entities.items()
            ]
        else:
            chosen_entity_stats = [(e, tuple(s)) for e, s in entity_stats.items()]

        chosen_windows: Mapping[str, int] = windows if windows is not None else yaml_windows
        chosen_amount: str = amount_col if amount_col is not None else yaml_amount_col

        # Validate stat names BEFORE storing — fail fast with a clear message.
        for entity, stats in chosen_entity_stats:
            unsupported = set(stats) - _SUPPORTED_STATS
            if unsupported:
                raise ValueError(
                    f"HistoricalStats: entity {entity!r} declares unsupported "
                    f"stats {sorted(unsupported)}; supported: {sorted(_SUPPORTED_STATS)}"
                )

        self.entity_stats: tuple[tuple[str, tuple[str, ...]], ...] = tuple(chosen_entity_stats)
        self.windows: tuple[tuple[str, int], ...] = tuple(chosen_windows.items())
        self.amount_col: str = chosen_amount
        self.timestamp_col: str = timestamp_col

    def fit(self, _df: pd.DataFrame) -> Self:
        """Stateless: returns self.

        State for online serving is reconstructed inside `transform`
        per call. Sprint 5's online path will reuse `transform` over
        a Redis-backed event log.
        """
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:  # noqa: PLR0912, PLR0915 — tied-group batching is a single algorithm; splitting across helpers would lose locality.
        """Compute per-entity, per-window strictly-past statistics.

        Algorithm (mirrors `VelocityCounter` but with a richer
        deque payload):
            1. Stable-sort by `timestamp_col`; pre-extract entity
               and amount arrays in sorted order.
            2. Iterate sorted positions, batching ties on timestamp.
            3. For each tied group at timestamp T:
               a. Pass 1 (stats): For each tied row × entity ×
                  window, evict deque heads older than `T - secs`,
                  build a numpy array of remaining amounts, dispatch
                  the configured stats. NaN entity → leave NaN
                  default; empty deque → leave NaN.
               b. Pass 2 (push): Push every tied row's
                  `(ts, amt)` tuple into the appropriate deques.
                  Skip NaN amounts (defensive) and NaN entities.

        Args:
            df: Frame to transform. Must contain `self.timestamp_col`,
                `self.amount_col`, and every entity column.

        Returns:
            `df.copy()` with one new column per (entity, stat, window)
            triple, named `{entity}_amt_{stat}_{label}`.

        Raises:
            KeyError: If any required column is missing from `df`.
        """
        required = {
            self.timestamp_col,
            self.amount_col,
            *(e for e, _ in self.entity_stats),
        }
        missing = sorted(required - set(df.columns))
        if missing:
            raise KeyError(f"HistoricalStats.transform: missing required column(s) {missing}")

        feature_names = self.get_feature_names()
        n = len(df)

        timestamps = df[self.timestamp_col].to_numpy()
        amounts = df[self.amount_col].to_numpy()
        sort_idx = np.argsort(timestamps, kind="stable")
        sorted_timestamps = timestamps[sort_idx]
        sorted_amounts = amounts[sort_idx]
        sorted_entities: dict[str, np.ndarray[Any, Any]] = {
            entity: df[entity].to_numpy()[sort_idx] for entity, _ in self.entity_stats
        }

        # NaN default for every output cell.
        results: dict[str, list[float]] = {name: [float("nan")] * n for name in feature_names}

        # state[(entity, secs)][entity_value] = deque[(timestamp, amount)]
        state: dict[tuple[str, int], dict[Any, deque[tuple[int, float]]]] = {
            (entity, secs): defaultdict(deque)
            for entity, _ in self.entity_stats
            for _, secs in self.windows
        }

        i = 0
        while i < n:
            j = i
            tie_value = sorted_timestamps[i]
            while j < n and sorted_timestamps[j] == tie_value:
                j += 1
            ts_value = int(tie_value)

            # Pass 1: compute stats from current deque contents.
            for k in range(i, j):
                orig_pos = int(sort_idx[k])
                for entity, stats in self.entity_stats:
                    entity_val = sorted_entities[entity][k]
                    if pd.isna(entity_val):
                        continue
                    for label, secs in self.windows:
                        d = state[(entity, secs)][entity_val]
                        window_start = ts_value - secs
                        while d and d[0][0] < window_start:
                            d.popleft()
                        if not d:
                            continue
                        arr = np.fromiter((a for _, a in d), dtype=np.float64, count=len(d))
                        for stat in stats:
                            col = f"{entity}_amt_{stat}_{label}"
                            if stat == "mean":
                                results[col][orig_pos] = float(arr.mean())
                            elif stat == "max":
                                results[col][orig_pos] = float(arr.max())
                            elif stat == "std" and arr.size >= _MIN_SAMPLES_FOR_STD:
                                results[col][orig_pos] = float(arr.std(ddof=_STD_DDOF))

            # Pass 2: push tied rows' (ts, amt) into deques.
            for k in range(i, j):
                amt = sorted_amounts[k]
                if pd.isna(amt):
                    continue
                amt_value = float(amt)
                for entity, _ in self.entity_stats:
                    entity_val = sorted_entities[entity][k]
                    if pd.isna(entity_val):
                        continue
                    for _, secs in self.windows:
                        state[(entity, secs)][entity_val].append((ts_value, amt_value))

            i = j

        out = df.copy()
        for name, vals in results.items():
            out[name] = vals
        return out

    def get_feature_names(self) -> list[str]:
        """Return the deterministic list of generated column names.

        Order: outer loop entity_stats (declaration order); middle
        loop stats (declaration order); inner loop windows. All three
        orderings are pinned at construction so the manifest is
        stable across re-runs.
        """
        return [
            f"{entity}_amt_{stat}_{label}"
            for entity, stats in self.entity_stats
            for stat in stats
            for label, _ in self.windows
        ]

    def get_business_rationale(self) -> str:
        """Return the manifest-rendered business rationale."""
        return (
            "Per-entity rolling mean / std / max of the amount column "
            "over fixed lookback windows. Captures expected-spending "
            "shape — a card whose 30-day mean is $40 suddenly seeing "
            "$2000 is suspicious; the same card with a 30-day mean of "
            "$1500 is not. Strict-past semantics avoid look-ahead leakage."
        )


__all__ = ["HistoricalStats", "VelocityCounter"]
