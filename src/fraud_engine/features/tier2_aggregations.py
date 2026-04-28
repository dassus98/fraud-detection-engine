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
from sklearn.model_selection import StratifiedKFold

from fraud_engine.config.settings import get_settings
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

# ---------------------------------------------------------------------
# `TargetEncoder` defaults.
# ---------------------------------------------------------------------

# Default smoothing strength. With α=10, a category with exactly 10
# observations sits halfway between its own observed rate and the
# global rate. Larger α shrinks toward the global rate; α → ∞
# produces a constant equal to the global rate (tested).
_DEFAULT_SMOOTHING_ALPHA: Final[float] = 10.0

# Default number of OOF folds. 5 is the standard choice in the
# fraud-ML literature; the implementation works for any K ≥ 2.
_DEFAULT_N_SPLITS: Final[int] = 5

# Default target column. Mirrors `data/splits.py:_LABEL_COLUMN`.
_DEFAULT_TARGET_COLUMN: Final[str] = "isFraud"

# Config filename for `TargetEncoder`.
_TARGET_ENCODER_CONFIG_FILENAME: Final[str] = "target_encoder.yaml"


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


class TargetEncoder(BaseFeatureGenerator):
    """Out-of-fold (OOF) target encoder for high-cardinality categoricals.

    For each categorical column, replaces the raw value with a
    smoothed estimate of the target rate conditional on that value.
    The encoded value at training row R uses ONLY data from folds
    that do NOT contain R; at val/test time, a full-train encoder
    fit on all of training is applied. This OOF discipline is the
    only correct defence against catastrophic self-leakage; the
    `tests/integration/test_tier2_no_target_leak.py` shuffled-labels
    gate fails if the discipline is violated.

    Smoothing formula:
        encoded = (sum_target + α × global_rate) / (count + α)

    With α = 0 this reduces to the raw rate. With α large, the
    encoded value shrinks toward the global rate, protecting
    categories with few observations from over-confident estimates.
    With α → ∞ every category encodes to the global rate (tested).

    Business rationale:
        High-cardinality categoricals (`card4`, `addr1`,
        `P_emaildomain`) carry strong fraud signal but are
        impractical to one-hot encode. Target encoding compresses
        each category into a single fraud-risk number that LightGBM
        splits on directly. Sprint 5's serving layer reuses the
        saved encoder over a Redis lookup keyed by category — same
        column-name contract.

    Trade-offs considered:
        - **OOF (StratifiedKFold) on `fit_transform`; full-train
          encoder on `fit` + `transform`.** The override pattern
          keeps the contract compatible with `BaseFeatureGenerator`
          while giving the OOF discipline its required two-mode
          behaviour. `FeaturePipeline.fit_transform` (after the
          2.2.d 1-line fix) calls each generator's `fit_transform`,
          so the override engages naturally inside a pipeline.
        - **Random-stratified KFold within training, NOT
          `TimeSeriesSplit`.** The temporal-discipline boundary in
          this project is at train/val/test (handled by
          `temporal_split`). Within training, OOF is purely a
          self-leakage prevention mechanism; random folds give the
          most stable per-category aggregates. `TimeSeriesSplit`
          would force fold 0 to encode against zero rows of
          training history, which is broken. Stratified-on-target
          ensures each fold has comparable fraud rates.
        - **Fold-specific global_rate as the smoothing prior.** When
          encoding fold k from the OTHER folds' data, the global
          rate used in smoothing is computed from those OTHER folds
          too — never from fold k or full training. Otherwise we'd
          inject a tiny bit of fold-k information into its own
          smoothing prior.
        - **NaN as its own category** (`groupby(col, dropna=False)`).
          For columns like `addr1` and `P_emaildomain` (high null
          rate per the EDA), the fraud rate among null-categorical
          rows is itself a real signal. The cleaner's
          `MissingIndicatorGenerator` captures the binary "is null";
          `TargetEncoder` captures the conditional rate.
        - **Unseen categories at val/test → global rate.** Falls
          out naturally: a category with `total = 0` produces
          `(0 + α × global_rate) / (0 + α) = global_rate`. No
          special-case branch needed.
        - **`fit` does NOT do OOF.** It fits only the full-train
          encoder. A caller doing `enc.fit(train).transform(train)`
          gets the leaked path — same misuse footgun every target
          encoder library has. Documented as: "use `fit_transform`
          on training; `fit` is for the val/test-only path."

    Attributes:
        cat_cols: tuple of categorical column names.
        target_col: target column name.
        alpha: smoothing strength.
        n_splits: number of OOF folds.
        random_state: seed for the StratifiedKFold split.
        mappings_: per-column dict of `{category: encoded_value}`
            for the full-train encoder. None pre-fit.
        global_rates_: per-column full-train target mean. None
            pre-fit. Used as fallback for unseen categories at
            transform time.
    """

    def __init__(  # noqa: PLR0913 — six explicit kwargs keep the YAML-override surface readable; condensing into a config dict would be worse.
        self,
        cat_cols: Sequence[str] | None = None,
        target_col: str | None = None,
        alpha: float | None = None,
        n_splits: int | None = None,
        random_state: int | None = None,
        config_path: Path | None = None,
    ) -> None:
        """Construct the target encoder.

        Args:
            cat_cols: Categorical column names. If `None`, read from
                the YAML's `cat_cols` list.
            target_col: Target column name. If `None`, YAML default
                (`isFraud`).
            alpha: Smoothing strength. If `None`, YAML default (10.0).
            n_splits: Number of OOF folds. If `None`, YAML default (5).
            random_state: Seed for the StratifiedKFold split. If
                `None`, falls back to `get_settings().seed`.
            config_path: Override the default YAML path. Production
                code uses the default; tests may override.
        """
        if cat_cols is None or target_col is None or alpha is None or n_splits is None:
            cfg = _load_yaml(config_path or _resolve_config_path(_TARGET_ENCODER_CONFIG_FILENAME))
            yaml_cat_cols: Sequence[str] = cfg.get("cat_cols", [])
            yaml_target_col: str = cfg.get("target_col", _DEFAULT_TARGET_COLUMN)
            yaml_alpha: float = float(cfg.get("alpha", _DEFAULT_SMOOTHING_ALPHA))
            yaml_n_splits: int = int(cfg.get("n_splits", _DEFAULT_N_SPLITS))
        else:
            yaml_cat_cols = []
            yaml_target_col = _DEFAULT_TARGET_COLUMN
            yaml_alpha = _DEFAULT_SMOOTHING_ALPHA
            yaml_n_splits = _DEFAULT_N_SPLITS

        self.cat_cols: tuple[str, ...] = tuple(cat_cols if cat_cols is not None else yaml_cat_cols)
        self.target_col: str = target_col if target_col is not None else yaml_target_col
        self.alpha: float = alpha if alpha is not None else yaml_alpha
        self.n_splits: int = n_splits if n_splits is not None else yaml_n_splits
        self.random_state: int = random_state if random_state is not None else get_settings().seed

        # Fitted state — populated by `fit` or `fit_transform`.
        self.mappings_: dict[str, dict[Any, float]] | None = None
        self.global_rates_: dict[str, float] | None = None

    # -----------------------------------------------------------------
    # Core helpers.
    # -----------------------------------------------------------------

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Raise if any required column is missing from `df`."""
        missing = sorted({self.target_col, *self.cat_cols} - set(df.columns))
        if missing:
            raise KeyError(f"TargetEncoder: missing required column(s) {missing}")

    def _compute_mapping(self, df: pd.DataFrame, col: str, global_rate: float) -> dict[Any, float]:
        """Compute the smoothed `{category: encoded_value}` mapping.

        `groupby(col, dropna=False)` includes the NaN group as its
        own key (numpy NaN). Downstream `_lookup` handles NaN-key
        retrieval explicitly because `dict.get(np.nan)` does not
        match (NaN != NaN).
        """
        grouped = df.groupby(col, dropna=False)[self.target_col]
        counts = grouped.count()
        sums = grouped.sum()
        smoothed = (sums + self.alpha * global_rate) / (counts + self.alpha)
        return {key: float(val) for key, val in smoothed.items()}

    @staticmethod
    def _lookup(mapping: dict[Any, float], cat_val: Any, fallback: float) -> float:
        """Return `mapping[cat_val]` with NaN-aware lookup; fall back if missing."""
        if pd.isna(cat_val):
            for key, value in mapping.items():
                if pd.isna(key):
                    return value
            return fallback
        return float(mapping.get(cat_val, fallback))

    # -----------------------------------------------------------------
    # `BaseFeatureGenerator` contract.
    # -----------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> Self:
        """Fit the FULL-train encoder. Use for val/test-only path.

        Note: `enc.fit(train).transform(train)` produces leaked
        encodings (same risk every target-encoder library has). For
        training rows, use `fit_transform` to get OOF encoding;
        `fit` + `transform` is for the val/test path.
        """
        self._validate_columns(df)
        global_rate = float(df[self.target_col].mean())
        mappings: dict[str, dict[Any, float]] = {}
        global_rates: dict[str, float] = {}
        for col in self.cat_cols:
            mappings[col] = self._compute_mapping(df, col, global_rate)
            global_rates[col] = global_rate
        self.mappings_ = mappings
        self.global_rates_ = global_rates
        return self

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """OOF-encode training rows; ALSO fit the full-train encoder.

        For each StratifiedKFold split (other_idx, oof_idx):
            1. Compute the fold-specific `global_rate` from `other_idx`.
            2. Compute the smoothed mapping from `other_idx` only.
            3. Apply the mapping to `oof_idx` rows and write into
               the output column at the rows' original positions.

        After OOF, fits the full-train encoder via `self.fit(df)` so
        a downstream `pipeline.transform(val)` call has the encoder
        ready (this is what `FeaturePipeline.fit_transform(train);
        pipeline.transform(val)` relies on).

        Args:
            df: Training frame; must contain `target_col` and every
                column in `cat_cols`.

        Returns:
            `df.copy()` with one new column per `cat_cols` entry,
            named `{col}_target_enc`. Each training row's encoded
            value derives from a slice of training that does NOT
            include the row itself.

        Raises:
            KeyError: If any required column is missing.
        """
        self._validate_columns(df)
        n = len(df)
        feature_names = self.get_feature_names()

        # Pre-allocate with NaN. Any NaN in the output post-OOF is a
        # bug — every training row should be assigned an encoded
        # value by exactly one fold.
        encoded: dict[str, list[float]] = {name: [float("nan")] * n for name in feature_names}

        targets = df[self.target_col].to_numpy()
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        for other_idx, oof_idx in skf.split(np.zeros(n), targets):
            other_df = df.iloc[other_idx]
            oof_df = df.iloc[oof_idx]
            fold_global_rate = float(other_df[self.target_col].mean())

            for col in self.cat_cols:
                mapping = self._compute_mapping(other_df, col, fold_global_rate)
                out_col = f"{col}_target_enc"
                cat_values = oof_df[col].to_numpy()
                for original_pos, cat_val in zip(oof_idx, cat_values, strict=True):
                    encoded[out_col][int(original_pos)] = self._lookup(
                        mapping, cat_val, fold_global_rate
                    )

        # Fit the full-train encoder for later `transform` calls.
        self.fit(df)

        out = df.copy()
        for name, vals in encoded.items():
            out[name] = vals
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the full-train encoder. Used for val / test / serving.

        Does NOT require `target_col` in `df` — this path is used
        for held-out predictions where the target is not available.

        Args:
            df: Frame to transform; must contain every column in
                `cat_cols`.

        Returns:
            `df.copy()` with one new column per `cat_cols` entry.

        Raises:
            AttributeError: If the encoder has not been fit.
            KeyError: If any required cat column is missing from `df`.
        """
        if self.mappings_ is None or self.global_rates_ is None:
            raise AttributeError("TargetEncoder must be fit before transform")
        missing = sorted(set(self.cat_cols) - set(df.columns))
        if missing:
            raise KeyError(f"TargetEncoder.transform: missing column(s) {missing}")

        out = df.copy()
        for col in self.cat_cols:
            mapping = self.mappings_[col]
            global_rate = self.global_rates_[col]
            cat_values = df[col].to_numpy()
            out[f"{col}_target_enc"] = [
                self._lookup(mapping, cat_val, global_rate) for cat_val in cat_values
            ]
        return out

    def get_feature_names(self) -> list[str]:
        """Return the deterministic list of generated column names."""
        return [f"{col}_target_enc" for col in self.cat_cols]

    def get_business_rationale(self) -> str:
        """Return the manifest-rendered business rationale."""
        return (
            "Out-of-fold (OOF) target encoding for high-cardinality "
            "categoricals (card4, addr1, P_emaildomain). Each "
            "training row's encoded value derives from a fold that "
            "does NOT contain the row itself — the only correct "
            "defence against self-leakage. Val / test use a full-train "
            "encoder. Smoothing toward the global rate via "
            "(sum + α × global_rate) / (count + α) protects "
            "low-cardinality categories from over-confident estimates."
        )


__all__ = ["HistoricalStats", "TargetEncoder", "VelocityCounter"]
