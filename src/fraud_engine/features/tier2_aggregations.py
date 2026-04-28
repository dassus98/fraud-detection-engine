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

# Default config filename. Path resolved at load time against the
# repo root so the generator works regardless of caller CWD.
_DEFAULT_CONFIG_FILENAME: Final[str] = "velocity.yaml"


def _resolve_default_config_path() -> Path:
    """Locate `configs/velocity.yaml` relative to the repo root.

    Mirrors `tier1_basic._resolve_default_config_path` — this file
    lives at `src/fraud_engine/features/tier2_aggregations.py`, so
    the repo root is three parents up.

    Returns:
        Absolute path to the default velocity YAML.
    """
    project_root = Path(__file__).resolve().parents[3]
    return project_root / "configs" / _DEFAULT_CONFIG_FILENAME


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
            cfg = _load_yaml(config_path or _resolve_default_config_path())
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


__all__ = ["VelocityCounter"]
