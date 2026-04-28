"""Tier-3 behavioural-deviation features and a cold-start helper.

Where Tier 2 captures velocity (raw counts) and rolling stats at the
*entity* level, Tier 3 asks the more nuanced question: **does this
transaction look anomalous given the cardholder's prior behaviour?**

`BehavioralDeviation` produces five features, all keyed on `card1`:

- `amt_zscore_vs_card1_history` — z-score of the current amount vs the
  card's prior amount distribution.
- `time_since_last_txn_zscore` — z-score of the current inter-arrival
  time vs the card's prior inter-arrival distribution.
- `addr_change_flag` — 1 if `addr1` differs from the card's mode
  `addr1` over its prior history.
- `device_change_flag` — 1 if `DeviceInfo` is not in the card's set of
  prior devices.
- `hour_deviation` — `abs(current hour − card's mean hour over priors)`.

`ColdStartHandler` is a thin sibling that emits `is_coldstart_{entity}`
flags so downstream models can identify rows where there isn't enough
history for behavioural features to be meaningful.

Business rationale:
    Velocity tells you "this card is hot"; behavioural deviation tells
    you "this transaction is unusual for THIS specific card". A card
    averaging $40 transactions suddenly seeing $2000 is suspicious;
    the same $2000 on a card averaging $1500 is not. These features
    drive a meaningful chunk of Sprint 3's tuning headroom and Sprint
    4's economic-cost analysis.

Trade-offs considered:
    - **Unbounded history vs windowed.** Spec says "card1 history"
      without a window; we accumulate state forever per card. For
      590k rows × ~14k unique cards, per-card running scalars are
      cheap (~MB-scale memory). Far cheaper than retaining a deque
      of every prior amount.
    - **Sample std (ddof=1) for z-scores.** Matches `HistoricalStats`;
      allows direct comparison of "card-level deviation" against
      "entity-level rolling deviation".
    - **First-event fallback = 0.0** (not NaN). 0 means "exactly the
      mean" → "not anomalous". LightGBM splits on 0 cleanly. The
      `ColdStartHandler`'s explicit `is_coldstart_card1` flag lets
      models know to weight these zeros differently. NaN would
      force imputation downstream.
    - **Tied-timestamp two-pass batching** (same as `VelocityCounter`).
      Tied rows for the same card see the same prior state; pass 2
      then updates state with each tied row's contribution. Strict-`<`
      semantics fall out naturally.
    - **NaN amount / addr / device.** Defensive: don't push to running
      state; in pass 1, fall back to 0 / no-change for the affected
      feature. Matches `HistoricalStats` and production fraud-system
      semantics where missing fields don't propagate as anomalies.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from math import sqrt
from pathlib import Path
from typing import Any, Final, Self

import numpy as np
import pandas as pd
import yaml

from fraud_engine.features.base import BaseFeatureGenerator

# ---------------------------------------------------------------------
# Module constants.
# ---------------------------------------------------------------------

_DEFAULT_TIMESTAMP_COL: Final[str] = "TransactionDT"
_DEFAULT_AMOUNT_COL: Final[str] = "TransactionAmt"
_DEFAULT_ENTITY_COL: Final[str] = "card1"
_DEFAULT_ADDR_COL: Final[str] = "addr1"
_DEFAULT_DEVICE_COL: Final[str] = "DeviceInfo"
_DEFAULT_HOUR_COL: Final[str] = "hour_of_day"
_DEFAULT_EPSILON: Final[float] = 1.0e-9

# Per-class config filenames. Both classes share the resolver helper
# below; renaming one filename touches one constant.
_BEHAVIORAL_CONFIG_FILENAME: Final[str] = "behavioral_deviation.yaml"
_COLDSTART_CONFIG_FILENAME: Final[str] = "coldstart.yaml"

# ColdStartHandler default. Smallest window where running mean / std /
# mode start to stabilise. Hits ~76% of cards within their first few
# transactions per the EDA.
_DEFAULT_MIN_HISTORY: Final[int] = 3

# Sample-std formula: variance = (sum_sq − n × mean²) / (n − 1).
# Below this floor the formula is undefined → fallback z = 0.
_MIN_SAMPLES_FOR_STD: Final[int] = 2

# `time_since_last_txn_zscore` needs ≥ 2 prior deltas to define a
# distribution. With 1 prior event there's only 1 delta; std undefined.
_MIN_DELTAS_FOR_TIME_STD: Final[int] = 2

# Output column names. Pinned constants so Sprint 5's serving layer
# can read these without inferring from generator defaults.
_AMT_Z_COL: Final[str] = "amt_zscore_vs_card1_history"
_TIME_Z_COL: Final[str] = "time_since_last_txn_zscore"
_ADDR_CHANGE_COL: Final[str] = "addr_change_flag"
_DEVICE_CHANGE_COL: Final[str] = "device_change_flag"
_HOUR_DEV_COL: Final[str] = "hour_deviation"


# ---------------------------------------------------------------------
# YAML helpers (private).
# ---------------------------------------------------------------------


def _resolve_config_path(filename: str) -> Path:
    """Resolve `configs/{filename}` relative to the repo root.

    Mirrors `tier2_aggregations._resolve_config_path` — this file lives
    at `src/fraud_engine/features/tier3_behavioral.py`, so the repo
    root is three parents up.
    """
    project_root = Path(__file__).resolve().parents[3]
    return project_root / "configs" / filename


def _load_yaml(path: Path) -> dict[str, Any]:
    """Read and parse a YAML config file.

    Raises:
        FileNotFoundError: If `path` does not exist.
        TypeError: If the YAML root is not a mapping.
    """
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected top-level mapping in {path}, got {type(loaded).__name__}")
    return loaded


# ---------------------------------------------------------------------
# Per-card running state for `BehavioralDeviation`.
# ---------------------------------------------------------------------


def _new_card_state() -> dict[str, Any]:
    """Construct the empty per-card running-state dict."""
    return {
        "count": 0,
        "sum_amt": 0.0,
        "sum_amt_sq": 0.0,
        "prev_ts": None,
        "delta_count": 0,
        "sum_delta": 0.0,
        "sum_delta_sq": 0.0,
        "addr_counter": Counter(),
        "device_set": set(),
        "sum_hour": 0.0,
    }


def _sample_std(n: int, sum_x: float, sum_x_sq: float) -> float:
    """Compute sample (ddof=1) std from running scalars.

    Returns NaN if `n < _MIN_SAMPLES_FOR_STD`. Clamps numerical-noise
    negatives in the variance to 0 before sqrt.
    """
    if n < _MIN_SAMPLES_FOR_STD:
        return float("nan")
    mean = sum_x / n
    variance = (sum_x_sq - n * mean * mean) / (n - 1)
    return sqrt(max(variance, 0.0))


# ---------------------------------------------------------------------
# `BehavioralDeviation`.
# ---------------------------------------------------------------------


class BehavioralDeviation(BaseFeatureGenerator):
    """Per-card1 behavioural-deviation features.

    Five strictly-past features per row, all keyed on `card1`:

    - `amt_zscore_vs_card1_history`
    - `time_since_last_txn_zscore`
    - `addr_change_flag`
    - `device_change_flag`
    - `hour_deviation`

    See module docstring for business rationale and trade-offs. The
    `transform` algorithm is the tied-group two-pass pattern from
    `VelocityCounter` / `HistoricalStats`: pass 1 reads from per-card
    state (so all tied rows for the same card see the same prior
    state); pass 2 sequentially updates state with each tied row's
    contribution.

    Attributes:
        entity_col, amount_col, addr_col, device_col, hour_col,
            timestamp_col: input column names.
        epsilon: smoothing constant in z-score denominators.
    """

    def __init__(  # noqa: PLR0913 — eight kwargs reflect the eight column-name overrides; condensing into a config dict would be worse for callers.
        self,
        entity_col: str | None = None,
        amount_col: str | None = None,
        addr_col: str | None = None,
        device_col: str | None = None,
        hour_col: str | None = None,
        timestamp_col: str | None = None,
        epsilon: float | None = None,
        config_path: Path | None = None,
    ) -> None:
        """Construct the deviation generator.

        Each kwarg falls back to the YAML default when `None`.
        """
        if (
            entity_col is None
            or amount_col is None
            or addr_col is None
            or device_col is None
            or hour_col is None
            or timestamp_col is None
            or epsilon is None
        ):
            cfg = _load_yaml(config_path or _resolve_config_path(_BEHAVIORAL_CONFIG_FILENAME))
        else:
            cfg = {}

        self.entity_col: str = (
            entity_col if entity_col is not None else cfg.get("entity_col", _DEFAULT_ENTITY_COL)
        )
        self.amount_col: str = (
            amount_col if amount_col is not None else cfg.get("amount_col", _DEFAULT_AMOUNT_COL)
        )
        self.addr_col: str = (
            addr_col if addr_col is not None else cfg.get("addr_col", _DEFAULT_ADDR_COL)
        )
        self.device_col: str = (
            device_col if device_col is not None else cfg.get("device_col", _DEFAULT_DEVICE_COL)
        )
        self.hour_col: str = (
            hour_col if hour_col is not None else cfg.get("hour_col", _DEFAULT_HOUR_COL)
        )
        self.timestamp_col: str = (
            timestamp_col
            if timestamp_col is not None
            else cfg.get("timestamp_col", _DEFAULT_TIMESTAMP_COL)
        )
        self.epsilon: float = (
            float(epsilon) if epsilon is not None else float(cfg.get("epsilon", _DEFAULT_EPSILON))
        )

    def fit(self, _df: pd.DataFrame) -> Self:
        """Stateless: returns self.

        Per-card running state is rebuilt inside every `transform` call
        (matches `VelocityCounter` / `HistoricalStats` semantics).
        Sprint 5's online path will reuse `transform` over a Redis-
        backed event log.
        """
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:  # noqa: PLR0912, PLR0915 — tied-group batching across 5 features × per-card state is a single algorithm; splitting across helpers would lose locality.
        """Compute the five strictly-past behavioural deviations.

        See module docstring for the algorithm sketch. Tied timestamps
        are processed two-pass: pass 1 reads pre-tied state for every
        tied row, pass 2 then updates state with each tied row's
        contribution.

        Raises:
            KeyError: If any required column is missing from `df`.
        """
        required = {
            self.timestamp_col,
            self.entity_col,
            self.amount_col,
            self.addr_col,
            self.device_col,
            self.hour_col,
        }
        missing = sorted(required - set(df.columns))
        if missing:
            raise KeyError(f"BehavioralDeviation.transform: missing required column(s) {missing}")

        n = len(df)
        feature_names = self.get_feature_names()
        results: dict[str, list[float]] = {name: [0.0] * n for name in feature_names}

        timestamps = df[self.timestamp_col].to_numpy()
        sort_idx = np.argsort(timestamps, kind="stable")
        sorted_timestamps = timestamps[sort_idx]
        sorted_entities = df[self.entity_col].to_numpy()[sort_idx]
        sorted_amounts = df[self.amount_col].to_numpy()[sort_idx]
        sorted_addrs = df[self.addr_col].to_numpy()[sort_idx]
        sorted_devices = df[self.device_col].to_numpy()[sort_idx]
        sorted_hours = df[self.hour_col].to_numpy()[sort_idx]

        state: dict[Any, dict[str, Any]] = defaultdict(_new_card_state)

        i = 0
        while i < n:
            j = i
            tie_value = sorted_timestamps[i]
            while j < n and sorted_timestamps[j] == tie_value:
                j += 1
            ts_value = int(tie_value)

            # Pass 1: compute features for every tied row from current state.
            for k in range(i, j):
                orig_pos = int(sort_idx[k])
                card = sorted_entities[k]
                if pd.isna(card):
                    continue  # leave 0 defaults
                s = state.get(card)
                if s is None or s["count"] == 0:
                    continue  # first event for this card → 0 defaults

                # amt z-score
                if s["count"] >= _MIN_SAMPLES_FOR_STD:
                    amt_mean = s["sum_amt"] / s["count"]
                    amt_std = _sample_std(s["count"], s["sum_amt"], s["sum_amt_sq"])
                    current_amt = sorted_amounts[k]
                    if not pd.isna(current_amt) and not np.isnan(amt_std):
                        results[_AMT_Z_COL][orig_pos] = float(
                            (current_amt - amt_mean) / (amt_std + self.epsilon)
                        )

                # time z-score
                if s["delta_count"] >= _MIN_DELTAS_FOR_TIME_STD and s["prev_ts"] is not None:
                    delta_mean = s["sum_delta"] / s["delta_count"]
                    delta_std = _sample_std(s["delta_count"], s["sum_delta"], s["sum_delta_sq"])
                    current_delta = ts_value - s["prev_ts"]
                    if not np.isnan(delta_std):
                        results[_TIME_Z_COL][orig_pos] = float(
                            (current_delta - delta_mean) / (delta_std + self.epsilon)
                        )

                # addr change
                current_addr = sorted_addrs[k]
                if s["addr_counter"] and not pd.isna(current_addr):
                    mode_addr, _ = s["addr_counter"].most_common(1)[0]
                    if current_addr != mode_addr:
                        results[_ADDR_CHANGE_COL][orig_pos] = 1.0

                # device change
                current_device = sorted_devices[k]
                if (
                    s["device_set"]
                    and not pd.isna(current_device)
                    and current_device not in s["device_set"]
                ):
                    results[_DEVICE_CHANGE_COL][orig_pos] = 1.0

                # hour deviation
                if s["count"] >= 1:
                    hour_mean = s["sum_hour"] / s["count"]
                    current_hour = sorted_hours[k]
                    if not pd.isna(current_hour):
                        results[_HOUR_DEV_COL][orig_pos] = float(abs(current_hour - hour_mean))

            # Pass 2: update state with each tied row's contribution.
            for k in range(i, j):
                card = sorted_entities[k]
                if pd.isna(card):
                    continue
                s = state[card]

                current_amt = sorted_amounts[k]
                if not pd.isna(current_amt):
                    s["count"] += 1
                    amt_f = float(current_amt)
                    s["sum_amt"] += amt_f
                    s["sum_amt_sq"] += amt_f * amt_f

                if s["prev_ts"] is not None:
                    delta = ts_value - s["prev_ts"]
                    s["sum_delta"] += float(delta)
                    s["sum_delta_sq"] += float(delta) * float(delta)
                    s["delta_count"] += 1
                s["prev_ts"] = ts_value

                current_addr = sorted_addrs[k]
                if not pd.isna(current_addr):
                    s["addr_counter"][current_addr] += 1

                current_device = sorted_devices[k]
                if not pd.isna(current_device):
                    s["device_set"].add(current_device)

                current_hour = sorted_hours[k]
                if not pd.isna(current_hour):
                    s["sum_hour"] += float(current_hour)

            i = j

        out = df.copy()
        for name, vals in results.items():
            out[name] = vals
        # Cast the two flag columns to int for cleaner downstream typing.
        out[_ADDR_CHANGE_COL] = out[_ADDR_CHANGE_COL].astype(int)
        out[_DEVICE_CHANGE_COL] = out[_DEVICE_CHANGE_COL].astype(int)
        return out

    def get_feature_names(self) -> list[str]:
        """Return the deterministic list of generated column names."""
        return [
            _AMT_Z_COL,
            _TIME_Z_COL,
            _ADDR_CHANGE_COL,
            _DEVICE_CHANGE_COL,
            _HOUR_DEV_COL,
        ]

    def get_business_rationale(self) -> str:
        """Return the manifest-rendered business rationale."""
        return (
            "Per-card1 behavioural deviations: amount z-score, "
            "inter-arrival z-score, addr change flag, device change "
            "flag, hour deviation. Captures 'is this transaction "
            "unusual for THIS specific cardholder?' — complementing "
            "Tier 2's 'is the entity hot?'. All strictly past-only; "
            "first-event rows fall back to 0 (not anomalous)."
        )


# ---------------------------------------------------------------------
# `ColdStartHandler`.
# ---------------------------------------------------------------------


class ColdStartHandler(BaseFeatureGenerator):
    """Emit `is_coldstart_{entity}` flags for entities with thin history.

    For each configured entity column, a row is "cold-start" if the
    entity's count of strictly-prior events is below `min_history`.
    The flag lets downstream models distinguish rows where `Tier-3`
    behavioural features are statistically meaningful vs rows where
    the per-card / per-entity history is too thin to trust.

    Business rationale:
        Behavioural features default to 0 ("no anomaly") on first-event
        rows, but a tree model can't tell from the value alone whether
        0 means "I have no history" or "this is squarely on the mean".
        The cold-start flag carries that distinction explicitly, so
        Sprint 3's LightGBM can learn to discount behavioural features
        when the flag is 1.

    Trade-offs considered:
        - **Strict `<` past-count.** A row's flag uses the count of
          events strictly before its timestamp. Tied rows for the
          same entity see the same past count.
        - **NaN entity → flag = 1.** Can't track NaN as a discrete
          entity, so any row whose entity is NaN is by definition
          cold-start.
        - **Output dtype is `int`** (not `Int8`). The flag is fully
          determined per row; no NaN propagation needed.
    """

    def __init__(
        self,
        entity_cols: Sequence[str] | None = None,
        min_history: int | None = None,
        timestamp_col: str | None = None,
        config_path: Path | None = None,
    ) -> None:
        """Construct the cold-start handler.

        Each kwarg falls back to the YAML default when `None`.
        """
        if entity_cols is None or min_history is None or timestamp_col is None:
            cfg = _load_yaml(config_path or _resolve_config_path(_COLDSTART_CONFIG_FILENAME))
        else:
            cfg = {}

        self.entity_cols: tuple[str, ...] = tuple(
            entity_cols
            if entity_cols is not None
            else cfg.get("entity_cols", [_DEFAULT_ENTITY_COL])
        )
        self.min_history: int = (
            int(min_history)
            if min_history is not None
            else int(cfg.get("min_history", _DEFAULT_MIN_HISTORY))
        )
        self.timestamp_col: str = (
            timestamp_col
            if timestamp_col is not None
            else cfg.get("timestamp_col", _DEFAULT_TIMESTAMP_COL)
        )

    def fit(self, _df: pd.DataFrame) -> Self:
        """Stateless: returns self."""
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Emit `is_coldstart_{entity}` flags for every configured entity.

        Tied-group two-pass: pass 1 reads pre-tied counts, pass 2 updates
        counts. Strict-`<` semantics fall out naturally.

        Raises:
            KeyError: If any required column is missing from `df`.
        """
        required = {self.timestamp_col, *self.entity_cols}
        missing = sorted(required - set(df.columns))
        if missing:
            raise KeyError(f"ColdStartHandler.transform: missing required column(s) {missing}")

        n = len(df)
        feature_names = self.get_feature_names()
        results: dict[str, list[int]] = {name: [0] * n for name in feature_names}

        timestamps = df[self.timestamp_col].to_numpy()
        sort_idx = np.argsort(timestamps, kind="stable")
        sorted_timestamps = timestamps[sort_idx]

        # state[(entity_col, entity_value)] = past_count
        state: dict[tuple[str, Any], int] = defaultdict(int)

        sorted_entity_arrays: dict[str, np.ndarray[Any, Any]] = {
            entity: df[entity].to_numpy()[sort_idx] for entity in self.entity_cols
        }

        i = 0
        while i < n:
            j = i
            tie_value = sorted_timestamps[i]
            while j < n and sorted_timestamps[j] == tie_value:
                j += 1

            # Pass 1: compute flags from pre-tied counts.
            for k in range(i, j):
                orig_pos = int(sort_idx[k])
                for entity in self.entity_cols:
                    entity_val = sorted_entity_arrays[entity][k]
                    col_name = f"is_coldstart_{entity}"
                    if pd.isna(entity_val):
                        results[col_name][orig_pos] = 1
                        continue
                    past_count = state[(entity, entity_val)]
                    if past_count < self.min_history:
                        results[col_name][orig_pos] = 1

            # Pass 2: increment counts for non-NaN entity values.
            for k in range(i, j):
                for entity in self.entity_cols:
                    entity_val = sorted_entity_arrays[entity][k]
                    if pd.isna(entity_val):
                        continue
                    state[(entity, entity_val)] += 1

            i = j

        out = df.copy()
        for name, vals in results.items():
            out[name] = vals
        return out

    def get_feature_names(self) -> list[str]:
        """Return the deterministic list of generated column names."""
        return [f"is_coldstart_{entity}" for entity in self.entity_cols]

    def get_business_rationale(self) -> str:
        """Return the manifest-rendered business rationale."""
        return (
            "Cold-start indicator: per-entity flag set to 1 when fewer "
            "than `min_history` strictly-prior events exist for the "
            "entity. Lets downstream models distinguish 'no history' "
            "from 'history says this is normal'; pairs naturally with "
            "BehavioralDeviation's first-event 0-fallbacks."
        )


__all__ = ["BehavioralDeviation", "ColdStartHandler"]
