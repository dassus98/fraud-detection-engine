"""Tier-1 basic feature generators (Amount + Time).

This module is the first concrete `BaseFeatureGenerator` subclass set
in Sprint 2. Tier-1 features are the cheapest, highest-prior-art
signals — they appear in every fraud paper as headline predictors and
need no per-entity state. Two generators land here:

- `AmountTransformer` — `log_amount` (heavy-tailed value compression)
  + `amount_decile` (tree-friendly discretisation at regime boundaries
  the EDA flagged in Section B.2).
- `TimeFeatureGenerator` — `hour_of_day`, `day_of_week`, `is_weekend`,
  `is_business_hours`, `hour_sin`, `hour_cos`. Sin/cos cyclical
  encoding lets tree models split "near midnight" without the 23 → 0
  wrap-around discontinuity.

Sprint 2's later prompts will add categorical encoders + identity
features to this same `tier1_basic.py`.

Business rationale:
    Tier-1 monotone-amount and diurnal-cycle signals were the strongest
    single-feature predictors in the EDA's Section B fraud-rate plots.
    Computing them once at the cleanest possible boundary — directly
    after the cleaner — means every Sprint 3 model and Sprint 4
    threshold experiment reads from the same feature matrix; no
    later sprint reimplements `log1p`.

Trade-offs considered:
    - **`pd.qcut` over `pd.cut` for `amount_decile`.** qcut adapts the
      bin edges to the actual amount distribution (so the Sprint 4
      cost-curve sees roughly 10% of fraud value in each bucket), at
      the cost of being non-deterministic at ties.
      `duplicates="drop"` makes ties safe by collapsing them; the
      resulting bucket count may be < 10 but every bucket is non-empty,
      which is the desired property.
    - **`is_business_hours = 9 ≤ hour < 17` UTC.** IEEE-CIS spans
      timezones and we don't have per-customer tz metadata, so a
      single UTC convention is the only honest choice. The feature
      captures "global daytime trough" — the fraud-rate by hour plot
      in Section B.4 shows a clear UTC daytime dip, validating the
      convention.
    - **Sin / cos cyclical encoding** rather than a single circular
      index for `hour`. Tree models (LightGBM in Sprint 3) cannot
      natively handle the 23 → 0 wrap-around; sin / cos place 23:00
      and 00:00 next to each other on the unit circle, so a single
      tree split cleanly separates "late-night" from "early-morning"
      without an explicit boundary feature.
    - **`day_of_week` and `is_weekend` overlap with the cleaner's
      output column names.** The generator deliberately re-derives
      these from `timestamp` rather than aliasing the cleaner's
      columns so it works standalone — Sprint 5's serving layer may
      ingest a frame that did not pass through the cleaner. Same
      values, harmless overwrite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final, Self

import numpy as np
import pandas as pd
import yaml

from fraud_engine.features.base import BaseFeatureGenerator

# Default input column names. Constants so a future schema rename
# (e.g. `transaction_amount`) updates exactly one place per generator.
_TRANSACTION_AMT_COLUMN: Final[str] = "TransactionAmt"
_TIMESTAMP_COLUMN: Final[str] = "timestamp"

# Default qcut bucket count. 10 is the convention for "decile";
# subclasses or callers can override via constructor.
_DEFAULT_N_DECILES: Final[int] = 10

# UTC business-hours bounds. Closed on the left (09:00 inside),
# open on the right (17:00 outside). See module trade-off note for
# the global-UTC justification.
_BUSINESS_HOURS_START_UTC: Final[int] = 9
_BUSINESS_HOURS_END_UTC: Final[int] = 17

# Hours-per-day denominator for the cyclical sin/cos mapping; pinned
# constant so the radian factor is auditable in one line.
_HOURS_IN_DAY: Final[int] = 24

# Pandas `.dt.dayofweek` is Monday=0..Sunday=6. Saturday is 5; the
# weekend flag fires on dow >= this threshold.
_WEEKEND_FIRST_DOW: Final[int] = 5

# Default input columns for `EmailDomainExtractor`. Cleaner output
# guarantees these are lowercase + stripped; the YAML lookup is
# therefore exact-match without case folding.
_DEFAULT_EMAIL_COLUMNS: Final[tuple[str, ...]] = ("P_emaildomain", "R_emaildomain")

# Default config filename. The full path is resolved at load time
# against the repo root (computed via `__file__`), so the generators
# work regardless of the cwd from which Python was launched.
_DEFAULT_CONFIG_FILENAME: Final[str] = "email_providers.yaml"

# Separator for `rsplit`-based domain decomposition. Module-level
# so a future "split on @" change updates one place.
_TLD_SEPARATOR: Final[str] = "."


class AmountTransformer(BaseFeatureGenerator):
    """Adds `log_amount` and `amount_decile` from a positive amount column.

    Business rationale:
        `TransactionAmt`'s distribution has a heavy right tail
        (Section B.8 of the EDA showed legit transactions skew small,
        fraud skews larger). `log_amount = log1p(TransactionAmt)`
        compresses that tail so a linear model sees a near-Gaussian
        feature, while `amount_decile` discretises the same signal
        into buckets a tree model can split on directly. Sprint 4's
        cost-curve analysis reads `amount_decile` to compute
        per-bucket expected losses.

    Trade-offs considered:
        - `pd.qcut` over `pd.cut` for `amount_decile`. qcut adapts to
          the data; cut needs hand-picked breakpoints that drift as
          the dataset evolves. `duplicates="drop"` handles tied
          values safely.
        - Negative-amount rejection in BOTH `fit` and `transform`,
          not just `fit`. The cleaner already filters `> 0` at
          ingest, but Sprint 5's serving layer may receive a fresh
          payload that has not been cleaner-validated; a defensive
          re-check at transform time is cheap and catches that case.

    Attributes:
        amount_col: Name of the amount column. Default `TransactionAmt`.
        n_deciles: Target number of qcut buckets. Default 10.
        decile_edges: Learned bin edges from `fit`; `None` pre-fit,
            `list[float]` post-fit. The list length is
            `actual_buckets + 1`; `actual_buckets` may be `< n_deciles`
            if the input had ties.
    """

    def __init__(
        self,
        amount_col: str = _TRANSACTION_AMT_COLUMN,
        n_deciles: int = _DEFAULT_N_DECILES,
    ) -> None:
        """Construct the transformer.

        Args:
            amount_col: Column name to read amounts from.
            n_deciles: Target number of qcut buckets; actual bucket
                count may be lower under heavy ties.
        """
        self.amount_col: str = amount_col
        self.n_deciles: int = n_deciles
        self.decile_edges: list[float] | None = None

    def fit(self, df: pd.DataFrame) -> Self:
        """Learn qcut bin edges from `df[self.amount_col]`.

        Args:
            df: Frame with a non-negative `amount_col`.

        Returns:
            self.

        Raises:
            ValueError: If any amount is negative.
        """
        amounts = df[self.amount_col]
        self._reject_negative(amounts, where="fit")
        # `duplicates="drop"` collapses tied edges (frequent in
        # production data where many transactions share rounded
        # amounts). Resulting bin count may be < n_deciles; every
        # bucket is still non-empty, which is what we want.
        _, edges = pd.qcut(amounts, q=self.n_deciles, retbins=True, duplicates="drop")
        self.decile_edges = [float(e) for e in edges]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply log + decile to `df[self.amount_col]`.

        Args:
            df: Frame to transform; must satisfy the same
                non-negativity constraint as `fit`.

        Returns:
            New DataFrame with every input column plus `log_amount`
            and `amount_decile`.

        Raises:
            AttributeError: If `fit` has not been called.
            ValueError: If any amount is negative.
        """
        if self.decile_edges is None:
            raise AttributeError("AmountTransformer must be fit before transform")
        amounts = df[self.amount_col]
        self._reject_negative(amounts, where="transform")
        out = df.copy()
        out["log_amount"] = np.log1p(amounts.to_numpy()).astype(np.float64)
        # `pd.cut` returns NaN for values outside the fit-time edge
        # range. Clip to the last bucket so downstream models always
        # see a valid integer index.
        deciles = pd.cut(
            amounts,
            bins=self.decile_edges,
            labels=False,
            include_lowest=True,
        )
        n_buckets = len(self.decile_edges) - 1
        out["amount_decile"] = deciles.fillna(n_buckets - 1).astype(int).clip(0, n_buckets - 1)
        return out

    def get_feature_names(self) -> list[str]:
        """Return the two columns this generator adds."""
        return ["log_amount", "amount_decile"]

    def get_business_rationale(self) -> str:
        """One-paragraph rationale for Sprint 5 manifest rendering."""
        return (
            "Monotone log-amount captures the heavy right tail of "
            "transaction values; amount_decile discretises the same "
            "signal so tree models can split at the regime boundaries "
            "the EDA flagged (mid-to-large buckets carry the most fraud). "
            "Sprint 4's cost-curve analysis keys off `amount_decile`."
        )

    @staticmethod
    def _reject_negative(amounts: pd.Series[float], *, where: str) -> None:
        """Raise `ValueError` if any amount is negative.

        Args:
            amounts: Series to validate.
            where: ``"fit"`` or ``"transform"`` — included in the
                error message so traceback points at the right call.
        """
        if (amounts < 0).any():
            n_neg = int((amounts < 0).sum())
            raise ValueError(
                f"AmountTransformer rejected {n_neg} negative amount(s) "
                f"in {where}; transaction amounts must be non-negative. "
                f"The cleaner's `TransactionAmt > 0` filter should have "
                f"caught these — check the input frame's lineage."
            )


class TimeFeatureGenerator(BaseFeatureGenerator):
    """Adds calendar features derived from a tz-aware `timestamp` column.

    Business rationale:
        Diurnal and weekday patterns are the strongest non-monetary
        signal in the EDA — Section B.4 (fraud rate by hour) and
        Section B.6 (dow × hour heatmap) both showed clear hot spots
        in late-night UTC weekday hours. A model that ignores time
        leaves obvious signal on the table.

    Trade-offs considered:
        - **Stateless `fit`.** Every output is a deterministic
          function of `timestamp` alone — no fitted state, no
          training-data dependency. The generator works correctly
          on any frame with a `timestamp` column without prior
          `fit_transform`.
        - **Re-derive `day_of_week` / `is_weekend` from `timestamp`**
          rather than aliasing the cleaner's existing columns. Lets
          this generator slot into a Sprint 5 serving pipeline that
          does not include the cleaner stage. Re-derivation is
          O(n) and uses the same logic, so values are bit-identical
          to the cleaner's output.
        - **Sin / cos pair for cyclical hour** rather than just a
          single circular index. Tree splits cannot natively handle
          the 23 → 0 wrap-around; sin / cos place adjacent hours at
          adjacent points on the unit circle, so a single split
          cleanly separates "late-night" from "early-morning".
        - **No tz-naive guard.** `ts.dt.hour` works on tz-naive
          input but yields "local" hour rather than UTC. The
          cleaner's contract is tz-aware UTC; if a future caller
          breaks that contract upstream, `is_business_hours`
          semantics shift silently. Documented; not enforced.

    Attributes:
        timestamp_col: Name of the input column. Default
            `timestamp`. Must be a tz-aware datetime64; the cleaner
            produces this dtype.
    """

    def __init__(self, timestamp_col: str = _TIMESTAMP_COLUMN) -> None:
        """Construct the generator.

        Args:
            timestamp_col: Column name to read timestamps from.
        """
        self.timestamp_col: str = timestamp_col

    def fit(self, _df: pd.DataFrame) -> Self:
        """No-op. The generator is stateless.

        Args:
            _df: Unused; present only to satisfy the
                `BaseFeatureGenerator.fit` signature.

        Returns:
            self.
        """
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive the six calendar features from `df[self.timestamp_col]`.

        Args:
            df: Frame with a tz-aware `timestamp` column.

        Returns:
            New DataFrame with every input column plus the six
            calendar features named in `get_feature_names`.
        """
        ts = df[self.timestamp_col]
        out = df.copy()
        out["hour_of_day"] = ts.dt.hour.astype(int)
        out["day_of_week"] = ts.dt.dayofweek.astype(int)
        out["is_weekend"] = (out["day_of_week"] >= _WEEKEND_FIRST_DOW).astype(int)
        out["is_business_hours"] = (
            (out["hour_of_day"] >= _BUSINESS_HOURS_START_UTC)
            & (out["hour_of_day"] < _BUSINESS_HOURS_END_UTC)
        ).astype(int)
        # Cyclical encoding: maps 0..23 onto the unit circle so a
        # tree split can put 23:00 and 00:00 in the same leaf.
        radians = 2.0 * np.pi * out["hour_of_day"].to_numpy() / _HOURS_IN_DAY
        out["hour_sin"] = np.sin(radians)
        out["hour_cos"] = np.cos(radians)
        return out

    def get_feature_names(self) -> list[str]:
        """Return the six columns this generator adds."""
        return [
            "hour_of_day",
            "day_of_week",
            "is_weekend",
            "is_business_hours",
            "hour_sin",
            "hour_cos",
        ]

    def get_business_rationale(self) -> str:
        """One-paragraph rationale for Sprint 5 manifest rendering."""
        return (
            "Diurnal and weekday signals surface as fraud-rate hot-spots "
            "in the EDA's Section B.4 by-hour plot and Section B.6 "
            "dow × hour heatmap. Cyclical (sin / cos) encoding lets "
            "tree splits handle the 23 → 0 wrap-around; "
            "`is_business_hours` tags the daytime trough where fraud "
            "rates sit below the overall 3.5%."
        )


def _resolve_default_config_path() -> Path:
    """Locate `configs/email_providers.yaml` relative to the repo root.

    Mirrors `Settings._PROJECT_ROOT`'s discovery: this file lives at
    `src/fraud_engine/features/tier1_basic.py`, so the repo root is
    three parents up.

    Returns:
        Absolute path to the default email-providers YAML.
    """
    project_root = Path(__file__).resolve().parents[3]
    return project_root / "configs" / _DEFAULT_CONFIG_FILENAME


def _load_yaml(path: Path) -> dict[str, Any]:
    """Read and parse a YAML config file.

    Args:
        path: Absolute path to a YAML file.

    Returns:
        The parsed top-level mapping. Raises `yaml.YAMLError` on
        malformed input; `FileNotFoundError` on missing path.
    """
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected top-level mapping in {path}, got {type(loaded).__name__}")
    return loaded


class EmailDomainExtractor(BaseFeatureGenerator):
    """Splits each email column into provider / TLD + free / disposable flags.

    Business rationale:
        EDA Section B.7 surfaced certain free email providers as
        carrying 10–30× the baseline fraud rate. A free / disposable
        flag captures that signal at single-feature granularity for
        every Sprint 3 model; the provider / TLD split lets target-
        encoding-style features in later prompts key off the
        normalised domain rather than the raw string.

    Trade-offs considered:
        - **Stateless `fit`.** The provider / disposable lists are
          loaded at `__init__` from YAML; `fit` is a no-op. Tests
          can bypass YAML entirely by passing explicit
          `free_providers` / `disposable_providers` sets.
        - **`rsplit(".", 1)`** (right-split, max 1 split) for the
          domain decomposition. "gmail.com" → ("gmail", "com"); the
          TLD is the last segment. Multi-segment TLDs like "co.uk"
          split as ("company.co", "uk") — pragmatic for IEEE-CIS
          where multi-segment TLDs are rare.
        - **`pd.Int8Dtype` (nullable Int8)** for the flag columns so
          a null email row produces `<NA>` (not `0`) in the flag.
          Tree models handle nullable ints; defaulting to 0 would
          collapse the "unknown" semantics into "false".
        - **YAML lists are loaded eagerly at `__init__`** to fail
          fast on a missing config rather than at first transform.

    Attributes:
        email_columns: Tuple of column names to process. Default is
            `("P_emaildomain", "R_emaildomain")`.
        free_providers: Set of free-email-provider domains.
        disposable_providers: Set of disposable-provider domains.
    """

    def __init__(
        self,
        email_columns: tuple[str, ...] = _DEFAULT_EMAIL_COLUMNS,
        free_providers: frozenset[str] | None = None,
        disposable_providers: frozenset[str] | None = None,
        config_path: Path | None = None,
    ) -> None:
        """Construct the extractor.

        Args:
            email_columns: Columns to derive features from.
            free_providers: Optional explicit free-provider set; if
                None, loaded from `config_path` (or the default).
            disposable_providers: Optional explicit disposable-provider
                set; same fallback rules as `free_providers`.
            config_path: Override for the YAML path; defaults to
                `<repo>/configs/email_providers.yaml`.
        """
        self.email_columns: tuple[str, ...] = email_columns
        if free_providers is not None and disposable_providers is not None:
            self.free_providers: frozenset[str] = free_providers
            self.disposable_providers: frozenset[str] = disposable_providers
        else:
            cfg = _load_yaml(config_path or _resolve_default_config_path())
            self.free_providers = (
                free_providers if free_providers is not None else frozenset(cfg["free_providers"])
            )
            self.disposable_providers = (
                disposable_providers
                if disposable_providers is not None
                else frozenset(cfg["disposable_providers"])
            )

    def fit(self, _df: pd.DataFrame) -> Self:
        """No-op. Provider lists are loaded at `__init__`.

        Returns:
            self.
        """
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive 4 columns per email column: provider, TLD, free flag, disposable flag.

        Nulls in the input email column produce `<NA>` in all four
        derived columns (pandas nullable string + Int8 dtypes).

        Args:
            df: Frame with the email columns named in `self.email_columns`.

        Returns:
            New DataFrame with 4 × len(email_columns) added columns.
        """
        out = df.copy()
        for col in self.email_columns:
            provider, tld = self._split_domain(df[col])
            out[f"{col}_provider"] = provider
            out[f"{col}_tld"] = tld
            # `.isin` on nullable Series returns bool; mask out rows
            # whose original email is null so the flag is `<NA>`,
            # not `0` — preserves "unknown" semantics for tree models.
            null_mask = df[col].isna()
            is_free = provider.isin(
                {p.rsplit(_TLD_SEPARATOR, 1)[0] for p in self.free_providers}
            ) | df[col].isin(self.free_providers)
            is_disposable = provider.isin(
                {p.rsplit(_TLD_SEPARATOR, 1)[0] for p in self.disposable_providers}
            ) | df[col].isin(self.disposable_providers)
            out[f"{col}_is_free"] = is_free.astype("Int8").mask(null_mask, pd.NA)
            out[f"{col}_is_disposable"] = is_disposable.astype("Int8").mask(null_mask, pd.NA)
        return out

    def get_feature_names(self) -> list[str]:
        """Return 4 × len(email_columns) column names."""
        return [
            f"{col}_{suffix}"
            for col in self.email_columns
            for suffix in ("provider", "tld", "is_free", "is_disposable")
        ]

    def get_business_rationale(self) -> str:
        """One-paragraph rationale for Sprint 5 manifest rendering."""
        return (
            "Email-domain features encode the EDA's Section B.7 finding "
            "that certain free / disposable providers carry 10–30× the "
            "baseline fraud rate. The provider / TLD split feeds Sprint "
            "2's later target-encoding generators; the free / disposable "
            "flags give a single-bit signal Sprint 3's baseline can use "
            "directly."
        )

    @staticmethod
    def _split_domain(series: pd.Series[Any]) -> tuple[pd.Series[Any], pd.Series[Any]]:
        """Vectorised null-safe `rsplit(".", 1)` on a string-like Series.

        Args:
            series: Email-domain column. May contain NaN; categorical
                dtype is round-tripped through string for `.str.rsplit`.

        Returns:
            Two Series: (provider, tld). Null source rows propagate
            as `<NA>` in BOTH columns; a non-null domain with no dot
            returns (domain, "").
        """
        # Coerce to nullable string so `.str.rsplit` works on category /
        # object inputs uniformly.
        as_str = series.astype("string")
        split = as_str.str.rsplit(_TLD_SEPARATOR, n=1, expand=True)
        if split.shape[1] == 1:
            # No dot anywhere in the column — pad to 2 columns so
            # downstream indexing always works. `pd.NA` (not "") so
            # null-source rows stay null.
            split[1] = pd.NA
        provider = split[0]
        tld = split[1]
        # Distinguish "null source" from "non-null source with no
        # dot": the former should remain `<NA>` in tld, the latter
        # should become `""`. `mask(cond, value)` replaces where
        # cond is True.
        null_source = as_str.isna()
        no_dot_with_value = ~null_source & tld.isna()
        tld = tld.mask(no_dot_with_value, "")
        return provider, tld


class MissingIndicatorGenerator(BaseFeatureGenerator):
    """Adds `is_null_{col}` indicators for columns that exceeded the threshold at fit time.

    Business rationale:
        EDA Section C.4's predictive-missingness analysis showed
        certain columns (D7 most strongly) carry a 5×+ fraud-rate
        lift when present vs null. An explicit `is_null_*` feature
        lets the Sprint 3 model exploit that signal directly,
        without requiring the model to re-discover it from the
        underlying column's NaN pattern.

    Trade-offs considered:
        - **Learns columns at fit; emits the same set at transform**
          regardless of whether the transform frame has any nulls in
          those columns. Val / test see the same feature surface as
          train; the model never sees a missing feature column.
        - **`target_columns` sorted alphabetically** for
          deterministic feature-manifest ordering. `df.isna().mean()`
          returns column-order from the input frame, which is not
          stable across concatenated frames.
        - **Schema drift at transform** — if a target column is
          absent from the transform frame, emit `is_null_{col} = 1`
          (the column is "missing" for every row). Stricter
          alternative would raise; the lenient form supports Sprint
          5 serving where input shape may legitimately vary.

    Attributes:
        threshold: Missingness fraction above which a column earns
            an indicator.
        target_columns: List of column names learned at `fit`; None
            pre-fit, sorted post-fit.
    """

    def __init__(
        self,
        threshold: float | None = None,
        config_path: Path | None = None,
    ) -> None:
        """Construct the generator.

        Args:
            threshold: Optional explicit threshold; if None, loads
                `missing_indicator_threshold` from the YAML config.
            config_path: Override for the YAML path; defaults to
                `<repo>/configs/email_providers.yaml`.
        """
        if threshold is not None:
            self.threshold: float = float(threshold)
        else:
            cfg = _load_yaml(config_path or _resolve_default_config_path())
            self.threshold = float(cfg["missing_indicator_threshold"])
        self.target_columns: list[str] | None = None

    def fit(self, df: pd.DataFrame) -> Self:
        """Identify columns whose missingness exceeds `self.threshold`.

        Args:
            df: Training frame; the missingness rate is computed
                column-wise.

        Returns:
            self.
        """
        miss = df.isna().mean()
        self.target_columns = sorted(
            [str(col) for col, rate in miss.items() if rate > self.threshold]
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Emit `is_null_{col}` for every column learned at `fit`.

        Args:
            df: Frame to transform.

        Returns:
            New DataFrame with `len(self.target_columns)` added
            columns. A target column absent from `df` triggers an
            all-1s indicator (the column is "missing for every row").

        Raises:
            AttributeError: If `fit` has not been called.
        """
        if self.target_columns is None:
            raise AttributeError("MissingIndicatorGenerator must be fit before transform")
        out = df.copy()
        for col in self.target_columns:
            if col in df.columns:
                out[f"is_null_{col}"] = df[col].isna().astype(int)
            else:
                # Column absent — treat as fully null. Stricter
                # alternative (raise) would break Sprint 5 serving.
                out[f"is_null_{col}"] = 1
        return out

    def get_feature_names(self) -> list[str]:
        """Return `is_null_*` column names. Empty list pre-fit."""
        if self.target_columns is None:
            return []
        return [f"is_null_{col}" for col in self.target_columns]

    def get_business_rationale(self) -> str:
        """One-paragraph rationale for Sprint 5 manifest rendering."""
        return (
            "EDA Section C.4 showed several columns carry a 5×+ fraud-"
            "rate lift when present vs null (D7 strongest). Explicit "
            "`is_null_*` features let Sprint 3's baseline exploit that "
            "predictive-missingness signal without re-discovering it "
            "from the underlying column's NaN pattern at every split."
        )


__all__ = [
    "AmountTransformer",
    "EmailDomainExtractor",
    "MissingIndicatorGenerator",
    "TimeFeatureGenerator",
]
