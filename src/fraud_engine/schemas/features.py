"""Pandera schemas for the post-Tier-1 and post-Tier-2 feature pipeline outputs.

`TierOneFeaturesSchema` is the outbound contract of
`scripts/build_features_tier1.py` (T1 generators only).
`TierTwoFeaturesSchema` extends it with the 20 deterministic Tier-2
columns produced by `VelocityCounter`, `HistoricalStats`, and
`TargetEncoder`; it is the outbound contract of
`scripts/build_features_tier1_2.py` and validates every frame written
to `data/processed/tier2_{train,val,test}.parquet`. Every Sprint 3+
model and Sprint 4 evaluator reads from those parquets, so catching
feature-pipeline drift at this boundary protects every downstream
stage.

Business rationale:
    Sprint 3's LightGBM tuning, Sprint 4's economic-cost evaluation,
    and Sprint 5's serving layer all read the processed parquets and
    expect a stable column contract. A failed schema validation here
    is a much cheaper failure than a model trained on silently-drifted
    features producing nonsense AUCs three sprints later.

Trade-offs considered:
    - Built via `InterimTransactionSchema.add_columns({...})` — same
      composition primitive `interim.py` uses to extend `MergedSchema`.
      Keeps the Tier-1-specific declarations in one place; raw + interim
      columns inherited automatically.
    - **`is_null_*` columns are NOT enumerated.** The
      `MissingIndicatorGenerator` learns its target set from the
      training data's missingness profile (~250 columns on real
      IEEE-CIS data; smaller on synthetic test fixtures). Pandera's
      inherited `strict=False` from `MergedSchema` lets these
      data-dependent columns pass through validation without
      explicit declaration.
    - **`day_of_week` and `is_weekend` are not re-declared** here —
      they're already in `InterimTransactionSchema`. The
      `TimeFeatureGenerator` overwrites them with bit-identical
      values; the inherited schema check still passes.
    - **`provider` / `tld` columns use `object` dtype** (not pandas
      string) because `EmailDomainExtractor` returns an `object`-dtype
      column for compatibility with the cleaner's email-column round-
      trip pattern. Same convention `MergedSchema` uses for
      `P_emaildomain` / `R_emaildomain`.
    - **`is_free` / `is_disposable` use `Int8` (nullable Int8)** so
      null source rows preserve `<NA>` semantics. Tree models handle
      nullable ints; defaulting to `0` would collapse "unknown" into
      "false".

Version history:
    1 — initial. `InterimTransactionSchema` v1 + 14 deterministic
        Tier-1 columns; `is_null_*` columns from
        `MissingIndicatorGenerator` pass through via inherited
        `strict=False`. Tier-2 schema (added in 2.2.e) extends Tier-1
        with 20 more columns; `FEATURE_SCHEMA_VERSION` stays at 1
        because Tier-2 is an additive extension and the manifest
        JSON shape (`_FEATURE_MANIFEST_SCHEMA_VERSION` in
        `pipeline.py`) hasn't changed. The 2.1.d
        `test_manifest_schema_version_matches` test continues to
        pass under this convention.
"""

from __future__ import annotations

from typing import Final

from pandera import Check, Column, DataFrameSchema

from fraud_engine.schemas.interim import InterimTransactionSchema

FEATURE_SCHEMA_VERSION: Final[int] = 1

# `log_amount = log1p(TransactionAmt)`. Lower bound 0 because the
# cleaner enforces `TransactionAmt > 0`; log1p of any positive value
# is non-negative.
_LOG_AMOUNT_MIN: Final[float] = 0.0

# `amount_decile`: target qcut bucket index. Range 0..(n_deciles-1)
# where `n_deciles` defaults to 10. With ties + `duplicates="drop"`
# the actual upper bound may be lower, but never exceeds 9.
_AMOUNT_DECILE_MIN: Final[int] = 0
_AMOUNT_DECILE_MAX: Final[int] = 9

# Pandas `.dt.hour` returns 0..23.
_HOUR_MIN: Final[int] = 0
_HOUR_MAX: Final[int] = 23

# `is_business_hours` and Int-typed flag features are 0/1.
_BINARY_VALUES: Final[list[int]] = [0, 1]

# Cyclical sin / cos must lie in [-1, 1] by construction. Pinned so
# a future-different cyclical encoding (e.g. `sin(2π · day / 7)`)
# can't silently drift the bounds.
_SIN_COS_MIN: Final[float] = -1.0
_SIN_COS_MAX: Final[float] = 1.0

# ---------------- Tier-2 column-range constants ---------------- #
#
# `VelocityCounter`: per-entity, strictly-past transaction counts in a
# fixed lookback window. Non-negative integers; NaN entity → 0.
_VELOCITY_MIN: Final[int] = 0

# `TargetEncoder`: smoothed `(sum + α × global_rate) / (count + α)`
# evaluated on `isFraud ∈ {0, 1}`. Encoded values land in [0, 1] up
# to floating-point noise. The Pandera bound includes a tiny slop so
# numerical drift at the float-precision boundary doesn't trip
# validation.
_TARGET_ENC_MIN: Final[float] = -1e-6
_TARGET_ENC_MAX: Final[float] = 1.0 + 1e-6

# `VelocityCounter` defaults: 4 entities × 3 windows = 12 columns.
_VELOCITY_ENTITIES: Final[tuple[str, ...]] = (
    "card1",
    "addr1",
    "DeviceInfo",
    "P_emaildomain",
)
_VELOCITY_WINDOWS: Final[tuple[str, ...]] = ("1h", "24h", "7d")

# `TargetEncoder` defaults: 3 categorical columns.
_TARGET_ENC_COLUMNS: Final[tuple[str, ...]] = ("card4", "addr1", "P_emaildomain")


TierOneFeaturesSchema: Final[DataFrameSchema] = InterimTransactionSchema.add_columns(
    {
        # ---------------- AmountTransformer ---------------- #
        "log_amount": Column(
            float,
            Check.greater_than_or_equal_to(_LOG_AMOUNT_MIN),
            nullable=False,
            required=True,
        ),
        "amount_decile": Column(
            int,
            Check.in_range(
                _AMOUNT_DECILE_MIN,
                _AMOUNT_DECILE_MAX,
                include_min=True,
                include_max=True,
            ),
            nullable=False,
            required=True,
        ),
        # ---------------- TimeFeatureGenerator ---------------- #
        # `day_of_week` and `is_weekend` are inherited from
        # `InterimTransactionSchema`; the generator overwrites them
        # with bit-identical values so the inherited check passes.
        "hour_of_day": Column(
            int,
            Check.in_range(_HOUR_MIN, _HOUR_MAX, include_min=True, include_max=True),
            nullable=False,
            required=True,
        ),
        "is_business_hours": Column(
            int,
            Check.isin(_BINARY_VALUES),
            nullable=False,
            required=True,
        ),
        "hour_sin": Column(
            float,
            Check.in_range(_SIN_COS_MIN, _SIN_COS_MAX, include_min=True, include_max=True),
            nullable=False,
            required=True,
        ),
        "hour_cos": Column(
            float,
            Check.in_range(_SIN_COS_MIN, _SIN_COS_MAX, include_min=True, include_max=True),
            nullable=False,
            required=True,
        ),
        # ---------------- EmailDomainExtractor ---------------- #
        # All four columns per email source are nullable because the
        # cleaner allows null `*_emaildomain` (R_emaildomain coverage
        # is ~24% on IEEE-CIS train).
        "P_emaildomain_provider": Column(object, nullable=True, required=True),
        "P_emaildomain_tld": Column(object, nullable=True, required=True),
        "P_emaildomain_is_free": Column("Int8", nullable=True, required=True),
        "P_emaildomain_is_disposable": Column("Int8", nullable=True, required=True),
        "R_emaildomain_provider": Column(object, nullable=True, required=True),
        "R_emaildomain_tld": Column(object, nullable=True, required=True),
        "R_emaildomain_is_free": Column("Int8", nullable=True, required=True),
        "R_emaildomain_is_disposable": Column("Int8", nullable=True, required=True),
    }
)


TierTwoFeaturesSchema: Final[DataFrameSchema] = TierOneFeaturesSchema.add_columns(
    {
        # ---------------- VelocityCounter ---------------- #
        # 4 entities × 3 windows = 12 columns. All strictly non-negative
        # integers; NaN entity → 0 so nullable=False holds.
        **{
            f"{entity}_velocity_{window}": Column(
                int,
                Check.greater_than_or_equal_to(_VELOCITY_MIN),
                nullable=False,
                required=True,
            )
            for entity in _VELOCITY_ENTITIES
            for window in _VELOCITY_WINDOWS
        },
        # ---------------- HistoricalStats ---------------- #
        # 3 stats for card1 + 2 stats for addr1 = 5 float columns.
        # nullable=True because:
        #   - first event for an entity → empty deque → all NaN
        #   - n=1 deque → std=NaN (sample std needs ≥ 2)
        "card1_amt_mean_30d": Column(float, nullable=True, required=True),
        "card1_amt_std_30d": Column(float, nullable=True, required=True),
        "card1_amt_max_30d": Column(float, nullable=True, required=True),
        "addr1_amt_mean_30d": Column(float, nullable=True, required=True),
        "addr1_amt_std_30d": Column(float, nullable=True, required=True),
        # ---------------- TargetEncoder ---------------- #
        # 3 columns. Floats in approximately [0, 1] (smoothed fraud
        # rates). nullable=False — `_lookup` always returns a float
        # (mapping value or fallback `global_rate`).
        **{
            f"{cat}_target_enc": Column(
                float,
                Check.in_range(
                    _TARGET_ENC_MIN,
                    _TARGET_ENC_MAX,
                    include_min=True,
                    include_max=True,
                ),
                nullable=False,
                required=True,
            )
            for cat in _TARGET_ENC_COLUMNS
        },
    }
)


__all__ = [
    "FEATURE_SCHEMA_VERSION",
    "TierOneFeaturesSchema",
    "TierTwoFeaturesSchema",
]
