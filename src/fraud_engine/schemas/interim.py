"""Pandera schema for the post-cleaning interim transaction frame.

This is the *outbound* contract of `fraud_engine.data.cleaner.TransactionCleaner`.
The cleaner adds four calendar columns (`timestamp`, `hour`,
`day_of_week`, `is_weekend`) on top of the merged raw frame and
validates the result against this schema before handing it off to
Sprint 2's feature pipeline.

Business rationale:
    Sprint 2's feature pipeline reads `timestamp`, `hour`,
    `day_of_week`, and `is_weekend` directly to build time-of-day,
    weekday-effect, and weekend-vs-weekday features. If any of those
    columns silently drift in dtype or domain (for example,
    `pandas` switching `.dt.dayofweek` to a 1..7 convention, or a
    timezone-naive timestamp slipping in), every downstream feature
    corrupts. Validating at the cleaner's exit boundary catches that
    drift loudly, with a traceable error, instead of allowing
    corrupted features to reach the model.

Trade-offs considered:
    - Built via `MergedSchema.add_columns({...})` rather than as a
      hand-written fresh `DataFrameSchema`. `add_columns` keeps the
      raw-column declarations in exactly one place: if `MergedSchema`
      grows a new required column, the interim schema picks it up
      without a parallel edit.
    - `strict=False` is inherited from `MergedSchema`. The cleaner is
      non-destructive — every raw column survives — and the wide V /
      C / D / M / id_* blocks are not enumerated explicitly. A
      `strict=True` schema would refuse those columns; `strict=False`
      validates the columns we name and lets the rest pass through.
    - `timestamp` is tz-aware UTC (`datetime64[ns, UTC]`) rather than
      naive. The anchor in `Settings.transaction_dt_anchor_iso` is
      already tz-aware (`+00:00`); making the schema tz-aware
      preserves correctness end-to-end and self-documents that every
      downstream comparison is in UTC, not "local time of whoever
      ran the pipeline."

Version history:
    1 — initial; `MergedSchema` v1 + four calendar columns derived
        from `TransactionDT`.
"""

from __future__ import annotations

from typing import Final

import pandas as pd
from pandera import Check, Column, DataFrameSchema

from fraud_engine.schemas.raw import MergedSchema

INTERIM_SCHEMA_VERSION: Final[int] = 1

# Pandas `.dt.hour` returns 0..23; closed interval mirrors that domain.
_HOUR_MIN: Final[int] = 0
_HOUR_MAX: Final[int] = 23

# Pandas `.dt.dayofweek` is Monday=0..Sunday=6. Different from
# numpy / ISO conventions; pinned here so any future pandas
# convention change fails the schema check rather than silently
# shifting the feature semantics.
_DAY_OF_WEEK_MIN: Final[int] = 0
_DAY_OF_WEEK_MAX: Final[int] = 6

# `is_weekend` is the binary projection of `day_of_week >= 5`. Stored
# as int (not bool) to match `isFraud`'s int(0/1) convention from the
# raw schema, so downstream models / features treat all binary flags
# uniformly.
_WEEKEND_VALUES: Final[list[int]] = [0, 1]

# Tz-aware UTC. See module docstring for the rationale.
_TIMESTAMP_DTYPE: Final[pd.DatetimeTZDtype] = pd.DatetimeTZDtype(tz="UTC")


InterimTransactionSchema: Final[DataFrameSchema] = MergedSchema.add_columns(
    {
        "timestamp": Column(
            _TIMESTAMP_DTYPE,
            nullable=False,
            required=True,
        ),
        "hour": Column(
            int,
            Check.in_range(_HOUR_MIN, _HOUR_MAX, include_min=True, include_max=True),
            nullable=False,
            required=True,
        ),
        "day_of_week": Column(
            int,
            Check.in_range(
                _DAY_OF_WEEK_MIN,
                _DAY_OF_WEEK_MAX,
                include_min=True,
                include_max=True,
            ),
            nullable=False,
            required=True,
        ),
        "is_weekend": Column(
            int,
            Check.isin(_WEEKEND_VALUES),
            nullable=False,
            required=True,
        ),
    }
)


__all__ = [
    "INTERIM_SCHEMA_VERSION",
    "InterimTransactionSchema",
]
