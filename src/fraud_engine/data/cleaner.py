"""Transactional cleaner: derive calendar columns + drop invalid rows.

`TransactionCleaner` is the bridge between `RawDataLoader.load_merged()`
and Sprint 2's feature pipeline. It takes the merged IEEE-CIS frame
and produces a cleaned interim frame validated against
`InterimTransactionSchema`. The output is what every later sprint
consumes — feature engineering, training, and serving all read from
the same calendar columns this module derives.

Business rationale:
    Every Sprint 2 feature that mentions time-of-day, weekday effect,
    or weekend behaviour reads `timestamp`, `hour`, `day_of_week`, or
    `is_weekend` from the cleaned frame. Deriving them here, in one
    canonical place, means Sprint 2 / 4 / 5 cannot disagree on the
    derivation. The defensive `TransactionAmt > 0` drop is a safety
    net: the raw schema already enforces this at ingest, but if a
    future relaxation lets zero-amount rows through the cleaner is
    the second line of defence. Email-domain standardisation
    stabilises the join keys for the email-domain features Sprint 2
    builds — `"Gmail.com"` and `"gmail.com"` would otherwise become
    distinct levels in the categorical encoder.

Trade-offs considered:
    - `pandas` for derivation rather than `polars` or `pyarrow.compute`:
      the rest of the pipeline (loader, splits, baseline) is pandas;
      a converter round-trip would cost more than the derivation
      itself.
    - Drop with logged reason rather than coerce-to-NaN: a transaction
      with `TransactionAmt == 0` is conceptually meaningless for the
      fraud-cost objective. Coercing to NaN would let it leak into
      modelling with imputed values; dropping is the loud signal.
    - Per-reason summary log over per-row log: row-by-row logging
      would emit O(thousands) records on real data and bloat the
      log store. The summary entry carries a row count and a sample
      of TransactionIDs — enough to investigate, not enough to spam.
      The row-count invariant
      ``rows_in - rows_out == sum(dropped_by_reason.values())`` is
      enforced in `CleanReport` and asserted by the unit tests, so
      no row is "lost" silently.
    - Email lowercase + strip leaves NaN as NaN: imputing a default
      value (``"unknown"``) is a feature-engineering decision, not a
      cleaning concern. The cleaner stays narrow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd

from fraud_engine.config.settings import Settings, get_settings
from fraud_engine.schemas.interim import (
    INTERIM_SCHEMA_VERSION,
    InterimTransactionSchema,
)
from fraud_engine.utils.logging import get_logger, log_call

# Drop-reason code emitted in `cleaner.drops` log records and used as
# the key in `CleanReport.dropped_by_reason`. Single-source-of-truth
# string constant so a typo in one place can't drift apart from another.
_NON_POSITIVE_AMOUNT: Final[str] = "non_positive_amount"

# Module-level catalogue of valid drop reasons. Currently a single
# entry; new reasons append here AND get a private string constant
# above. Kept as a tuple (immutable) so consumers can't accidentally
# mutate it.
_DROP_REASONS: Final[tuple[str, ...]] = (_NON_POSITIVE_AMOUNT,)

# Email-domain columns the cleaner standardises. Both are nullable in
# the raw schema; NaN passes through `.str.lower().str.strip()`
# unchanged. Kept as a constant so a future schema rename updates
# exactly one place.
_EMAIL_COLUMNS: Final[tuple[str, ...]] = ("P_emaildomain", "R_emaildomain")

# Column the amount-validity drop checks. Mirrors the raw schema's
# `Check.greater_than(0.0)` constraint, restated here as a defensive
# duplicate.
_AMOUNT_COLUMN: Final[str] = "TransactionAmt"

# Column the cleaner partitions calendar derivations on. Module-level
# constant so a future rename (in Sprint 2's feature pipeline, for
# example) updates exactly one place.
_TIME_COLUMN: Final[str] = "TransactionDT"

# Cap the `sample_ids` list emitted in the `cleaner.drops` log
# warning. 10 is enough to investigate a small drop count without
# bloating log records when a future drop reason fires on thousands
# of rows.
_DROP_LOG_SAMPLE_SIZE: Final[int] = 10

# Saturday=5, Sunday=6 in pandas `.dt.dayofweek`. Module-level so the
# `is_weekend` derivation is auditable in one line.
_WEEKEND_THRESHOLD: Final[int] = 5


@dataclass(frozen=True)
class CleanReport:
    """Lightweight result object describing the cleaner's output.

    Attributes:
        rows_in: Row count of the input DataFrame, before any drops.
        rows_out: Row count of the returned DataFrame, post-validation.
        rows_dropped: ``rows_in - rows_out``. Equal to
            ``sum(dropped_by_reason.values())`` by construction.
        dropped_by_reason: Map from drop-reason code (one of
            ``_DROP_REASONS``) to the number of rows dropped for that
            reason. Reasons with zero drops are omitted.
        schema_version: The `INTERIM_SCHEMA_VERSION` the output frame
            was validated against.
    """

    rows_in: int
    rows_out: int
    rows_dropped: int
    dropped_by_reason: dict[str, int]
    schema_version: int


class TransactionCleaner:
    """Clean a merged IEEE-CIS transaction frame into the interim shape.

    Business rationale:
        See module docstring. The cleaner is the boundary between
        the raw-CSV truth (loader) and feature engineering (Sprint
        2). Every downstream consumer reads from the cleaned frame.

    Trade-offs considered:
        - A class, not a free function: matches `RawDataLoader`'s
          shape — carries injected `Settings` (the anchor lives
          there), holds `last_report` for inspection, gives Sprint
          2's pipeline driver a stable API surface.
        - The cleaner does *not* cache. Re-running on the same
          frame is millisecond-scale; caching would only mask
          non-determinism bugs that we want to surface.
        - `last_report` is a public instance attribute rather than
          a return value alongside the frame. Returning a tuple
          would force every caller (today and in Sprint 2) to
          unpack; the attribute lets one-liner usage stay clean
          while still exposing the report to callers that want it.

    Attributes:
        last_report: The `CleanReport` produced by the most recent
            `clean()` call, or `None` before the first call.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Construct the cleaner.

        Args:
            settings: Override for the `Settings` singleton. Tests
                pass a monkeypatched instance to redirect the
                anchor; production code accepts the default.
        """
        self._settings: Settings = settings or get_settings()
        self._log = get_logger(__name__)
        self.last_report: CleanReport | None = None

    @log_call
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the full cleaning pipeline and return an interim frame.

        Pipeline order:
            1. Drop rows with `TransactionAmt <= 0` (logged per
               reason).
            2. Derive `timestamp` from `TransactionDT` plus the
               configured anchor.
            3. Derive `hour`, `day_of_week`, `is_weekend` from
               `timestamp`.
            4. Standardise email-domain columns (lowercase, strip).
            5. Validate the result against
               `InterimTransactionSchema`.

        Args:
            df: Merged IEEE-CIS frame, post-load. Must satisfy
                `MergedSchema` (the cleaner does not re-validate
                the input — that is the loader's responsibility).

        Returns:
            A new DataFrame with all input columns preserved plus
            four derived calendar columns, validated against
            `InterimTransactionSchema`.

        Raises:
            pandera.errors.SchemaErrors: If the cleaned frame
                violates `InterimTransactionSchema`. Lazy
                validation collects every failure into a single
                exception.
        """
        rows_in = int(len(df))
        cleaned, dropped_by_reason = self._drop_invalid_rows(df)
        cleaned = self._derive_calendar_columns(cleaned)
        cleaned = self._standardise_email_columns(cleaned)
        InterimTransactionSchema.validate(cleaned, lazy=True)
        rows_out = int(len(cleaned))
        report = CleanReport(
            rows_in=rows_in,
            rows_out=rows_out,
            rows_dropped=rows_in - rows_out,
            dropped_by_reason=dropped_by_reason,
            schema_version=INTERIM_SCHEMA_VERSION,
        )
        self.last_report = report
        self._emit_report(report)
        return cleaned

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _drop_invalid_rows(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        """Apply every drop-rule and return ``(filtered_df, counts)``.

        Each rule logs a single ``cleaner.drops`` warning carrying
        its reason code, the row count, and a capped sample of
        affected `TransactionID` values. The capped sample lets an
        operator pull the offending rows for investigation without
        the log record itself ballooning when a future drop reason
        fires on thousands of rows.

        Args:
            df: The input merged frame.

        Returns:
            A two-tuple of ``(filtered_df, dropped_by_reason)``
            where `dropped_by_reason` is a map from reason code to
            count, with zero-count reasons omitted.
        """
        dropped: dict[str, int] = {}
        non_positive_mask = df[_AMOUNT_COLUMN] <= 0
        non_positive_count = int(non_positive_mask.sum())
        if non_positive_count > 0:
            sample_ids = (
                df.loc[non_positive_mask, "TransactionID"]
                .head(_DROP_LOG_SAMPLE_SIZE)
                .tolist()
            )
            self._log.warning(
                "cleaner.drops",
                reason=_NON_POSITIVE_AMOUNT,
                count=non_positive_count,
                sample_ids=sample_ids,
            )
            dropped[_NON_POSITIVE_AMOUNT] = non_positive_count
        # `.copy()` so subsequent in-place column writes in the
        # downstream derivation steps do not mutate a view of the
        # caller's input frame.
        return df.loc[~non_positive_mask].copy(), dropped

    def _derive_calendar_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add `timestamp`, `hour`, `day_of_week`, `is_weekend`.

        The anchor is parsed from ``Settings.transaction_dt_anchor_iso``,
        which is a tz-aware ISO-8601 string. Adding a `pd.Timedelta`
        derived from `TransactionDT` (seconds) produces a tz-aware
        UTC timestamp; pandas `.dt` accessors then yield the
        calendar columns. `is_weekend` is the binary projection of
        Saturday(5)/Sunday(6) on `dayofweek`.

        Args:
            df: A frame already filtered by `_drop_invalid_rows`.

        Returns:
            A new DataFrame with the four calendar columns appended.
            Returns a fresh copy; does not mutate the input.
        """
        anchor = pd.Timestamp(self._settings.transaction_dt_anchor_iso)
        delta = pd.to_timedelta(df[_TIME_COLUMN], unit="s")
        out = df.copy()
        out["timestamp"] = anchor + delta
        # `int` (np.int64 on Linux/WSL) matches `Column(int)` in the
        # interim schema. Avoiding `int8` keeps things explicit:
        # downstream models / SHAP do not need a tighter dtype here.
        out["hour"] = out["timestamp"].dt.hour.astype(int)
        out["day_of_week"] = out["timestamp"].dt.dayofweek.astype(int)
        out["is_weekend"] = (out["day_of_week"] >= _WEEKEND_THRESHOLD).astype(int)
        return out

    def _standardise_email_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lowercase + strip both email-domain columns; preserve NaN.

        The loader (`RawDataLoader._optimize`) typically promotes
        these columns to `category` dtype to halve memory; the
        cleaner round-trips through the nullable ``string`` dtype
        for NaN-safe `.str` ops, then re-casts to the original
        dtype. Categories with `"Gmail.com "` and `"gmail.com"`
        collapse to a single normalised category — the desired
        outcome.

        Args:
            df: A frame post-derivation.

        Returns:
            A new DataFrame with email columns normalised. Columns
            that are absent from `df` (e.g., a synthetic fixture
            that omits them) are silently skipped — those columns
            are nullable and not strictly required at this layer.
        """
        out = df.copy()
        for col in _EMAIL_COLUMNS:
            if col not in out.columns:
                continue
            original_dtype = out[col].dtype
            normalised = out[col].astype("string").str.lower().str.strip()
            if isinstance(original_dtype, pd.CategoricalDtype):
                out[col] = normalised.astype("category")
            else:
                # Cast back to ``object`` so pandera's `Column(object)`
                # check on `MergedSchema` (inherited by interim) does
                # not reject the `StringDtype` left behind by the
                # round-trip.
                out[col] = normalised.astype(object)
        return out

    def _emit_report(self, report: CleanReport) -> None:
        """Log the post-clean summary as a `cleaner.report` info event.

        Args:
            report: The completed `CleanReport` to log.
        """
        self._log.info(
            "cleaner.report",
            rows_in=report.rows_in,
            rows_out=report.rows_out,
            rows_dropped=report.rows_dropped,
            dropped_by_reason=report.dropped_by_reason,
            schema_version=report.schema_version,
        )


__all__ = ["CleanReport", "TransactionCleaner"]
