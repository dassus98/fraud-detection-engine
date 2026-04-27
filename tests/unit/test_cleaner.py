"""Unit tests for `fraud_engine.data.cleaner.TransactionCleaner`.

Uses a small in-memory synthetic frame designed against
`MergedSchema`'s required column set. No filesystem fixtures —
the cleaner is a pure transformation, so a DataFrame fixture is
sufficient.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from fraud_engine.config.settings import Settings
from fraud_engine.data.cleaner import CleanReport, TransactionCleaner
from fraud_engine.schemas.interim import (
    INTERIM_SCHEMA_VERSION,
    InterimTransactionSchema,
)

# TransactionDT values in seconds since the default anchor
# (2017-12-01T00:00:00+00:00). The mapping is:
#   0       → 2017-12-01 (Friday)       — weekday
#   86_400  → 2017-12-02 (Saturday)     — weekend
#   172_800 → 2017-12-03 (Sunday)       — weekend
#   259_200 → 2017-12-04 (Monday)       — weekday
#   345_600 → 2017-12-05 (Tuesday)      — weekday
# Combined with intra-day offsets like 14h (50_400s) the fixture
# exercises both calendar arithmetic and weekend detection.
_FRI_MIDNIGHT: int = 0
_SAT_MIDNIGHT: int = 86_400
_SUN_MIDNIGHT: int = 172_800
_MON_MIDNIGHT: int = 259_200
_TUE_MIDNIGHT: int = 345_600
_FRI_2PM: int = 50_400
_SAT_9PM: int = 86_400 + 75_600
_SUN_7AM: int = 172_800 + 25_200


def _merged_fixture_df() -> pd.DataFrame:
    """Return a 10-row frame satisfying `MergedSchema`'s required set.

    Row design:
        - rows 5 and 6 carry `TransactionAmt <= 0` → must drop
        - rows 2 / 3 / 8 / 9 / 10 land on weekends (Sat/Sun)
        - rows 2, 3, 8 carry email domains needing normalisation
        - row 4 has NaN emails (passthrough)

    The frame includes the six required `MergedSchema` columns plus
    the two email columns (`P_emaildomain`, `R_emaildomain`) which
    are required at `TransactionSchema` level and need to be present
    for the email-standardisation tests.
    """
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "isFraud": [0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
            "TransactionDT": [
                _FRI_MIDNIGHT,
                _SAT_MIDNIGHT,
                _SUN_MIDNIGHT,
                _FRI_2PM,
                _FRI_MIDNIGHT,
                _FRI_MIDNIGHT,
                _MON_MIDNIGHT,
                _SAT_9PM,
                _SUN_7AM,
                _TUE_MIDNIGHT,
            ],
            "TransactionAmt": [
                10.0,
                25.0,
                30.0,
                50.0,
                0.0,
                -5.0,
                12.5,
                100.0,
                200.0,
                7.0,
            ],
            "ProductCD": ["W", "W", "H", "W", "W", "W", "W", "C", "R", "S"],
            "P_emaildomain": [
                "gmail.com",
                "Gmail.com ",
                "  YAHOO.com",
                None,
                "gmail.com",
                "yahoo.com",
                "outlook.com",
                "gmail.com",
                "yahoo.com",
                "gmail.com",
            ],
            "R_emaildomain": [
                "yahoo.com",
                None,
                "OUTLOOK.com",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
        }
    )


@pytest.fixture
def cleaner() -> TransactionCleaner:
    """A cleaner using the default Settings (anchor 2017-12-01 UTC)."""
    return TransactionCleaner(settings=Settings())


def test_drops_rows_with_non_positive_amount(cleaner: TransactionCleaner) -> None:
    """Rows with TransactionAmt ≤ 0 are dropped and accounted for."""
    out = cleaner.clean(_merged_fixture_df())
    assert len(out) == 8
    assert cleaner.last_report is not None
    assert cleaner.last_report.rows_dropped == 2
    assert cleaner.last_report.dropped_by_reason == {"non_positive_amount": 2}
    # The dropped TransactionIDs must not appear in the output.
    assert 5 not in out["TransactionID"].tolist()
    assert 6 not in out["TransactionID"].tolist()


def test_email_standardisation_lowercases_and_strips(
    cleaner: TransactionCleaner,
) -> None:
    """ "Gmail.com " → "gmail.com"; NaN passes through unchanged."""
    out = cleaner.clean(_merged_fixture_df())
    by_id = out.set_index("TransactionID")
    # Original "Gmail.com " on row 2 → "gmail.com"
    assert by_id.loc[2, "P_emaildomain"] == "gmail.com"
    # Original "  YAHOO.com" on row 3 → "yahoo.com"
    assert by_id.loc[3, "P_emaildomain"] == "yahoo.com"
    # Original "OUTLOOK.com" on R column row 3 → "outlook.com"
    assert by_id.loc[3, "R_emaildomain"] == "outlook.com"
    # NaN inputs stay NaN
    assert pd.isna(by_id.loc[4, "P_emaildomain"])
    assert pd.isna(by_id.loc[4, "R_emaildomain"])


def test_output_validates_against_interim_schema(
    cleaner: TransactionCleaner,
) -> None:
    """The cleaner's output passes `InterimTransactionSchema` checks."""
    out = cleaner.clean(_merged_fixture_df())
    InterimTransactionSchema.validate(out, lazy=True)
    assert isinstance(out["timestamp"].dtype, pd.DatetimeTZDtype)
    assert str(out["timestamp"].dtype) == "datetime64[ns, UTC]"
    assert out["hour"].between(0, 23, inclusive="both").all()
    assert out["day_of_week"].between(0, 6, inclusive="both").all()
    assert set(out["is_weekend"].unique()).issubset({0, 1})


def test_dropped_count_matches_log_entries(
    cleaner: TransactionCleaner,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Per-reason drop counts in logs match the report's totals.

    The project's structlog→stdlib bridge places the structured
    event dict in `record.msg` (a dict), not on individual record
    attributes — see `src/fraud_engine/utils/logging.py` for the
    `ProcessorFormatter.wrap_for_formatter` configuration.
    """
    with caplog.at_level(logging.WARNING, logger="fraud_engine.data.cleaner"):
        cleaner.clean(_merged_fixture_df())
    drop_records = [
        r
        for r in caplog.records
        if isinstance(r.msg, dict) and r.msg.get("event") == "cleaner.drops"
    ]
    assert len(drop_records) == 1, "exactly one drop-reason fired"
    event = drop_records[0].msg
    assert event["reason"] == "non_positive_amount"
    assert event["count"] == 2
    assert event["sample_ids"] == [5, 6]
    # Row-count invariant: report.rows_dropped == sum of per-reason counts
    assert cleaner.last_report is not None
    assert cleaner.last_report.rows_dropped == sum(cleaner.last_report.dropped_by_reason.values())


def test_calendar_columns_are_correct_for_known_dates(
    cleaner: TransactionCleaner,
) -> None:
    """`hour`/`day_of_week`/`is_weekend` follow the spec-supplied anchor."""
    out = cleaner.clean(_merged_fixture_df())
    by_id = out.set_index("TransactionID")
    # Row 2: TransactionDT=86400 → 2017-12-02 (Sat)
    assert by_id.loc[2, "timestamp"] == pd.Timestamp("2017-12-02T00:00:00+00:00")
    assert by_id.loc[2, "hour"] == 0
    assert by_id.loc[2, "day_of_week"] == 5
    assert by_id.loc[2, "is_weekend"] == 1
    # Row 7: TransactionDT=259200 → 2017-12-04 (Mon)
    assert by_id.loc[7, "day_of_week"] == 0
    assert by_id.loc[7, "is_weekend"] == 0
    # Row 8: TransactionDT=86400+75600 → 2017-12-02 21:00 (Sat 9pm)
    assert by_id.loc[8, "hour"] == 21
    assert by_id.loc[8, "day_of_week"] == 5
    assert by_id.loc[8, "is_weekend"] == 1
    # Row 9: TransactionDT=172800+25200 → 2017-12-03 07:00 (Sun 7am)
    assert by_id.loc[9, "hour"] == 7
    assert by_id.loc[9, "day_of_week"] == 6
    assert by_id.loc[9, "is_weekend"] == 1


def test_email_dtype_preserved_when_input_is_category(
    cleaner: TransactionCleaner,
) -> None:
    """Category-typed emails round-trip through normalisation."""
    df = _merged_fixture_df()
    df["P_emaildomain"] = df["P_emaildomain"].astype("category")
    out = cleaner.clean(df)
    assert isinstance(out["P_emaildomain"].dtype, pd.CategoricalDtype)
    # The normalisation collapses "Gmail.com " and "gmail.com" into a
    # single category — no orphaned mixed-case categories remain.
    categories = list(out["P_emaildomain"].cat.categories)
    for cat in categories:
        assert cat == cat.lower().strip()


def test_clean_does_not_mutate_input_frame(cleaner: TransactionCleaner) -> None:
    """The input DataFrame is unchanged after `clean()`."""
    df = _merged_fixture_df()
    df_snapshot = df.copy(deep=True)
    cleaner.clean(df)
    pd.testing.assert_frame_equal(df, df_snapshot)


def test_clean_idempotent_on_already_clean_data(
    cleaner: TransactionCleaner,
) -> None:
    """A second `clean()` on already-clean output produces the same frame."""
    # Drop the two bad-amount rows up-front to give a clean input.
    df = _merged_fixture_df().query("TransactionAmt > 0").reset_index(drop=True)
    first = cleaner.clean(df)
    assert cleaner.last_report is not None
    assert cleaner.last_report.rows_dropped == 0
    assert cleaner.last_report.dropped_by_reason == {}
    # Second call: input already has timestamp/hour/day_of_week/
    # is_weekend columns; the derivation overwrites them with the
    # same values.
    second = cleaner.clean(first)
    pd.testing.assert_frame_equal(first, second)


def test_settings_injection_uses_custom_anchor() -> None:
    """A custom `transaction_dt_anchor_iso` shifts the derived timestamp."""
    custom_settings = Settings(
        transaction_dt_anchor_iso="2018-01-01T00:00:00+00:00",
    )
    custom_cleaner = TransactionCleaner(settings=custom_settings)
    df = _merged_fixture_df().head(1).copy()  # row with TransactionDT=0
    out = custom_cleaner.clean(df)
    assert out["timestamp"].iloc[0] == pd.Timestamp("2018-01-01T00:00:00+00:00")


def test_schema_rejects_corrupted_output(cleaner: TransactionCleaner) -> None:
    """Hand-corrupting an output column trips the schema check."""
    out = cleaner.clean(_merged_fixture_df())
    out.loc[out.index[0], "hour"] = 99  # outside [0, 23]
    with pytest.raises(SchemaErrors):
        InterimTransactionSchema.validate(out, lazy=True)


def test_clean_report_carries_schema_version(
    cleaner: TransactionCleaner,
) -> None:
    """`CleanReport.schema_version` reflects `INTERIM_SCHEMA_VERSION`."""
    cleaner.clean(_merged_fixture_df())
    assert isinstance(cleaner.last_report, CleanReport)
    assert cleaner.last_report.schema_version == INTERIM_SCHEMA_VERSION
