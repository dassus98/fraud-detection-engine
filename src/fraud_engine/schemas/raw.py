"""Pandera schemas for the IEEE-CIS raw tables.

These are the *inbound* contracts: they validate what we just read from
the Kaggle CSV, before any dtype optimisation or join. Every downstream
stage then starts from a known-good shape.

Business rationale:
    Fraud pipelines fail silently when a single column's dtype or
    cardinality drifts — a missing `isFraud` row gets coerced to NaN,
    a new `ProductCD` value sneaks through, a V-column disappears.
    Schema validation at the ingest boundary catches those loud, fast,
    with a traceable error rather than a broken feature pipeline four
    sprints later.

Trade-offs considered:
    - `strict=False` so columns we have not enumerated (e.g. a future
      V340) do not fail the load. The columns we *do* list are marked
      `required=True` where the model depends on them, so the failure
      mode is "critical column missing", which is the right signal.
    - Vesta's V1..V339, C1..C14, D1..D15, M1..M9, and id_01..id_38 are
      declared as regex column groups rather than 400+ explicit rows.
      A typo like `V34O` would slip past the regex, but the loader
      reads the header and asserts the expected column count, so a
      drift of that shape is caught a layer up.
    - We validate the *raw* DataFrame (object/int64/float64) here, not
      the dtype-optimised version. Optimisation (category, float32,
      Int8) is tested separately. Keeping the schema tied to the CSV
      contract means schema evolution tracks upstream changes, not our
      internal representation choices.

Version history:
    1 — initial schema; matches Kaggle IEEE-CIS snapshot circa 2019.
        Transactions: 393 required/optional columns; identity: 41
        required/optional columns; merged: transaction ∪ identity.
"""

from __future__ import annotations

from typing import Final

from pandera import Check, Column, DataFrameSchema

SCHEMA_VERSION: Final[int] = 1

# IEEE-CIS documentation fixes these closed-set values. If Kaggle ever
# reshapes the dataset, bump SCHEMA_VERSION and record the drift in the
# module docstring's version history.
_PRODUCT_CODES: Final[list[str]] = ["C", "H", "R", "S", "W"]
_CARD4_BRANDS: Final[list[str]] = [
    "american express",
    "discover",
    "mastercard",
    "visa",
]
_CARD6_TYPES: Final[list[str]] = [
    "charge card",
    "credit",
    "debit",
    "debit or credit",
]
_M_FLAG_VALUES: Final[list[str]] = ["F", "T"]
# M4 is a three-way match indicator per IEEE-CIS docs — distinct from
# the binary T/F flags used by the other M columns.
_M4_VALUES: Final[list[str]] = ["M0", "M1", "M2"]


TransactionSchema: Final[DataFrameSchema] = DataFrameSchema(
    columns={
        # --- identity / target / time / amount (non-null core) ---
        "TransactionID": Column(int, nullable=False, unique=True, required=True),
        "isFraud": Column(int, Check.isin([0, 1]), nullable=False, required=True),
        "TransactionDT": Column(
            int,
            Check.greater_than_or_equal_to(0),
            nullable=False,
            required=True,
        ),
        "TransactionAmt": Column(
            float,
            Check.greater_than(0.0),
            nullable=False,
            required=True,
        ),
        "ProductCD": Column(
            object,
            Check.isin(_PRODUCT_CODES),
            nullable=False,
            required=True,
        ),
        # --- card columns ---
        # card1 is always populated per IEEE-CIS docs; card2/3/5 may be NaN
        # (pandas promotes to float64 on read so nullable=True).
        "card1": Column(int, nullable=False, required=True),
        "card2": Column(float, nullable=True, required=True),
        "card3": Column(float, nullable=True, required=True),
        "card4": Column(
            object,
            Check.isin(_CARD4_BRANDS),
            nullable=True,
            required=True,
        ),
        "card5": Column(float, nullable=True, required=True),
        "card6": Column(
            object,
            Check.isin(_CARD6_TYPES),
            nullable=True,
            required=True,
        ),
        # --- address / distance ---
        "addr1": Column(float, nullable=True, required=True),
        "addr2": Column(float, nullable=True, required=True),
        "dist1": Column(float, nullable=True, required=False),
        "dist2": Column(float, nullable=True, required=False),
        # --- email domains (free-form strings) ---
        "P_emaildomain": Column(object, nullable=True, required=True),
        "R_emaildomain": Column(object, nullable=True, required=True),
        # --- regex column groups ---
        # Vesta's engineered count, delta, match, and V-feature columns.
        # `required=False` so the schema declares dtypes for any
        # matching columns without demanding every sub-block be
        # present. If V300 vanishes upstream we want that caught by a
        # count-based drift test, not by pandera refusing the frame.
        r"^C\d{1,2}$": Column(float, nullable=True, regex=True, required=False),
        r"^D\d{1,2}$": Column(float, nullable=True, regex=True, required=False),
        # Binary match flags (everything except M4).
        r"^M[12356789]$": Column(
            object,
            Check.isin(_M_FLAG_VALUES),
            nullable=True,
            regex=True,
            required=False,
        ),
        # M4 is the three-way match column.
        "M4": Column(
            object,
            Check.isin(_M4_VALUES),
            nullable=True,
            required=False,
        ),
        r"^V\d{1,3}$": Column(float, nullable=True, regex=True, required=False),
    },
    strict=False,
    ordered=False,
    name="transaction_v1",
)


# IEEE-CIS identity split: of the id_12..id_38 range, a sub-set loads
# as float64 (numeric codes with NaN) while the rest loads as object
# (strings / mixed). The split was confirmed against
# `train_identity.csv` shipped by Kaggle. Drifts in this mapping are
# exactly what schema validation should surface — a loud failure at
# ingest beats a silent dtype coercion in feature engineering.
_IDENTITY_NUMERIC_COLS: Final[list[str]] = [
    "id_01",
    "id_02",
    "id_03",
    "id_04",
    "id_05",
    "id_06",
    "id_07",
    "id_08",
    "id_09",
    "id_10",
    "id_11",
    "id_13",
    "id_14",
    "id_17",
    "id_18",
    "id_19",
    "id_20",
    "id_21",
    "id_22",
    "id_24",
    "id_25",
    "id_26",
    "id_32",
]
_IDENTITY_OBJECT_COLS: Final[list[str]] = [
    "id_12",
    "id_15",
    "id_16",
    "id_23",
    "id_27",
    "id_28",
    "id_29",
    "id_30",
    "id_31",
    "id_33",
    "id_34",
    "id_35",
    "id_36",
    "id_37",
    "id_38",
]

_identity_columns: dict[str, Column] = {
    "TransactionID": Column(int, nullable=False, unique=True, required=True),
    "DeviceType": Column(object, nullable=True, required=False),
    "DeviceInfo": Column(object, nullable=True, required=False),
}
# `required=False` so sparse fixtures (e.g. synthetic tests that only
# carry id_01, id_02, id_12) validate. Production loads carry the full
# set and hit the dtype checks below.
for _col in _IDENTITY_NUMERIC_COLS:
    _identity_columns[_col] = Column(float, nullable=True, required=False)
for _col in _IDENTITY_OBJECT_COLS:
    _identity_columns[_col] = Column(object, nullable=True, required=False)

IdentitySchema: Final[DataFrameSchema] = DataFrameSchema(
    columns=_identity_columns,
    strict=False,
    ordered=False,
    name="identity_v1",
)


MergedSchema: Final[DataFrameSchema] = DataFrameSchema(
    columns={
        # Transaction-side required columns survive the left-join.
        "TransactionID": Column(int, nullable=False, unique=True, required=True),
        "isFraud": Column(int, Check.isin([0, 1]), nullable=False, required=True),
        "TransactionDT": Column(
            int,
            Check.greater_than_or_equal_to(0),
            nullable=False,
            required=True,
        ),
        "TransactionAmt": Column(
            float,
            Check.greater_than(0.0),
            nullable=False,
            required=True,
        ),
        "ProductCD": Column(
            object,
            Check.isin(_PRODUCT_CODES),
            nullable=False,
            required=True,
        ),
        # Identity-side columns are optional — roughly 24% coverage on
        # IEEE-CIS means the majority of merged rows carry NaN here.
        "DeviceType": Column(object, nullable=True, required=False),
        "DeviceInfo": Column(object, nullable=True, required=False),
    },
    strict=False,
    ordered=False,
    name="merged_v1",
)


__all__ = [
    "IdentitySchema",
    "MergedSchema",
    "SCHEMA_VERSION",
    "TransactionSchema",
]
