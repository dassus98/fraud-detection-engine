"""Unit tests for `fraud_engine.schemas.raw`.

These tests exercise the schemas directly (not through the loader) so
validation failures surface as `pandera.errors.SchemaError` with the
exact offending column named. A companion positive-path suite lives
in `test_raw_loader.py`, which validates through the full read → coerce
→ validate loader flow, and a real-data integrity suite lives in
`tests/lineage/test_raw_lineage.py`.

Business rationale:
    Schema validation is the ingest boundary; if this layer ever stops
    rejecting bad data, every downstream feature pipeline, split, and
    model silently inherits the corruption. We pin the rejection
    behaviour here so a future refactor of `raw.py` cannot loosen a
    check without a test going red.

Trade-offs considered:
    - We test the minimal set of invariants promised in CLAUDE.md §7:
      uniqueness on the natural key, binary-label domain, positive
      monetary amount, and closed-set categorical (`ProductCD`). Every
      additional `Check.isin` added to the schema should come with its
      own rejection test — we do not try to enumerate them exhaustively
      here.
    - `lazy=True` on `.validate()` so a single call surfaces every
      violation in one `SchemaErrors` (plural), but the spec requires
      `SchemaError` (singular). We call with `lazy=False` for the
      negative-path tests so the raised type matches the spec wording,
      and use `lazy=True` for the positive-path happy case to mirror
      production usage.
"""

from __future__ import annotations

import pandas as pd
import pandera.errors as pa_errors
import pytest

from fraud_engine.schemas.raw import (
    IdentitySchema,
    TransactionSchema,
)


def _valid_transactions() -> pd.DataFrame:
    """Return five rows that satisfy `TransactionSchema`.

    The shape is deliberately minimal — only the required columns plus
    one representative of each regex group (C1, D1, M1, V1). Optional
    dist1/dist2 are intentionally omitted to confirm the schema does
    not insist on them.

    Returns:
        Schema-compliant DataFrame with 5 rows.
    """
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4, 5],
            "isFraud": [0, 1, 0, 1, 0],
            "TransactionDT": [100, 200, 300, 400, 500],
            "TransactionAmt": [10.0, 20.5, 33.3, 44.0, 55.75],
            "ProductCD": ["W", "C", "H", "R", "S"],
            "card1": [1001, 1002, 1003, 1004, 1005],
            "card2": [100.0, None, 200.0, 300.0, 400.0],
            "card3": [150.0, 150.0, None, 150.0, 150.0],
            "card4": ["visa", "mastercard", None, "visa", "discover"],
            "card5": [100.0, 100.0, 200.0, None, 100.0],
            "card6": ["credit", "debit", "credit", "debit", None],
            "addr1": [100.0, 200.0, None, 300.0, 400.0],
            "addr2": [87.0, 87.0, 87.0, None, 87.0],
            "P_emaildomain": ["gmail.com", None, "yahoo.com", "outlook.com", None],
            "R_emaildomain": [None, "gmail.com", None, "yahoo.com", None],
            "C1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "D1": [10.0, None, 30.0, 40.0, 50.0],
            "M1": ["T", "F", None, "T", "F"],
            "V1": [0.1, 0.2, 0.3, None, 0.5],
        }
    )


def _valid_identity() -> pd.DataFrame:
    """Return three rows that satisfy `IdentitySchema`.

    Returns:
        Schema-compliant identity DataFrame with 3 rows.
    """
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "DeviceType": ["desktop", "mobile", None],
            "DeviceInfo": ["Windows", "iOS 14", None],
            "id_01": [0.0, -5.0, -10.0],
            "id_12": ["NotFound", "Found", None],
        }
    )


def test_transaction_schema_accepts_valid_sample() -> None:
    """A schema-compliant transaction frame round-trips without error."""
    df = _valid_transactions()
    validated = TransactionSchema.validate(df, lazy=True)
    assert validated.shape == df.shape
    assert list(validated.columns) == list(df.columns)


def test_identity_schema_accepts_valid_sample() -> None:
    """A schema-compliant identity frame round-trips without error."""
    df = _valid_identity()
    validated = IdentitySchema.validate(df, lazy=True)
    assert validated.shape == df.shape


def test_transaction_schema_rejects_negative_amount() -> None:
    """`TransactionAmt` must be strictly positive (Check.greater_than(0))."""
    df = _valid_transactions()
    df.loc[0, "TransactionAmt"] = -5.0
    with pytest.raises(pa_errors.SchemaError, match="TransactionAmt"):
        TransactionSchema.validate(df)


def test_transaction_schema_rejects_unknown_product_cd() -> None:
    """`ProductCD` is a closed set {C, H, R, S, W}; unknown codes fail."""
    df = _valid_transactions()
    df.loc[2, "ProductCD"] = "Z"
    with pytest.raises(pa_errors.SchemaError, match="ProductCD"):
        TransactionSchema.validate(df)


def test_transaction_schema_rejects_isfraud_outside_binary() -> None:
    """`isFraud` domain is {0, 1}; any other value must be rejected."""
    df = _valid_transactions()
    df.loc[1, "isFraud"] = 2
    with pytest.raises(pa_errors.SchemaError, match="isFraud"):
        TransactionSchema.validate(df)


def test_transaction_schema_rejects_duplicate_transaction_id() -> None:
    """`TransactionID` is the natural key and must be unique."""
    df = _valid_transactions()
    df.loc[4, "TransactionID"] = df.loc[0, "TransactionID"]
    with pytest.raises(pa_errors.SchemaError, match="TransactionID"):
        TransactionSchema.validate(df)


def test_identity_schema_rejects_duplicate_transaction_id() -> None:
    """Uniqueness invariant also holds on the identity side."""
    df = _valid_identity()
    df.loc[2, "TransactionID"] = df.loc[0, "TransactionID"]
    with pytest.raises(pa_errors.SchemaError, match="TransactionID"):
        IdentitySchema.validate(df)
