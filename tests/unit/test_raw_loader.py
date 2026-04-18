"""Unit tests for `fraud_engine.data.loader.RawDataLoader`.

Uses synthetic CSVs written into `tmp_path` so no test ever depends
on the real `data/raw/` directory. The lineage suite covers the
real-data path once `make data-download` has run.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fraud_engine.config.settings import Settings
from fraud_engine.data.loader import RawDataLoader
from fraud_engine.schemas.raw import IdentitySchema, TransactionSchema


def _minimal_transaction_df() -> pd.DataFrame:
    """Return five rows that satisfy `TransactionSchema`."""
    return pd.DataFrame(
        {
            "TransactionID": [100, 101, 102, 103, 104],
            "isFraud": [0, 1, 0, 1, 0],
            "TransactionDT": [1_000, 2_000, 3_000, 4_000, 5_000],
            "TransactionAmt": [10.0, 25.0, 30.0, 42.5, 50.0],
            "ProductCD": ["W", "W", "H", "R", "S"],
            "card1": [1001, 1002, 1003, 1004, 1005],
            "card2": [100.0, 200.0, None, 400.0, 500.0],
            "card3": [150.0, None, 150.0, 150.0, 150.0],
            "card4": ["visa", "mastercard", None, "discover", "visa"],
            "card5": [100.0, 100.0, 100.0, None, 100.0],
            "card6": ["credit", "debit", "credit", "debit", None],
            "addr1": [100.0, None, 200.0, 300.0, 400.0],
            "addr2": [87.0, 87.0, 87.0, 87.0, None],
            "P_emaildomain": ["gmail.com", "yahoo.com", None, "outlook.com", None],
            "R_emaildomain": [None, "gmail.com", None, "yahoo.com", None],
            "C1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "D1": [10.0, 20.0, None, 40.0, 50.0],
            "M1": ["T", "F", None, "T", "F"],
            "V1": [0.5, 0.6, 0.7, None, 0.9],
        }
    )


def _minimal_identity_df() -> pd.DataFrame:
    """Return three rows that satisfy `IdentitySchema` (fewer than txn)."""
    return pd.DataFrame(
        {
            "TransactionID": [100, 102, 104],
            "DeviceType": ["desktop", "mobile", "desktop"],
            "DeviceInfo": ["Windows", "iOS Device", "Android"],
            "id_01": [0.0, -5.0, -12.0],
            "id_02": [100.0, 200.0, 300.0],
            "id_12": ["NotFound", "Found", "NotFound"],
        }
    )


@pytest.fixture
def synthetic_raw_dir(tmp_path: Path) -> Path:
    """Write schema-compliant synthetic CSVs into a fresh raw dir."""
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    _minimal_transaction_df().to_csv(raw_dir / "train_transaction.csv", index=False)
    _minimal_identity_df().to_csv(raw_dir / "train_identity.csv", index=False)
    return raw_dir


@pytest.fixture
def loader_with_synthetic_tree(
    synthetic_raw_dir: Path,
    tmp_path: Path,
) -> RawDataLoader:
    """Build a loader pointing at the synthetic raw dir."""
    settings = Settings(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        logs_dir=tmp_path / "logs",
    )
    return RawDataLoader(raw_dir=synthetic_raw_dir, settings=settings)


def test_raises_on_missing_file(tmp_path: Path) -> None:
    """Loader surfaces a clear FileNotFoundError when the CSV is absent."""
    loader = RawDataLoader(raw_dir=tmp_path / "nope")
    with pytest.raises(FileNotFoundError, match="train_transaction.csv"):
        loader.load_transactions()


def test_load_transactions_passes_schema(loader_with_synthetic_tree: RawDataLoader) -> None:
    """A schema-compliant CSV round-trips through the loader."""
    df = loader_with_synthetic_tree.load_transactions(optimize=False)
    TransactionSchema.validate(df, lazy=True)
    assert df.shape[0] == 5


def test_load_identity_passes_schema(loader_with_synthetic_tree: RawDataLoader) -> None:
    """Identity CSV validates and loads."""
    df = loader_with_synthetic_tree.load_identity(optimize=False)
    IdentitySchema.validate(df, lazy=True)
    assert df.shape[0] == 3


def test_load_merged_preserves_transaction_row_count(
    loader_with_synthetic_tree: RawDataLoader,
) -> None:
    """Left-join keeps every transaction row even when identity is sparse."""
    merged = loader_with_synthetic_tree.load_merged(optimize=False)
    assert merged.shape[0] == 5
    assert merged["TransactionID"].is_unique
    # Only 3 of 5 transactions matched identity; the rest carry NaN.
    assert merged["DeviceType"].notna().sum() == 3


def test_optimize_reduces_memory(loader_with_synthetic_tree: RawDataLoader) -> None:
    """Optimised frame is strictly smaller than the un-optimised one."""
    raw = loader_with_synthetic_tree.load_transactions(optimize=False)
    opt = loader_with_synthetic_tree.load_transactions(optimize=True)
    raw_bytes = int(raw.memory_usage(deep=True).sum())
    opt_bytes = int(opt.memory_usage(deep=True).sum())
    assert opt_bytes < raw_bytes
    # Float columns should downcast to float32.
    assert opt["TransactionAmt"].dtype.itemsize == 4
