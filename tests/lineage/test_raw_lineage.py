"""Lineage contracts for the raw IEEE-CIS data.

These assertions enforce what a production fraud pipeline would call
the "ingest SLA": the file fingerprints recorded in MANIFEST.json must
still match the files on disk, the merged DataFrame must preserve the
transaction row count through a left-join, and the headline rates
(fraud ~3.5%, identity coverage ~24%) must match the known snapshot.

Business rationale:
    Every sprint from here on asks "are we still working from the
    same data?" — and that question must be answerable mechanically,
    not by looking at numbers in a notebook. These tests are the
    gate. A Kaggle re-release, a half-extracted ZIP, or an
    accidentally-modified CSV all fail here before they poison
    features or metrics downstream.

Trade-offs considered:
    - Every test is gated by `@pytest.mark.skipif` on the manifest's
      presence so CI green-lights bootstrap runs where raw data is
      not checked out. John can run `make test-lineage` locally once
      data is downloaded.
    - Row-count / rate checks use tight tolerances (≤1% absolute)
      because the known snapshot is deterministic. A drift wider
      than that signals an upstream change worth investigating.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Final

import pandas as pd
import pytest

from fraud_engine.config.settings import get_settings
from fraud_engine.data.loader import RawDataLoader
from fraud_engine.schemas.raw import (
    IdentitySchema,
    MergedSchema,
    TransactionSchema,
)

_HASH_CHUNK_BYTES: Final[int] = 1 << 20  # 1 MiB


def _manifest_path() -> Path:
    return get_settings().raw_dir / "MANIFEST.json"


def _load_manifest() -> dict[str, object]:
    return json.loads(_manifest_path().read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


pytestmark = pytest.mark.skipif(
    not _manifest_path().is_file(),
    reason=(
        "data/raw/MANIFEST.json not present — run `make data-download` "
        "to fetch IEEE-CIS before running lineage tests."
    ),
)


@pytest.fixture(scope="module")
def manifest() -> dict[str, object]:
    """Parsed MANIFEST.json, shared across the module's tests."""
    return _load_manifest()


@pytest.fixture(scope="module")
def merged_df() -> pd.DataFrame:
    """Merged, un-optimised DataFrame for shape and rate assertions.

    The frame is large (~1 GB RAM); `scope=module` keeps one copy
    shared across every test in the file.
    """
    loader = RawDataLoader()
    return loader.load_merged(optimize=False)


def test_manifest_hashes_match_disk(manifest: dict[str, object]) -> None:
    """Every file in MANIFEST.json must still match on disk.

    If this fails, either the raw CSVs were modified in place (do not
    do that) or the manifest is stale (re-run `make data-download
    --force`). Either way, downstream metrics are no longer
    reproducible from the recorded fingerprint.
    """
    raw_dir = get_settings().raw_dir
    recorded = manifest["files"]
    assert isinstance(recorded, dict) and recorded, "manifest has no files entry"
    for name, entry in recorded.items():
        assert isinstance(entry, dict)
        path = raw_dir / name
        assert path.is_file(), f"{name} in manifest but missing on disk"
        assert path.stat().st_size == entry["bytes"], f"byte count drift for {name}"
        assert _sha256(path) == entry["sha256"], f"sha256 drift for {name}"


def test_merged_row_count_equals_transactions() -> None:
    """Left-join must not drop or duplicate transaction rows.

    Uses a fresh loader so we assert on a real invariant, not on the
    fixture's cached frame.
    """
    loader = RawDataLoader()
    tx = loader.load_transactions(optimize=False)
    merged = loader.load_merged(optimize=False)
    assert len(merged) == len(tx)
    assert merged["TransactionID"].is_unique


def test_transaction_id_unique_across_both_tables() -> None:
    """TransactionID is the natural key and must be unique in each CSV.

    Pandas' left-join merge validation would normally catch this, but
    we assert it explicitly so a future refactor that bypasses the
    loader still gets caught.
    """
    loader = RawDataLoader()
    tx = loader.load_transactions(optimize=False)
    idt = loader.load_identity(optimize=False)
    assert tx["TransactionID"].is_unique
    assert idt["TransactionID"].is_unique


def test_fraud_rate_matches_snapshot(merged_df: pd.DataFrame) -> None:
    """Overall fraud rate is ~3.5% (±0.5pp) on the stock Kaggle snapshot."""
    fraud_rate = float(merged_df["isFraud"].mean())
    assert 0.030 <= fraud_rate <= 0.040, f"fraud_rate={fraud_rate:.4f} out of window"


def test_identity_coverage_matches_snapshot(merged_df: pd.DataFrame) -> None:
    """Identity coverage (any id_* present) is ~24% (±2pp)."""
    id_cols = [c for c in merged_df.columns if c.startswith("id_")]
    assert id_cols, "no id_* columns in merged frame"
    has_id = merged_df[id_cols].notna().any(axis=1)
    coverage = float(has_id.mean())
    assert 0.22 <= coverage <= 0.26, f"identity_coverage={coverage:.4f} out of window"


def test_transaction_schema_validates() -> None:
    """`TransactionSchema` must validate the raw transaction CSV."""
    loader = RawDataLoader()
    tx = loader.load_transactions(optimize=False)
    TransactionSchema.validate(tx, lazy=True)


def test_identity_schema_validates() -> None:
    """`IdentitySchema` must validate the raw identity CSV."""
    loader = RawDataLoader()
    idt = loader.load_identity(optimize=False)
    IdentitySchema.validate(idt, lazy=True)


def test_merged_schema_validates(merged_df: pd.DataFrame) -> None:
    """The merged frame must still satisfy its contract after the join."""
    MergedSchema.validate(merged_df, lazy=True)
