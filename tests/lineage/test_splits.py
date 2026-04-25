"""Lineage contracts for the Sprint 1 temporal splitter.

These tests run against the *real* merged IEEE-CIS frame — not a
synthetic fixture — so they catch drift between the splitter's
behaviour and the snapshot numbers quoted in
`reports/sprint1_eda_summary.md`. The unit suite in
`tests/unit/test_splits.py` covers boundary conditions on
hand-built frames; this file answers the coarser question: "does
the splitter, run with repo defaults on the real dataset, still
land where Sprint 1 said it would?"

Business rationale:
    Sprint 2's feature pipeline and Sprint 4's cost-curve evaluation
    both depend on the assumption that the train/val/test row
    membership is stable across sprints. A refactor that quietly
    shifted the split would silently revalue every downstream
    metric. Pinning the partition to mechanical assertions here
    surfaces that class of bug in CI rather than in a sprint
    post-mortem.

Trade-offs considered:
    - All tests are gated on `data/raw/MANIFEST.json` so
      bootstrap-only CI runs pass without the 1.2 GB dataset.
    - Tolerances on fraud rate follow the ±0.5pp window quoted in
      the splits docstring; the existing raw-lineage suite already
      proves the overall 3.5% rate, so we only need to confirm the
      three sub-slices stay close to it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fraud_engine.config.settings import get_settings
from fraud_engine.data.loader import RawDataLoader
from fraud_engine.data.splits import (
    SplitFrames,
    temporal_split,
    validate_no_overlap,
    write_split_manifest,
)


def _manifest_path() -> Path:
    return get_settings().raw_dir / "MANIFEST.json"


pytestmark = [
    pytest.mark.lineage,
    pytest.mark.skipif(
        not _manifest_path().is_file(),
        reason=(
            "data/raw/MANIFEST.json not present — run `make data-download` "
            "to fetch IEEE-CIS before running lineage tests."
        ),
    ),
]


@pytest.fixture(scope="module")
def merged_df() -> pd.DataFrame:
    """Load the full merged frame once per module for split assertions."""
    loader = RawDataLoader()
    return loader.load_merged(optimize=False)


@pytest.fixture(scope="module")
def splits(merged_df: pd.DataFrame) -> SplitFrames:
    """Canonical splits produced from repo-default Settings."""
    return temporal_split(merged_df)


def test_every_row_in_exactly_one_split(merged_df: pd.DataFrame, splits: SplitFrames) -> None:
    """Union of TransactionID sets == original; pairwise intersection is empty."""
    original_ids = set(merged_df["TransactionID"].tolist())
    train_ids = set(splits.train["TransactionID"].tolist())
    val_ids = set(splits.val["TransactionID"].tolist())
    test_ids = set(splits.test["TransactionID"].tolist())

    assert train_ids | val_ids | test_ids == original_ids
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert len(train_ids) + len(val_ids) + len(test_ids) == len(original_ids)


def test_fraud_rates_within_tolerance(splits: SplitFrames) -> None:
    """Each split's fraud rate is within ±0.5pp of the overall 3.5% rate.

    The overall rate is pinned by the raw-lineage suite; we only
    confirm here that the calendar-month partition does not
    accidentally concentrate fraud in one slice.
    """
    overall = splits.manifest["fraud_rate_overall"]
    for key in ("fraud_rate_train", "fraud_rate_val", "fraud_rate_test"):
        rate = splits.manifest[key]
        assert abs(rate - overall) <= 0.005, (
            f"{key}={rate:.4f} deviates from overall={overall:.4f} by " f"more than 0.5pp"
        )


def test_temporal_bounds_honoured(splits: SplitFrames) -> None:
    """Every row lands on the correct side of `train_end_dt` / `val_end_dt`."""
    settings = get_settings()
    assert splits.train["TransactionDT"].max() < settings.train_end_dt
    assert splits.val["TransactionDT"].min() >= settings.train_end_dt
    assert splits.val["TransactionDT"].max() < settings.val_end_dt
    assert splits.test["TransactionDT"].min() >= settings.val_end_dt


def test_manifest_round_trip(splits: SplitFrames, tmp_path: Path) -> None:
    """`write_split_manifest` → `json.loads` → equality with the in-memory dict."""
    out = tmp_path / "splits_manifest.json"
    write_split_manifest(splits, out)
    reloaded = json.loads(out.read_text(encoding="utf-8"))
    assert reloaded == splits.manifest


def test_validate_no_overlap_raises_on_bad_input(merged_df: pd.DataFrame) -> None:
    """Hand-build an overlapping `SplitFrames` and assert the validator rejects it.

    We deliberately corrupt the partition by copying a single row
    from train into val. If `validate_no_overlap` ever stopped
    catching that, a bug upstream could silently bleed training
    rows into the validation set.
    """
    clean = temporal_split(merged_df)
    # Splice one training row into val so the TransactionID sets overlap.
    leak_row = clean.train.iloc[[0]]
    bad_val = pd.concat([clean.val, leak_row], ignore_index=True)
    corrupted = SplitFrames(
        train=clean.train,
        val=bad_val,
        test=clean.test,
        manifest=clean.manifest,
    )
    with pytest.raises(ValueError, match="TransactionID overlap"):
        validate_no_overlap(corrupted)
