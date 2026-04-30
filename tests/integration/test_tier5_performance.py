"""Tier-5 graph-feature performance benchmark.

Spec ceiling: <20 min on the full IEEE-CIS train split (~414 k rows).
Skip-gated on `data/raw/train_transaction.csv` (mirrors the 3.2.a CI
follow-up — `MANIFEST.json` is checked into git, so manifest-presence
gating fires false-positive on CI runners).

Two test classes:

- `TestEndToEnd10k`: 10 k stratified sample; `fit_transform(train_80)`
  → `transform(val_20)` round-trip; verifies all 8 columns present,
  no `inf`, NaN-rates stay within sane bounds.
- `TestPerformance` (`@pytest.mark.slow`): full 414 k train split;
  hard gate on wall-time < 20 min; echoes per-column stats to stdout
  for the completion report.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from fraud_engine.config.settings import get_settings
from fraud_engine.data.cleaner import TransactionCleaner
from fraud_engine.data.loader import RawDataLoader
from fraud_engine.data.splits import temporal_split
from fraud_engine.features.tier5_graph import GraphFeatureExtractor

pytestmark = pytest.mark.integration

# Spec ceiling: <20 min on full data.
_PERF_CEILING_SECONDS: float = 20.0 * 60.0  # 1200 s

# Stratified-sample size for the 10k end-to-end test. Module-scoped so
# the ~30 s load + cleaner amortises across all tests in the file.
_INTEGRATION_SAMPLE_SIZE: int = 10_000
_INTEGRATION_SEED: int = 42

# 80/20 temporal split for the 10k end-to-end test (same fraction
# pattern as `test_tier4_performance.py`).
_TEMPORAL_SPLIT_FRACTION: float = 0.8

# Expected feature count: 1 cc_size + 4 entity_degree + 1
# fraud_neighbor_rate + 1 pagerank + 1 clustering = 8.
_EXPECTED_FEATURE_COUNT: int = 8


def _train_csv_path() -> Path:
    """Path to the gitignored full IEEE-CIS train transactions CSV.

    Mirrors `tests/unit/test_tier5_graph_construction.py` — the
    presence of this CSV (NOT the always-tracked `MANIFEST.json`
    sidecar) gates the full-data benchmark.
    """
    return get_settings().raw_dir / "train_transaction.csv"


@pytest.fixture(scope="module")
def merged_10k_cleaned() -> pd.DataFrame:
    """10 k stratified sample of the cleaned merged frame.

    Module-scoped so the ~30 s load + cleaner amortises across all
    tests in this file. Skip-gated on the gitignored CSV.
    """
    if not _train_csv_path().is_file():
        pytest.skip("data/raw/train_transaction.csv not present — run `make data-download`.")
    loader = RawDataLoader()
    full = loader.load_merged(optimize=False)
    cleaned = TransactionCleaner().clean(full)
    sample, _ = train_test_split(
        cleaned,
        train_size=_INTEGRATION_SAMPLE_SIZE,
        stratify=cleaned["isFraud"],
        random_state=_INTEGRATION_SEED,
    )
    return sample.reset_index(drop=True)


@pytest.fixture(scope="module")
def merged_train_split() -> pd.DataFrame:
    """Full 414 k train split (post cleaner + temporal_split).

    Module-scoped; skip-gated on the gitignored CSV.
    """
    if not _train_csv_path().is_file():
        pytest.skip("data/raw/train_transaction.csv not present — run `make data-download`.")
    settings = get_settings()
    loader = RawDataLoader()
    full = loader.load_merged(optimize=False)
    cleaned = TransactionCleaner().clean(full)
    splits = temporal_split(cleaned, settings=settings)
    return splits.train.reset_index(drop=True)


# ---------------------------------------------------------------------
# `TestEndToEnd10k`: fast e2e correctness on a 10 k stratified sample.
# ---------------------------------------------------------------------


class TestEndToEnd10k:
    """End-to-end fit_transform / transform round-trip on a 10 k sample."""

    def test_fit_transform_then_transform(self, merged_10k_cleaned: pd.DataFrame) -> None:
        """fit_transform on train_80; transform on val_20; sanity-check shape."""
        sample = merged_10k_cleaned.sort_values("TransactionDT").reset_index(drop=True)
        cut = int(len(sample) * _TEMPORAL_SPLIT_FRACTION)
        train = sample.iloc[:cut].reset_index(drop=True)
        val = sample.iloc[cut:].reset_index(drop=True)

        gen = GraphFeatureExtractor()
        train_out = gen.fit_transform(train)
        val_out = gen.transform(val)

        feature_names = gen.get_feature_names()
        assert len(feature_names) == _EXPECTED_FEATURE_COUNT

        # Train output: every feature column must be present; no `inf`
        # values (NaN is fine — singleton entities, all-NaN rows).
        for col in feature_names:
            assert col in train_out.columns
            arr = train_out[col].to_numpy()
            assert not np.any(np.isinf(arr)), f"inf in train {col}"
        assert len(train_out) == len(train)

        # Val output: same column shape; NaN rates within sane bounds.
        for col in feature_names:
            assert col in val_out.columns
            arr = val_out[col].to_numpy()
            assert not np.any(np.isinf(arr)), f"inf in val {col}"
        assert len(val_out) == len(val)

        # Val txn-level features are 100% NaN by design.
        assert val_out["connected_component_size"].isna().all()
        assert val_out["pagerank_score"].isna().all()
        assert val_out["clustering_coefficient"].isna().all()


# ---------------------------------------------------------------------
# `TestPerformance`: full-data wall-time hard gate.
# ---------------------------------------------------------------------


class TestPerformance:
    """Spec contract: full 414 k train split fits in <20 min wall-time."""

    @pytest.mark.slow
    def test_full_data_under_20min(self, merged_train_split: pd.DataFrame) -> None:
        """fit_transform on the full train split; HARD gate on wall-time."""
        gen = GraphFeatureExtractor()
        start = time.perf_counter()
        out = gen.fit_transform(merged_train_split)
        elapsed = time.perf_counter() - start

        feature_names = gen.get_feature_names()

        # Echo to stdout for the completion report.
        print(
            f"\n[tier5-feat-perf] rows = {len(out):,}; "
            f"elapsed = {elapsed:.1f}s ({elapsed / 60:.2f} min)"
        )
        for col in feature_names:
            arr = out[col].to_numpy()
            nan_pct = np.isnan(arr).mean() * 100
            valid = arr[~np.isnan(arr)]
            min_v = float(np.min(valid)) if valid.size else float("nan")
            max_v = float(np.max(valid)) if valid.size else float("nan")
            print(
                f"[tier5-feat-perf]   {col}: nan%={nan_pct:.2f}; "
                f"min={min_v:.6g}; max={max_v:.6g}"
            )

        assert elapsed < _PERF_CEILING_SECONDS, (
            f"GraphFeatureExtractor.fit_transform took {elapsed:.1f}s, "
            f"exceeding the {_PERF_CEILING_SECONDS:.0f}s spec ceiling"
        )

        # Also assert no `inf` snuck through — those would corrupt the
        # downstream LightGBM training.
        for col in feature_names:
            arr = out[col].to_numpy()
            assert not np.any(np.isinf(arr)), f"inf in {col}"
