"""End-to-end Tier-3 feature pipeline test against a 10k stratified sample.

Exercises the full path: real merged frame → cleaner → 10k stratified
sample → Tier-1 + Tier-2 + Tier-3 `FeaturePipeline.fit_transform`.
Two shape checks:

- Output validates against `TierThreeFeaturesSchema` (lazy=True). The
  V columns dropped by `NanGroupReducer` pass through inherited
  `strict=False`; only the 6 deterministic Tier-3 columns are
  enforced.
- Row counts preserved (the Tier-3 generators add columns or drop
  V columns; they never drop rows).

Mirrors `test_tier2_e2e.py`'s `merged_10k_cleaned` fixture pattern —
skip-gated on `data/raw/MANIFEST.json` so bootstrap-only CI keeps
passing.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from fraud_engine.config.settings import get_settings
from fraud_engine.data.cleaner import TransactionCleaner
from fraud_engine.data.loader import RawDataLoader
from fraud_engine.features import (
    BehavioralDeviation,
    ColdStartHandler,
    FeaturePipeline,
    HistoricalStats,
    NanGroupReducer,
    TargetEncoder,
    VelocityCounter,
)
from fraud_engine.features.tier1_basic import (
    AmountTransformer,
    EmailDomainExtractor,
    MissingIndicatorGenerator,
    TimeFeatureGenerator,
)
from fraud_engine.schemas.features import TierThreeFeaturesSchema

pytestmark = pytest.mark.integration

_INTEGRATION_SAMPLE_SIZE: int = 10_000
_INTEGRATION_SEED: int = 42


def _manifest_path() -> Path:
    return get_settings().raw_dir / "MANIFEST.json"


@pytest.fixture(scope="module")
def merged_10k_cleaned() -> pd.DataFrame:
    """Load merged + cleaned + 10k stratified sample.

    Module-scoped so the ~30 s load is amortised across both tests.
    """
    if not _manifest_path().is_file():
        pytest.skip("data/raw/MANIFEST.json not present — run `make data-download`.")
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


def _build_pipeline() -> FeaturePipeline:
    """Construct the canonical Tier-1 + Tier-2 + Tier-3 pipeline."""
    return FeaturePipeline(
        generators=[
            AmountTransformer(),
            TimeFeatureGenerator(),
            EmailDomainExtractor(),
            MissingIndicatorGenerator(),
            VelocityCounter(),
            HistoricalStats(),
            TargetEncoder(),
            BehavioralDeviation(),
            ColdStartHandler(),
            NanGroupReducer(),
        ]
    )


def test_pipeline_fit_transform_validates_against_schema(
    merged_10k_cleaned: pd.DataFrame,
) -> None:
    """End-to-end: fit on the 10k sample; output validates against TierThreeFeaturesSchema."""
    pipeline = _build_pipeline()
    out = pipeline.fit_transform(merged_10k_cleaned)
    TierThreeFeaturesSchema.validate(out, lazy=True)


def test_pipeline_preserves_row_counts(merged_10k_cleaned: pd.DataFrame) -> None:
    """Tier-1 + Tier-2 + Tier-3 generators add/remove columns but never drop rows."""
    pipeline = _build_pipeline()
    out = pipeline.fit_transform(merged_10k_cleaned)
    assert len(out) == len(merged_10k_cleaned)
