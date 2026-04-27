"""Tier-1 lineage contract tests.

Three invariants:

- Pipeline fit on train applies cleanly to val (no schema-drift /
  missing-column errors).
- Non-nullable Tier-1 columns (log_amount, amount_decile,
  hour_of_day, is_business_hours, hour_sin, hour_cos) have zero
  NaN in the val output.
- The pipeline-emitted `feature_manifest.json` carries
  `schema_version: 1`, matching `FEATURE_SCHEMA_VERSION`.

Skip-gated on `data/raw/MANIFEST.json` like every other lineage test
that loads real merged data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from fraud_engine.config.settings import get_settings
from fraud_engine.data.cleaner import TransactionCleaner
from fraud_engine.data.loader import RawDataLoader
from fraud_engine.data.splits import temporal_split
from fraud_engine.features import FeaturePipeline
from fraud_engine.features.tier1_basic import (
    AmountTransformer,
    EmailDomainExtractor,
    MissingIndicatorGenerator,
    TimeFeatureGenerator,
)
from fraud_engine.schemas.features import (
    FEATURE_SCHEMA_VERSION,
    TierOneFeaturesSchema,
)

pytestmark = pytest.mark.lineage

_LINEAGE_SAMPLE_SIZE: int = 10_000
_LINEAGE_SEED: int = 42


def _manifest_path() -> Path:
    return get_settings().raw_dir / "MANIFEST.json"


@pytest.fixture(scope="module")
def fitted_pipeline_and_splits() -> tuple[FeaturePipeline, dict[str, pd.DataFrame]]:
    """Fit on train; return pipeline + transformed train/val/test.

    Loads the real merged frame, cleans it, takes a 10k stratified
    sample, then `temporal_split`s using the project's default
    `Settings.train_end_dt` / `val_end_dt`. Pipeline is fit on train
    only; val and test are transformed with the fitted state.
    """
    if not _manifest_path().is_file():
        pytest.skip("data/raw/MANIFEST.json not present — run `make data-download`.")
    settings = get_settings()
    loader = RawDataLoader()
    full = loader.load_merged(optimize=False)
    cleaned = TransactionCleaner().clean(full)
    sample, _ = train_test_split(
        cleaned,
        train_size=_LINEAGE_SAMPLE_SIZE,
        stratify=cleaned["isFraud"],
        random_state=_LINEAGE_SEED,
    )
    sample = sample.reset_index(drop=True)
    splits = temporal_split(sample, settings=settings)

    pipeline = FeaturePipeline(
        generators=[
            AmountTransformer(),
            TimeFeatureGenerator(),
            EmailDomainExtractor(),
            MissingIndicatorGenerator(),
        ]
    )
    train_out = pipeline.fit_transform(splits.train)
    val_out = pipeline.transform(splits.val)
    test_out = pipeline.transform(splits.test)
    return pipeline, {"train": train_out, "val": val_out, "test": test_out}


def test_pipeline_train_to_val_no_errors(
    fitted_pipeline_and_splits: tuple[FeaturePipeline, dict[str, pd.DataFrame]],
) -> None:
    """Pipeline fit on train applies cleanly to val (no schema-drift errors).

    The strongest validation we have for "fit on train, transform val"
    is that `TierOneFeaturesSchema` accepts the val output. Catches:
    column drift (a generator produces different columns at val
    time), dtype drift (cleaner output dtype shifts somehow), range
    drift (a Tier-1 column lands outside its declared bounds).
    """
    _, splits = fitted_pipeline_and_splits
    TierOneFeaturesSchema.validate(splits["val"], lazy=True)


def test_no_nan_in_non_nullable_cols(
    fitted_pipeline_and_splits: tuple[FeaturePipeline, dict[str, pd.DataFrame]],
) -> None:
    """Tier-1 non-nullable cols have zero NaN in the val output.

    The schema declares these columns `nullable=False`. A NaN in any
    of them means a generator silently corrupted its output; this
    test catches that before downstream models see the corrupted
    feature.
    """
    _, splits = fitted_pipeline_and_splits
    val = splits["val"]
    non_nullable_cols = [
        "log_amount",
        "amount_decile",
        "hour_of_day",
        "is_business_hours",
        "hour_sin",
        "hour_cos",
    ]
    for col in non_nullable_cols:
        assert val[col].notna().all(), f"NaN found in non-nullable column {col}"


def test_manifest_schema_version_matches(
    fitted_pipeline_and_splits: tuple[FeaturePipeline, dict[str, pd.DataFrame]],
    tmp_path: Path,
) -> None:
    """`feature_manifest.json` `schema_version` equals `FEATURE_SCHEMA_VERSION`.

    The manifest carries the *manifest file* schema version (from
    2.1.a), which currently coincides with `FEATURE_SCHEMA_VERSION`
    (both 1). If they diverge in a future sprint, this test surfaces
    the mismatch and one of the two constants needs to bump.
    """
    pipeline, _ = fitted_pipeline_and_splits
    pipeline.save(tmp_path)
    manifest = json.loads((tmp_path / "feature_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == FEATURE_SCHEMA_VERSION
