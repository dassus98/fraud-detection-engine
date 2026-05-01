"""End-to-end Tier-4 feature pipeline test against a 10k stratified sample.

Exercises the full path: real merged frame → cleaner → 10k stratified
sample → 11-generator `FeaturePipeline` (T1+T2+T3+T4). Four checks:

- Output validates against `TierFourFeaturesSchema` (lazy=True). The
  V columns dropped by `NanGroupReducer` pass through inherited
  `strict=False`; the 24 EWM columns are enforced.
- Row counts preserved.
- All 24 EWM columns present, finite, and non-negative (catches
  `inf` which `Check.greater_than_or_equal_to` doesn't reject).
- **Soft-warn AUC sanity check.** The 10k sample is too noisy to
  hard-gate at the spec's 0.92-0.93 target; instead, print a
  `UserWarning` if val AUC < 0.90 and assert only the catastrophic
  floor (val AUC > 0.5). The build script reports the realised
  full-data AUC.

Filename note: spec calls this `test_tier4_performance.py` (not the
`test_tier4_e2e.py` shape Sprint 2 used). The "performance" naming
reflects the AUC sanity check; structural checks (schema validation,
row preservation) are also covered here.

Mirrors `test_tier3_e2e.py`'s `merged_10k_cleaned` fixture pattern —
skip-gated on `data/raw/MANIFEST.json` so bootstrap-only CI keeps
passing.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from fraud_engine.config.settings import get_settings
from fraud_engine.data.cleaner import TransactionCleaner
from fraud_engine.data.loader import RawDataLoader
from fraud_engine.features import (
    BehavioralDeviation,
    ColdStartHandler,
    ExponentialDecayVelocity,
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
from fraud_engine.schemas.features import TierFourFeaturesSchema

pytestmark = pytest.mark.integration

_INTEGRATION_SAMPLE_SIZE: int = 10_000
_INTEGRATION_SEED: int = 42

# 4 entities × 3 lambdas × 2 signals = 24.
_EXPECTED_EWM_COLUMN_COUNT: int = 24

# AUC thresholds for `test_val_auc_sanity_with_soft_warn`. Spec target
# is 0.92-0.93 on full data; the 10k sample is too noisy to gate at
# that precision (see test docstring).
_AUC_SOFT_WARN_FLOOR: float = 0.90
_AUC_CATASTROPHIC_FLOOR: float = 0.5

# Train/val split fraction within the 10k sample, sorted by TransactionDT.
_TEMPORAL_SPLIT_FRACTION: float = 0.8

_NON_FEATURE_COLS: frozenset[str] = frozenset(
    {"TransactionID", "TransactionDT", "isFraud", "timestamp"}
)


def _manifest_path() -> Path:
    return get_settings().raw_dir / "MANIFEST.json"


@pytest.fixture(scope="module")
def merged_10k_cleaned() -> pd.DataFrame:
    """Load merged + cleaned + 10k stratified sample.

    Module-scoped so the ~30 s load is amortised across all four tests.
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
    """Construct the canonical Tier-1+2+3+4 pipeline (11 generators).

    Order matters: ExponentialDecayVelocity is placed at position 10
    (between ColdStartHandler and NanGroupReducer). NanGroupReducer
    must stay last per its class docstring — it removes V columns;
    no downstream stage may reference them.
    """
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
            ExponentialDecayVelocity(),
            NanGroupReducer(),
        ]
    )


def _select_lgbm_features(df: pd.DataFrame) -> list[str]:
    """Return the subset of columns LightGBM's sklearn API can ingest.

    Mirrors `scripts/build_features_all_tiers.py:_select_lgbm_features`.
    Drops non-feature columns and any object/string-dtype columns
    (provider/tld would need explicit categorical-feature enumeration).
    """
    return [
        col
        for col in df.columns
        if col not in _NON_FEATURE_COLS
        and not pd.api.types.is_object_dtype(df[col])
        and not pd.api.types.is_string_dtype(df[col])
    ]


def _expected_ewm_column_names() -> set[str]:
    """The 24 EWM column names this pipeline must produce.

    Mirrors `tier4_decay._LAMBDA_FORMAT_SPEC` ("g") and the default
    YAML's entities + lambdas. If any of those defaults change, this
    helper (and the schema) must be updated together.
    """
    entities = ("card1", "addr1", "DeviceInfo", "P_emaildomain")
    lambdas = (0.05, 0.1, 0.5)
    return {
        f"{entity}_{suffix}_lambda_{lam:g}"
        for entity in entities
        for lam in lambdas
        for suffix in ("v_ewm", "fraud_v_ewm")
    }


def test_pipeline_fit_transform_validates_against_schema(
    merged_10k_cleaned: pd.DataFrame,
) -> None:
    """End-to-end: fit on 10k sample; output validates against TierFourFeaturesSchema."""
    pipeline = _build_pipeline()
    out = pipeline.fit_transform(merged_10k_cleaned)
    TierFourFeaturesSchema.validate(out, lazy=True)


def test_pipeline_preserves_row_counts(merged_10k_cleaned: pd.DataFrame) -> None:
    """Tier-1 + Tier-2 + Tier-3 + Tier-4 generators add/remove columns but never drop rows."""
    pipeline = _build_pipeline()
    out = pipeline.fit_transform(merged_10k_cleaned)
    assert len(out) == len(merged_10k_cleaned)


def test_all_24_ewm_columns_present_and_finite(merged_10k_cleaned: pd.DataFrame) -> None:
    """The 24 EWM columns are present, finite, and non-negative.

    Goes beyond `Check.greater_than_or_equal_to` (which only enforces
    the lower bound): also rejects `inf`. Catches a hypothetical bug
    where decay arithmetic produces +inf (e.g., division-by-zero in
    a future variant of the formula).
    """
    pipeline = _build_pipeline()
    out = pipeline.fit_transform(merged_10k_cleaned)
    expected = _expected_ewm_column_names()
    assert len(expected) == _EXPECTED_EWM_COLUMN_COUNT
    for col in expected:
        assert col in out.columns, f"Expected EWM column {col} missing from output"
        values = out[col].to_numpy()
        assert np.isfinite(values).all(), f"Non-finite values in {col}"
        assert (values >= 0.0).all(), f"Negative values in {col}"


def test_val_auc_sanity_with_soft_warn(merged_10k_cleaned: pd.DataFrame) -> None:
    """Sanity-checks the 10k-sample val AUC; soft-warn below 0.90, hard-fail below 0.5.

    The 10k stratified sample is too noisy to gate at the spec's
    0.92-0.93 target — that range is for full data and is reported
    by the build script. A soft warning surfaces low AUC in CI logs
    without producing flaky failures; the hard floor (0.5) catches
    catastrophic pipeline regressions (below random chance).
    """
    sorted_df = merged_10k_cleaned.sort_values("TransactionDT").reset_index(drop=True)
    n_train = int(len(sorted_df) * _TEMPORAL_SPLIT_FRACTION)
    train_df = sorted_df.iloc[:n_train].copy()
    val_df = sorted_df.iloc[n_train:].copy()

    pipeline = _build_pipeline()
    train_out = pipeline.fit_transform(train_df)
    val_out = pipeline.transform(val_df)

    feature_cols = _select_lgbm_features(train_out)
    settings = get_settings()
    clf = LGBMClassifier(
        **settings.lgbm_defaults,
        random_state=settings.seed,
        verbose=-1,
    )
    clf.fit(
        train_out[feature_cols],
        train_out["isFraud"],
        categorical_feature="auto",
    )
    val_proba = clf.predict_proba(val_out[feature_cols])[:, 1]
    val_auc = float(roc_auc_score(val_out["isFraud"], val_proba))

    print(f"\n[tier4-perf] val_auc on 10k sample = {val_auc:.4f}")

    if val_auc < _AUC_SOFT_WARN_FLOOR:
        warnings.warn(
            f"Val AUC {val_auc:.4f} on 10k sample is below the {_AUC_SOFT_WARN_FLOOR} "
            f"sanity floor. Spec target is 0.92-0.93 on full data; check the build "
            f"script's output for the realised full-data AUC.",
            UserWarning,
            stacklevel=2,
        )
    assert val_auc > _AUC_CATASTROPHIC_FLOOR, (
        f"Catastrophic regression: val AUC {val_auc:.4f} <= {_AUC_CATASTROPHIC_FLOOR} "
        f"(below random chance). This indicates a fundamental pipeline bug."
    )
