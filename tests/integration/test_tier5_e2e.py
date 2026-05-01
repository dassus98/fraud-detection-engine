"""End-to-end Tier-5 feature pipeline test against a 10k stratified sample.

Exercises the FULL T1+T2+T3+T4+T5 path: real merged frame → cleaner →
10k stratified sample → 12-generator `FeaturePipeline`. Five checks:

- **Schema validation** against `TierFiveFeaturesSchema` (lazy=True).
  V columns dropped by `NanGroupReducer` pass through inherited
  `strict=False`; the 8 graph-feature columns are enforced.
- **Row counts preserved** — no generator drops rows.
- **All 8 graph columns present** with correct nullability (txn-level
  features 100% NaN on val by design; entity-level features bounded).
- **Soft-warn val-AUC sanity check.** Spec target is 0.93-0.94 on the
  full 414k train; the 10k sample is too noisy to hard-gate at that
  precision, so we `UserWarning` if val AUC < 0.90 and assert only
  the catastrophic floor (val AUC > 0.5). Mirrors the pattern in
  `tests/integration/test_tier4_performance.py`.
- **Shuffled-labels target-leak gate.** Trains LightGBM on the full
  T1-T5 pipeline with TRAINING labels SHUFFLED; asserts val AUC <
  0.55. Catches any system-level interaction where the 12-generator
  composition leaks training labels into encoded values that
  survive the train/val boundary. Extends 3.1.b's T1-T4 leak gate.

Filename note: `test_tier5_e2e.py` (not `test_tier5_performance.py`,
which 3.2.b already shipped as a wall-time-only benchmark — see
`tests/integration/test_tier5_performance.py`). The "e2e" naming
reflects that this test exercises schema + correctness + leak
discipline; runtime is not the gate here.

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
    GraphFeatureExtractor,
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
from fraud_engine.schemas.features import TierFiveFeaturesSchema

pytestmark = pytest.mark.integration

_INTEGRATION_SAMPLE_SIZE: int = 10_000
_INTEGRATION_SEED: int = 42

# 8 graph columns: 1 cc_size + 4 entity_degree + 1 fraud_neighbor_rate
# + 1 pagerank + 1 clustering.
_EXPECTED_GRAPH_COLUMN_COUNT: int = 8

# Train/val split fraction within the 10k sample (sorted by TransactionDT).
_TEMPORAL_SPLIT_FRACTION: float = 0.8

# AUC thresholds for `test_val_auc_sanity_with_soft_warn`. Spec target
# 0.93-0.94 on full data; the 10k sample is too noisy to gate that
# precision, so we soft-warn at 0.90 and hard-fail only at the
# catastrophic-floor 0.5.
_AUC_SOFT_WARN_FLOOR: float = 0.90
_AUC_CATASTROPHIC_FLOOR: float = 0.5

# Spec ceiling for the shuffled-labels leak gate. Genuine target
# leakage produces AUC >> 0.6; 0.55 allows ~5% finite-sample noise.
_LEAK_AUC_CEILING: float = 0.55

_NON_FEATURE_COLS: frozenset[str] = frozenset(
    {"TransactionID", "TransactionDT", "isFraud", "timestamp"}
)


def _manifest_path() -> Path:
    return get_settings().raw_dir / "MANIFEST.json"


@pytest.fixture(scope="module")
def merged_10k_cleaned() -> pd.DataFrame:
    """Load merged + cleaned + 10k stratified sample.

    Module-scoped so the ~30 s load is amortised across all tests.
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
    """Construct the canonical T1+T2+T3+T4+T5 pipeline (12 generators).

    Matches `scripts/build_features_all_tiers.py:_build_pipeline`.
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
            GraphFeatureExtractor(),
            NanGroupReducer(),
        ]
    )


def _select_lgbm_features(df: pd.DataFrame) -> list[str]:
    """Drop non-feature columns + object/string-dtype columns."""
    return [
        col
        for col in df.columns
        if col not in _NON_FEATURE_COLS
        and not pd.api.types.is_object_dtype(df[col])
        and not pd.api.types.is_string_dtype(df[col])
    ]


def _expected_graph_column_names() -> list[str]:
    """The 8 graph column names this pipeline must produce.

    Mirrors `GraphFeatureExtractor.get_feature_names` with the default
    entity-column config.
    """
    return [
        "connected_component_size",
        "entity_degree_card1",
        "entity_degree_addr1",
        "entity_degree_DeviceInfo",
        "entity_degree_P_emaildomain",
        "fraud_neighbor_rate",
        "pagerank_score",
        "clustering_coefficient",
    ]


# ---------------------------------------------------------------------
# Schema + structural correctness.
# ---------------------------------------------------------------------


def test_t5_pipeline_validates_against_schema(
    merged_10k_cleaned: pd.DataFrame,
) -> None:
    """T1-T5 fit_transform output validates against TierFiveFeaturesSchema."""
    sample = merged_10k_cleaned.sort_values("TransactionDT").reset_index(drop=True)
    cut = int(len(sample) * _TEMPORAL_SPLIT_FRACTION)
    train = sample.iloc[:cut].reset_index(drop=True)
    val = sample.iloc[cut:].reset_index(drop=True)

    pipeline = _build_pipeline()
    train_out = pipeline.fit_transform(train)
    val_out = pipeline.transform(val)

    TierFiveFeaturesSchema.validate(train_out, lazy=True)
    TierFiveFeaturesSchema.validate(val_out, lazy=True)


def test_t5_pipeline_preserves_row_counts(
    merged_10k_cleaned: pd.DataFrame,
) -> None:
    """No generator drops rows — train + val outputs equal their inputs."""
    sample = merged_10k_cleaned.sort_values("TransactionDT").reset_index(drop=True)
    cut = int(len(sample) * _TEMPORAL_SPLIT_FRACTION)
    train = sample.iloc[:cut].reset_index(drop=True)
    val = sample.iloc[cut:].reset_index(drop=True)

    pipeline = _build_pipeline()
    train_out = pipeline.fit_transform(train)
    val_out = pipeline.transform(val)

    assert len(train_out) == len(train)
    assert len(val_out) == len(val)


def test_t5_emits_all_8_graph_columns(
    merged_10k_cleaned: pd.DataFrame,
) -> None:
    """All 8 graph columns present; val txn-level features 100% NaN."""
    sample = merged_10k_cleaned.sort_values("TransactionDT").reset_index(drop=True)
    cut = int(len(sample) * _TEMPORAL_SPLIT_FRACTION)
    train = sample.iloc[:cut].reset_index(drop=True)
    val = sample.iloc[cut:].reset_index(drop=True)

    pipeline = _build_pipeline()
    train_out = pipeline.fit_transform(train)
    val_out = pipeline.transform(val)

    expected = _expected_graph_column_names()
    assert len(expected) == _EXPECTED_GRAPH_COLUMN_COUNT

    for col in expected:
        assert col in train_out.columns, f"{col} missing from train_out"
        assert col in val_out.columns, f"{col} missing from val_out"
        # No `inf` values anywhere.
        train_arr = train_out[col].to_numpy()
        val_arr = val_out[col].to_numpy()
        assert not np.any(np.isinf(train_arr)), f"inf in train {col}"
        assert not np.any(np.isinf(val_arr)), f"inf in val {col}"

    # Val txn-level features are 100% NaN by design (val txns are not
    # in the training graph; the temporal-safety contract).
    for col in (
        "connected_component_size",
        "pagerank_score",
        "clustering_coefficient",
    ):
        assert (
            val_out[col].isna().all()
        ), f"val {col} should be 100% NaN; got {val_out[col].isna().mean():.2%}"


# ---------------------------------------------------------------------
# Soft-warn val AUC sanity.
# ---------------------------------------------------------------------


def test_t5_val_auc_sanity_with_soft_warn(
    merged_10k_cleaned: pd.DataFrame,
) -> None:
    """LightGBM on T1-T5 features predicts val above the catastrophic floor.

    The 10k sample is too noisy to hard-gate at the spec's 0.93-0.94
    target. We `UserWarning` if val AUC < 0.90 and assert only the
    catastrophic floor (val AUC > 0.5). The full-data benchmark in
    `scripts/build_features_all_tiers.py` reports the realised
    full-data AUC.
    """
    sample = merged_10k_cleaned.sort_values("TransactionDT").reset_index(drop=True)
    cut = int(len(sample) * _TEMPORAL_SPLIT_FRACTION)
    train = sample.iloc[:cut].reset_index(drop=True)
    val = sample.iloc[cut:].reset_index(drop=True)

    pipeline = _build_pipeline()
    train_out = pipeline.fit_transform(train)
    val_out = pipeline.transform(val)

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

    print(f"\n[t1-t5 e2e] val_auc = {val_auc:.4f}")

    if val_auc < _AUC_SOFT_WARN_FLOOR:
        warnings.warn(
            f"T1-T5 val AUC {val_auc:.4f} below soft-warn floor "
            f"{_AUC_SOFT_WARN_FLOOR:.2f}. Spec target on full data is "
            f"0.93-0.94; the 10k sample is noisy but a regression of "
            f"this magnitude is worth flagging.",
            UserWarning,
            stacklevel=2,
        )

    assert val_auc > _AUC_CATASTROPHIC_FLOOR, (
        f"T1-T5 val AUC {val_auc:.4f} below catastrophic floor "
        f"{_AUC_CATASTROPHIC_FLOOR}; pipeline is broken."
    )


# ---------------------------------------------------------------------
# Shuffled-labels target-leak gate.
# ---------------------------------------------------------------------


def test_t5_shuffled_labels_no_target_leak(
    merged_10k_cleaned: pd.DataFrame,
) -> None:
    """LightGBM on T1-T5 with shuffled-train labels must NOT predict val above chance.

    Extends 3.1.b's T1-T4 leak gate to the 12-generator pipeline.
    `GraphFeatureExtractor` reads training labels at fit time when
    computing OOF `fraud_neighbor_rate` — same risk class as
    `TargetEncoder` and `ExponentialDecayVelocity.fraud_v_ewm`. The
    StratifiedKFold OOF discipline makes it inherently fold-safe at
    the unit level (verified by 3.2.b's `TestOOFContract`); this gate
    confirms the contract holds end-to-end through the full
    12-generator stack.

    AUC ceiling 0.55 — same as 3.1.b's T1-T4 gate.
    """
    sample = merged_10k_cleaned.sort_values("TransactionDT").reset_index(drop=True)
    cut = int(len(sample) * _TEMPORAL_SPLIT_FRACTION)
    train = sample.iloc[:cut].reset_index(drop=True).copy()
    val = sample.iloc[cut:].reset_index(drop=True).copy()

    # Shuffle TRAIN labels only. Val labels are untouched ground truth.
    rng = np.random.default_rng(_INTEGRATION_SEED)
    train["isFraud"] = rng.permutation(train["isFraud"].to_numpy())

    pipeline = _build_pipeline()
    train_out = pipeline.fit_transform(train)
    val_out = pipeline.transform(val)

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

    print(f"\n[t1-t5 leak-gate] val_auc = {val_auc:.4f}  ceiling = {_LEAK_AUC_CEILING:.4f}")

    assert val_auc < _LEAK_AUC_CEILING, (
        f"TARGET LEAK DETECTED in T1-T5 pipeline. Val AUC = {val_auc:.4f} >= "
        f"{_LEAK_AUC_CEILING}. With training labels shuffled, no feature in "
        f"the 12-generator stack should predict val labels above chance. AUC "
        f"inflation indicates a temporal-safety failure in the new Tier-5 "
        f"generator (`GraphFeatureExtractor` accidentally pushing val labels "
        f"into state) OR a regression in one of the upstream T1/T2/T3/T4 "
        f"generators."
    )
