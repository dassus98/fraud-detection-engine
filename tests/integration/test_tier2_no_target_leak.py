"""Integration test: shuffled-labels target-leak gate for Tier-2 features.

Trains a LightGBM on Tier 1 + Tier 2 + `TargetEncoder` features
with **TRAINING labels SHUFFLED**. If any feature leaks training
labels into encoded values in a way that survives the train/val
boundary, val AUC inflates above chance. Spec ceiling: val AUC < 0.55.

Mechanism:
    With shuffled training labels, no genuine signal connects features
    to val's TRUE labels. If `TargetEncoder` is correctly OOF on
    training and fit on TRAIN only for val/test, val rows' encoded
    values reflect shuffled-train statistics — uncorrelated with
    val's true labels. Val AUC ≈ 0.5.

    The failure mode this gate catches: encoder accidentally fit on
    (train + val), so val rows' encoded values reflect their TRUE
    labels. Even with shuffled train labels, val predictions become
    unnaturally accurate. AUC inflates to 0.6+ (often 0.8+).

Skip-gate: `MANIFEST.json` presence — same convention as every
other real-data integration test in the project.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from fraud_engine.config.settings import get_settings
from fraud_engine.data.cleaner import TransactionCleaner
from fraud_engine.data.loader import RawDataLoader
from fraud_engine.data.splits import temporal_split
from fraud_engine.features import (
    FeaturePipeline,
    HistoricalStats,
    TargetEncoder,
    VelocityCounter,
)
from fraud_engine.features.tier1_basic import (
    AmountTransformer,
    EmailDomainExtractor,
    MissingIndicatorGenerator,
    TimeFeatureGenerator,
)

pytestmark = pytest.mark.integration

_LEAK_TEST_SAMPLE_SIZE: int = 20_000
_LEAK_TEST_SEED: int = 42

# Spec ceiling. AUC genuinely at chance is ~0.5; we allow up to 0.55
# for finite-sample noise + structural-feature signal that survives
# label shuffling (e.g. velocity counts that correlate with hour-of-day
# rhythms unrelated to fraud). Genuine target leakage produces AUC
# >> 0.6, often > 0.8 — comfortably caught by this ceiling.
_LEAK_AUC_CEILING: float = 0.55

_NON_FEATURE_COLS: frozenset[str] = frozenset(
    {"TransactionID", "TransactionDT", "isFraud", "timestamp"}
)


@pytest.fixture(scope="module")
def shuffled_train_val() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load + clean + sample + temporal-split; SHUFFLE TRAIN labels only."""
    if not (get_settings().raw_dir / "MANIFEST.json").is_file():
        pytest.skip("data/raw/MANIFEST.json not present — run `make data-download`.")
    settings = get_settings()
    full = RawDataLoader().load_merged(optimize=False)
    cleaned = TransactionCleaner().clean(full)
    sample, _ = train_test_split(
        cleaned,
        train_size=_LEAK_TEST_SAMPLE_SIZE,
        stratify=cleaned["isFraud"],
        random_state=_LEAK_TEST_SEED,
    )
    sample = sample.reset_index(drop=True)
    splits = temporal_split(sample, settings=settings)
    train = splits.train.copy()
    val = splits.val.copy()

    # Shuffle TRAIN labels only. Val labels are untouched — those are
    # the ground truth we measure AUC against.
    rng = np.random.default_rng(_LEAK_TEST_SEED)
    train["isFraud"] = rng.permutation(train["isFraud"].to_numpy())

    return train, val


def _select_lgbm_features(df: pd.DataFrame) -> list[str]:
    """Drop non-feature columns + object/string-dtype columns the sklearn LightGBM API can't ingest."""
    return [
        col
        for col in df.columns
        if col not in _NON_FEATURE_COLS
        and not pd.api.types.is_object_dtype(df[col])
        and not pd.api.types.is_string_dtype(df[col])
    ]


def test_shuffled_labels_no_target_leak(
    shuffled_train_val: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """LightGBM on shuffled-train labels must NOT predict val above chance."""
    train, val = shuffled_train_val

    pipeline = FeaturePipeline(
        generators=[
            AmountTransformer(),
            TimeFeatureGenerator(),
            EmailDomainExtractor(),
            MissingIndicatorGenerator(),
            VelocityCounter(),
            HistoricalStats(),
            TargetEncoder(),
        ]
    )

    train_features = pipeline.fit_transform(train)
    val_features = pipeline.transform(val)

    feature_cols = _select_lgbm_features(train_features)
    settings = get_settings()
    clf = LGBMClassifier(
        **settings.lgbm_defaults,
        random_state=settings.seed,
        verbose=-1,
    )
    clf.fit(
        train_features[feature_cols],
        train_features["isFraud"],
        categorical_feature="auto",
    )

    val_probs = clf.predict_proba(val_features[feature_cols])[:, 1]
    val_auc = float(roc_auc_score(val_features["isFraud"], val_probs))

    # Echo the realised val AUC to stdout so a `pytest -s` run records
    # it in the completion report. The success path otherwise hides it.
    print(f"\n[leak-gate] val_auc = {val_auc:.4f}  ceiling = {_LEAK_AUC_CEILING:.4f}")

    assert val_auc < _LEAK_AUC_CEILING, (
        f"TARGET LEAK DETECTED. Val AUC = {val_auc:.4f} >= {_LEAK_AUC_CEILING}. "
        f"With training labels shuffled, no feature should predict val labels "
        f"above chance. AUC inflation indicates that target-encoded features "
        f"carry leaked label information into val — typically because the "
        f"encoder was fit on (train + val) instead of train-only, OR because "
        f"OOF discipline is broken within training so the encoder leaks "
        f"shuffled-label structure that survives the temporal boundary."
    )
