"""Integration test: shuffled-labels target-leak gate for the FULL T1-T4 pipeline.

Trains a LightGBM on the complete 11-generator pipeline (Tier 1 +
Tier 2 + Tier 3 + Tier 4) with **TRAINING labels SHUFFLED**. If any
of the 11 generators leak training labels into encoded values in a
way that survives the train/val boundary, val AUC inflates above
chance. Spec ceiling: val AUC < 0.55.

Extends 2.3.c's gate (which covered the 10-generator T1-T3 pipeline)
to the FULL T1-T4 pipeline. The Tier-4 generator
(`ExponentialDecayVelocity`) reads training labels at fit time when
`fraud_weighted=True` — same risk class as Sprint-2 `TargetEncoder`.
The pass-1/pass-2 read-before-push discipline makes `fraud_v_ewm`
inherently OOF-safe at the unit level (verified by 3.1.a's
`test_oof_safety_with_fraud_label`); this gate confirms the contract
holds end-to-end through the canonical 11-generator pipeline.

Failure modes this gate catches (in addition to 2.3.c's):

- `ExponentialDecayVelocity.fit_transform` accidentally pushing val
  labels into state (would inflate val EWM under shuffled train).
- `ExponentialDecayVelocity.transform` accidentally pushing val
  labels (per the design, val transform is read-only; a regression
  here would corrupt val features).
- Any system-level interaction where the 11-generator composition
  produces leakage that the unit tests don't catch.

Skip-gate: `MANIFEST.json` presence.
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

pytestmark = pytest.mark.integration

_LEAK_TEST_SAMPLE_SIZE: int = 20_000
_LEAK_TEST_SEED: int = 42

# Spec ceiling. AUC genuinely at chance is ~0.5; we allow up to 0.55
# for finite-sample noise + structural-feature signal that survives
# label shuffling. Genuine target leakage produces AUC >> 0.6, often
# > 0.8 — comfortably caught by this ceiling.
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

    # Shuffle TRAIN labels only. Val labels are untouched — they are
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


def test_shuffled_labels_no_target_leak_full_t4_pipeline(
    shuffled_train_val: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """LightGBM on shuffled-train labels through the FULL T1-T4 pipeline must NOT predict val above chance."""
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
            BehavioralDeviation(),
            ColdStartHandler(),
            ExponentialDecayVelocity(),
            NanGroupReducer(),
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
    # it in the audit report. The success path otherwise hides it.
    print(f"\n[t1-t4 leak-gate] val_auc = {val_auc:.4f}  ceiling = {_LEAK_AUC_CEILING:.4f}")

    assert val_auc < _LEAK_AUC_CEILING, (
        f"TARGET LEAK DETECTED in T1-T4 pipeline. Val AUC = {val_auc:.4f} >= "
        f"{_LEAK_AUC_CEILING}. With training labels shuffled, no feature in "
        f"the 11-generator stack should predict val labels above chance. AUC "
        f"inflation indicates a temporal-safety failure in the new Tier-4 "
        f"generator (`ExponentialDecayVelocity` accidentally pushing val labels "
        f"into state) OR a regression in one of the upstream T1/T2/T3 "
        f"generators. The 2.3.c narrower gate (T1-T3) PASSED at val AUC = "
        f"0.4747; deviating from that here would localise the failure to the "
        f"new Tier-4 surface."
    )
