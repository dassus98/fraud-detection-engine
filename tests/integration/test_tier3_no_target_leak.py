"""Integration test: shuffled-labels target-leak gate for the FULL T1-T3 pipeline.

Trains a LightGBM on the complete 10-generator pipeline (Tier 1 +
Tier 2 + Tier 3) with **TRAINING labels SHUFFLED**. If any of the 10
generators leak training labels into encoded values in a way that
survives the train/val boundary, val AUC inflates above chance. Spec
ceiling: val AUC < 0.55.

Extends 2.2.d's gate (which covered the 7-generator T1+T2+TargetEncoder
pipeline) to the FULL T1-T3 pipeline. The Tier-3 generators
(`BehavioralDeviation`, `ColdStartHandler`, `NanGroupReducer`) don't
read val labels by construction; this test gates that contract end-to-
end through the canonical pipeline. Together with 2.2.d's narrower
gate, this confirms zero target leakage anywhere in the Sprint-2
feature surface.

Failure modes this gate catches (in addition to 2.2.d's):
- `BehavioralDeviation` accidentally aggregates val rows into card
  history (would inflate val z-scores under shuffled train).
- `ColdStartHandler` accidentally counts val rows in past_count
  (would underflag val rows as warm).
- `NanGroupReducer` accidentally fits its kept-list on (train + val)
  (would let val rows' V values inform their own retention).

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


def test_shuffled_labels_no_target_leak_full_pipeline(
    shuffled_train_val: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """LightGBM on shuffled-train labels through the FULL T1-T3 pipeline must NOT predict val above chance."""
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
    print(f"\n[t1-t3 leak-gate] val_auc = {val_auc:.4f}  ceiling = {_LEAK_AUC_CEILING:.4f}")

    assert val_auc < _LEAK_AUC_CEILING, (
        f"TARGET LEAK DETECTED in T1-T3 pipeline. Val AUC = {val_auc:.4f} >= "
        f"{_LEAK_AUC_CEILING}. With training labels shuffled, no feature in "
        f"the 10-generator stack should predict val labels above chance. AUC "
        f"inflation indicates a temporal-safety failure in one of the Tier-3 "
        f"generators (BehavioralDeviation, ColdStartHandler, NanGroupReducer) "
        f"OR a regression in one of the upstream T1/T2 generators. The 2.2.d "
        f"narrower gate (T1+T2+TargetEncoder) PASSED at val AUC = 0.4943; "
        f"deviating from that here would localise the failure to the new "
        f"Tier-3 surface."
    )
