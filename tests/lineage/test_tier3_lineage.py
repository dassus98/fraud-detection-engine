"""Tier-3 temporal-integrity gate.

Runs `assert_no_future_leak` for every Tier-3 deterministic column on
50 random rows of the val output. Any single failure fails the test.

The Tier-3 generators that ADD columns are `BehavioralDeviation` (5
cols: amt z-score, time z-score, addr/device change flags, hour
deviation) and `ColdStartHandler` (1 col: is_coldstart_card1).
`NanGroupReducer` is excluded — it removes columns rather than adding
them, so there's nothing to leak-check (its kept V columns are pre-
existing inputs already covered by 2.2.e's Tier-2 lineage walk).

Mechanism mirrors `tests/lineage/test_tier2_temporal_integrity.py`:
fit pipeline on train, transform val. For each Tier-3 column,
build a recompute lambda using the FITTED generator's
`transform(slice_df)` and pass it to `assert_no_future_leak`. For
the row at the slice's max timestamp, the recompute MUST equal the
value `transform(val)` produced for that row. Any deviation is a
temporal-leak bug.

`source_df` is the full pipeline output (`val_out`) rather than the
cleaner-output `splits.val`. Reason: Tier-3 generators depend on
upstream columns added by Tier-1 — `BehavioralDeviation` reads
`hour_of_day` (produced by `TimeFeatureGenerator`); the cleaner
output doesn't have it, so passing `splits.val` would fire a
`KeyError` inside the recompute lambda. Passing `val_out` gives the
recompute the full upstream context the Tier-3 generators need.
This differs from 2.2.e's Tier-2 leak walk (which used
`splits.val` because Tier-2 generators read only cleaner-output
columns) and is intentional.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
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
from fraud_engine.features.base import BaseFeatureGenerator
from fraud_engine.features.temporal_guards import assert_no_future_leak
from fraud_engine.features.tier1_basic import (
    AmountTransformer,
    EmailDomainExtractor,
    MissingIndicatorGenerator,
    TimeFeatureGenerator,
)

pytestmark = pytest.mark.lineage

_LINEAGE_SAMPLE_SIZE: int = 10_000
_LINEAGE_SEED: int = 42
_N_SAMPLES_PER_FEATURE: int = 50

# Tier-3 generators whose feature columns get the leak walk. Excludes
# `NanGroupReducer` because it removes columns rather than adding them.
_TIER3_GENERATOR_TYPES: tuple[type, ...] = (
    BehavioralDeviation,
    ColdStartHandler,
)


def _manifest_path() -> Path:
    return get_settings().raw_dir / "MANIFEST.json"


@pytest.fixture(scope="module")
def fitted_pipeline_and_val() -> tuple[FeaturePipeline, pd.DataFrame]:
    """Fit a full T1+T2+T3 pipeline on train; return pipeline + val output.

    The 10k stratified sample is `temporal_split`-ed using the project
    defaults; the pipeline is fit on `splits.train`; the val output is
    used for BOTH `feature_df` and `source_df` in the leak walk
    (Tier-3 generators depend on Tier-1-augmented columns like
    `hour_of_day`, which the cleaner-output `splits.val` doesn't
    have — see module docstring).
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
            VelocityCounter(),
            HistoricalStats(),
            TargetEncoder(),
            BehavioralDeviation(),
            ColdStartHandler(),
            NanGroupReducer(),
        ]
    )
    pipeline.fit_transform(splits.train)
    val_out = pipeline.transform(splits.val)
    return pipeline, val_out


def _make_recompute(
    gen: BaseFeatureGenerator, col_name: str
) -> Callable[[pd.DataFrame], pd.Series[Any]]:
    """Build a recompute lambda that runs the FITTED generator on a slice."""

    def recompute(slice_df: pd.DataFrame) -> pd.Series[Any]:
        out = gen.transform(slice_df)
        series = out[col_name]
        series.name = col_name
        return series

    return recompute


def test_assert_no_future_leak_on_all_tier3_features(
    fitted_pipeline_and_val: tuple[FeaturePipeline, pd.DataFrame],
) -> None:
    """Every Tier-3 deterministic feature passes `assert_no_future_leak` on val.

    Sweeps `BehavioralDeviation.get_feature_names()` (5 cols) and
    `ColdStartHandler.get_feature_names()` (1 col). For each column,
    runs `assert_no_future_leak` against the val output with
    `n_samples = 50`. Failures are accumulated into a single error
    message so all leaking columns are reported together.
    """
    pipeline, val_out = fitted_pipeline_and_val

    failures: list[str] = []
    n_features_checked = 0

    for gen in pipeline.generators:
        if not isinstance(gen, _TIER3_GENERATOR_TYPES):
            continue
        for col_name in gen.get_feature_names():
            n_features_checked += 1
            try:
                assert_no_future_leak(
                    feature_df=val_out,
                    source_df=val_out,  # see module docstring for rationale
                    feature_func=_make_recompute(gen, col_name),
                    n_samples=_N_SAMPLES_PER_FEATURE,
                )
            except AssertionError as exc:
                failures.append(f"{type(gen).__name__}.{col_name}: {exc}")

    # Sanity check: 5 BehavioralDeviation + 1 ColdStartHandler = 6.
    expected_n_features = 6
    assert n_features_checked == expected_n_features, (
        f"Expected to check {expected_n_features} Tier-3 features; "
        f"checked {n_features_checked}. Has the default config changed?"
    )

    if failures:
        joined = "\n  - ".join(failures)
        raise AssertionError(
            f"Temporal leak detected in {len(failures)}/"
            f"{n_features_checked} Tier-3 features:\n  - {joined}"
        )
