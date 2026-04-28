"""Tier-2 temporal-integrity gate.

Runs `assert_no_future_leak` for **every Tier-2 feature column** on
50 random rows of the val output. Any single failure fails the test.

Mechanism:
    For each Tier-2 generator (`VelocityCounter`, `HistoricalStats`,
    `TargetEncoder`), the FITTED generator's `transform(slice_df)` is
    used as the recompute lambda.

    - `VelocityCounter` and `HistoricalStats` are stateless (`fit` is
      a no-op): `transform` rebuilds per-entity deque state from the
      passed slice. For a val row R at timestamp T, transforming the
      strictly-past slice (`source[ts <= T]`) and reading the column
      at R's index MUST equal the value `transform(val)` produced.
      Any deviation is a temporal-leak bug.
    - `TargetEncoder` applies a frozen full-train encoder; recomputing
      it on any slice yields the same encoded value for the same
      category. The leak walk passes trivially — but including it is
      a useful regression detector: if someone ever breaks the
      encoder so it's not a frozen lookup (e.g. by re-fitting on
      slice data), the walk would catch the change.

The leak walk is run on the val output, not train. On train rows,
`TargetEncoder`'s OOF encoding mixes future training rows into the
"OTHER folds" by random-stratified KFold (deliberately — see 2.2.d's
trade-off note); applying `assert_no_future_leak` to OOF train rows
would fire on every row. The val path is the natural target: every
Tier-2 feature on a val row is a function of data temporally `<=`
that row, by construction.
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
    FeaturePipeline,
    HistoricalStats,
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

# The three Tier-2 generator types whose feature columns get the
# `assert_no_future_leak` walk. Tier-1 generators are static
# transformations of the same row (no temporal lookback) so are not
# included.
_TIER2_GENERATOR_TYPES: tuple[type, ...] = (
    VelocityCounter,
    HistoricalStats,
    TargetEncoder,
)


def _manifest_path() -> Path:
    return get_settings().raw_dir / "MANIFEST.json"


@pytest.fixture(scope="module")
def fitted_pipeline_and_val() -> tuple[FeaturePipeline, pd.DataFrame, pd.DataFrame]:
    """Fit a full T1+T2 pipeline on train; return pipeline + val output + cleaner-output val.

    The 10 k stratified sample is then `temporal_split`-ed using
    project defaults; the pipeline is fit on `splits.train` (so
    `TargetEncoder.mappings_` is populated against train labels);
    the val output is the test target.
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
        ]
    )
    pipeline.fit_transform(splits.train)
    val_out = pipeline.transform(splits.val)
    return pipeline, val_out, splits.val


def _make_recompute(
    gen: BaseFeatureGenerator, col_name: str
) -> Callable[[pd.DataFrame], pd.Series[Any]]:
    """Build a recompute lambda that runs the FITTED generator on a slice.

    `assert_no_future_leak` calls this with the strictly-past slice;
    the return must be a Series with `.name == col_name`.
    """

    def recompute(slice_df: pd.DataFrame) -> pd.Series[Any]:
        out = gen.transform(slice_df)
        series = out[col_name]
        series.name = col_name
        return series

    return recompute


def test_assert_no_future_leak_on_all_tier2_features(
    fitted_pipeline_and_val: tuple[FeaturePipeline, pd.DataFrame, pd.DataFrame],
) -> None:
    """Every Tier-2 feature passes `assert_no_future_leak` on val.

    Sweeps every Tier-2 generator's `get_feature_names()`; for each
    column, runs `assert_no_future_leak` against the val output with
    `n_samples = 50`. Failures are accumulated into a single error
    message so all leaking columns are reported together.
    """
    pipeline, val_out, val_source = fitted_pipeline_and_val

    failures: list[str] = []
    n_features_checked = 0

    for gen in pipeline.generators:
        if not isinstance(gen, _TIER2_GENERATOR_TYPES):
            continue
        for col_name in gen.get_feature_names():
            n_features_checked += 1
            try:
                assert_no_future_leak(
                    feature_df=val_out,
                    source_df=val_source,
                    feature_func=_make_recompute(gen, col_name),
                    n_samples=_N_SAMPLES_PER_FEATURE,
                )
            except AssertionError as exc:
                failures.append(f"{type(gen).__name__}.{col_name}: {exc}")

    # Sanity check: we should have exercised every Tier-2 feature.
    # 12 (VelocityCounter) + 5 (HistoricalStats) + 3 (TargetEncoder) = 20.
    expected_n_features = 20
    assert n_features_checked == expected_n_features, (
        f"Expected to check {expected_n_features} Tier-2 features; "
        f"checked {n_features_checked}. Has the default config changed?"
    )

    if failures:
        joined = "\n  - ".join(failures)
        raise AssertionError(
            f"Temporal leak detected in {len(failures)}/"
            f"{n_features_checked} Tier-2 features:\n  - {joined}"
        )
