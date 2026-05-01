"""Tier-5 temporal-integrity gate.

Runs `assert_no_future_leak` for every Tier-5 deterministic column on
50 random rows of the val output. Any single failure fails the test.

The Tier-5 generator that ADDS columns is `GraphFeatureExtractor` (8
cols: cc_size, 4 entity_degree_*, fraud_neighbor_rate, pagerank_score,
clustering_coefficient). Mechanism mirrors
`tests/lineage/test_tier3_lineage.py` and `test_tier4_lineage.py` (if
present): fit pipeline on train, transform val. For each Tier-5
column, build a recompute lambda using the FITTED generator's
`transform(slice_df)` and pass it to `assert_no_future_leak`.

Tier-5 has a slightly different temporal-safety story than Tier-2/3/4.
The graph is fitted ONCE on the training frame; `transform(slice_df)`
on any slice queries the SAME frozen lookups (`entity_degree_`,
`entity_fraud_sum_`, `entity_total_count_`). The slice's contents
affect only WHICH rows the function returns features for, not WHAT
features each row gets. So:

- Train-time leak risk: the generator's `fit_transform` runs OOF
  StratifiedKFold for `fraud_neighbor_rate` (per-fold rebuild +
  walk); each oof row's rate is computed from the OTHER folds.
  Self-leakage is impossible by construction.
- Inference (transform) leak risk: query-only against frozen state.
  Trivially leak-free.

For val rows, txn-level features (CC size, pagerank, clustering)
emit NaN — the val txn isn't in the training graph by temporal-
safety contract. NaN-NaN comparison passes through `_values_match`
as a match (both genuinely missing), so `assert_no_future_leak`
trivially holds for these columns.

`source_df` is the full pipeline output (`val_out`) rather than the
cleaner-output `splits.val`. Reason mirrors Tier-3: Tier-5 generators
depend on upstream Tier-1 + Tier-2 columns (entity columns are
cleaner-output, but the recompute lambda must run through the fitted
generator instance which expects the full schema).
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
    ExponentialDecayVelocity,
    FeaturePipeline,
    GraphFeatureExtractor,
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

# Tier-5 generators whose feature columns get the leak walk.
_TIER5_GENERATOR_TYPES: tuple[type, ...] = (GraphFeatureExtractor,)

# Expected number of Tier-5 features (1 cc_size + 4 entity_degree_*
# + 1 fraud_neighbor_rate + 1 pagerank_score + 1 clustering_coefficient).
_EXPECTED_N_TIER5_FEATURES: int = 8


def _manifest_path() -> Path:
    return get_settings().raw_dir / "MANIFEST.json"


@pytest.fixture(scope="module")
def fitted_pipeline_and_val() -> tuple[FeaturePipeline, pd.DataFrame]:
    """Fit a full T1+T2+T3+T4+T5 pipeline on train; return pipeline + val output.

    The 10k stratified sample is `temporal_split`-ed using project
    defaults; the pipeline is fit on `splits.train`; the val output is
    used for BOTH `feature_df` and `source_df` in the leak walk
    (Tier-5 generators depend on Tier-1 columns indirectly through
    the upstream stack — see module docstring).
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
            ExponentialDecayVelocity(),
            GraphFeatureExtractor(),
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


def test_assert_no_future_leak_on_all_tier5_features(
    fitted_pipeline_and_val: tuple[FeaturePipeline, pd.DataFrame],
) -> None:
    """Every Tier-5 graph-feature column passes `assert_no_future_leak` on val.

    Sweeps `GraphFeatureExtractor.get_feature_names()` (8 cols). For
    each column, runs `assert_no_future_leak` against the val output
    with `n_samples = 50`. Failures are accumulated into a single
    error message so all leaking columns are reported together.

    Tier-5 features query frozen training-graph state, so the
    recompute is trivially leak-free; this test is the integration-
    level confirmation that the contract holds end-to-end.
    """
    pipeline, val_out = fitted_pipeline_and_val

    failures: list[str] = []
    n_features_checked = 0

    for gen in pipeline.generators:
        if not isinstance(gen, _TIER5_GENERATOR_TYPES):
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

    assert n_features_checked == _EXPECTED_N_TIER5_FEATURES, (
        f"Expected to check {_EXPECTED_N_TIER5_FEATURES} Tier-5 features; "
        f"checked {n_features_checked}. Has the default config changed?"
    )

    if failures:
        joined = "\n  - ".join(failures)
        raise AssertionError(
            f"Temporal leak detected in {len(failures)}/"
            f"{n_features_checked} Tier-5 features:\n  - {joined}"
        )
