"""Feature engineering. Sprint 2 onwards.

Public surface:
    BaseFeatureGenerator: ABC every Sprint 2+ feature generator inherits.
    BehavioralDeviation: Tier-3 per-card1 behavioural-deviation features
        (amount z-score, time z-score, addr/device change, hour deviation).
    ColdStartHandler: Tier-3 thin sibling that emits
        `is_coldstart_{entity}` flags for entities with thin history.
    ExponentialDecayVelocity: Tier-4 per-(entity, λ) exponentially-decayed
        velocity (EWM). O(1) running-state per event with read-before-push
        two-pass discipline; fraud-weighted variant is OOF-safe.
    FeaturePipeline: sequential composition with save / load + manifest.
    GraphFeatureExtractor: Tier-5 per-transaction graph features
        derived from the bipartite TransactionEntityGraph. Five
        distinct features (8 columns): connected_component_size,
        entity_degree_{entity} (×4), fraud_neighbor_rate (OOF-safe),
        pagerank_score, clustering_coefficient. Cold-start val/test
        rows emit NaN for txn-level features.
    HistoricalStats: Tier-2 per-entity rolling mean / std / max over
        an amount column. Captures expected-spending shape.
    NanGroupReducer: Tier-3 V-feature reducer that drops redundant
        siblings within shared NaN-groups. The lone exception to the
        BaseFeatureGenerator 'preserve all columns' contract.
    TargetEncoder: Tier-2 out-of-fold target encoder for
        high-cardinality categoricals. OOF on training; full-train
        encoder for val / test.
    TemporalSafeGenerator: row-iterating ABC subclass that is leak-free
        by construction. Reference shape for tier 2-5 vectorized
        generators.
    TransactionEntityGraph: Tier-5 bipartite graph linking transactions
        to their entities (card1, addr1, DeviceInfo, P_emaildomain).
        Construction primitive; subsequent prompts derive feature
        columns from this graph.
    VelocityCounter: Tier-2 deque-based per-entity transaction counts
        over fixed lookback windows. Canonical fraud signal.
    assert_no_future_leak: sample-based test helper that catches
        look-ahead leakage in any time-windowed feature.
"""

from __future__ import annotations

from fraud_engine.features.base import BaseFeatureGenerator
from fraud_engine.features.pipeline import FeaturePipeline
from fraud_engine.features.temporal_guards import (
    TemporalSafeGenerator,
    assert_no_future_leak,
)
from fraud_engine.features.tier2_aggregations import (
    HistoricalStats,
    TargetEncoder,
    VelocityCounter,
)
from fraud_engine.features.tier3_behavioral import (
    BehavioralDeviation,
    ColdStartHandler,
)
from fraud_engine.features.tier4_decay import ExponentialDecayVelocity
from fraud_engine.features.tier5_graph import (
    GraphFeatureExtractor,
    TransactionEntityGraph,
)
from fraud_engine.features.v_reduction import NanGroupReducer

__all__ = [
    "BaseFeatureGenerator",
    "BehavioralDeviation",
    "ColdStartHandler",
    "ExponentialDecayVelocity",
    "FeaturePipeline",
    "GraphFeatureExtractor",
    "HistoricalStats",
    "NanGroupReducer",
    "TargetEncoder",
    "TemporalSafeGenerator",
    "TransactionEntityGraph",
    "VelocityCounter",
    "assert_no_future_leak",
]
