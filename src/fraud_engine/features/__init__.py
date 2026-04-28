"""Feature engineering. Sprint 2 onwards.

Public surface:
    BaseFeatureGenerator: ABC every Sprint 2+ feature generator inherits.
    FeaturePipeline: sequential composition with save / load + manifest.
    HistoricalStats: Tier-2 per-entity rolling mean / std / max over
        an amount column. Captures expected-spending shape.
    TargetEncoder: Tier-2 out-of-fold target encoder for
        high-cardinality categoricals. OOF on training; full-train
        encoder for val / test.
    TemporalSafeGenerator: row-iterating ABC subclass that is leak-free
        by construction. Reference shape for tier 2-5 vectorized
        generators.
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

__all__ = [
    "BaseFeatureGenerator",
    "FeaturePipeline",
    "HistoricalStats",
    "TargetEncoder",
    "TemporalSafeGenerator",
    "VelocityCounter",
    "assert_no_future_leak",
]
