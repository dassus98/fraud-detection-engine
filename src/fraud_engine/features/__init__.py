"""Feature engineering. Sprint 2 onwards.

Public surface:
    BaseFeatureGenerator: ABC every Sprint 2+ feature generator inherits.
    FeaturePipeline: sequential composition with save / load + manifest.
    TemporalSafeGenerator: row-iterating ABC subclass that is leak-free
        by construction. Reference shape for tier 2-5 vectorized
        generators.
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

__all__ = [
    "BaseFeatureGenerator",
    "FeaturePipeline",
    "TemporalSafeGenerator",
    "assert_no_future_leak",
]
