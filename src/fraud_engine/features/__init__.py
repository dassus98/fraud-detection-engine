"""Feature engineering. Sprint 2 onwards.

Public surface:
    BaseFeatureGenerator: ABC every Sprint 2+ feature generator inherits.
    FeaturePipeline: sequential composition with save / load + manifest.
"""

from __future__ import annotations

from fraud_engine.features.base import BaseFeatureGenerator
from fraud_engine.features.pipeline import FeaturePipeline

__all__ = ["BaseFeatureGenerator", "FeaturePipeline"]
