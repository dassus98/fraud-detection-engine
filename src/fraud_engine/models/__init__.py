"""Model training and inference.

Sprint 1 ships the LightGBM baseline (random + temporal variants).
Sprint 3 extends this package with the Optuna-tuned LightGBM, the
entity-embedding neural network, and the PyG graph model.
"""

from __future__ import annotations

from fraud_engine.models.baseline import BaselineResult, Variant, train_baseline

__all__ = [
    "BaselineResult",
    "Variant",
    "train_baseline",
]
