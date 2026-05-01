"""Model training and inference.

Sprint 1 ships the LightGBM baseline (random + temporal variants) via
`train_baseline` (sklearn-style `LGBMClassifier` API).

Sprint 3 ships `LightGBMFraudModel`, a wrapper around the native
`lgb.Booster` API with explicit early stopping, joblib + JSON-manifest
persistence, and `feature_importance` / `predict_proba` / `save` /
`load` surface area. This is the production-realistic model surface
the hyperparameter-tuning and economic-cost evaluator stages consume.
Subsequent prompts add the Optuna-tuned LightGBM, the entity-embedding
neural network, and the PyG graph model.
"""

from __future__ import annotations

from fraud_engine.models.baseline import BaselineResult, Variant, train_baseline
from fraud_engine.models.lightgbm_model import LightGBMFraudModel

__all__ = [
    "BaselineResult",
    "LightGBMFraudModel",
    "Variant",
    "train_baseline",
]
