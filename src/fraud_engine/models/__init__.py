"""Model training and inference.

Sprint 1 ships the LightGBM baseline (random + temporal variants) via
`train_baseline` (sklearn-style `LGBMClassifier` API).

Sprint 3 ships `LightGBMFraudModel`, a wrapper around the native
`lgb.Booster` API with explicit early stopping, joblib + JSON-manifest
persistence, and `feature_importance` / `predict_proba` / `save` /
`load` surface area. This is the production-realistic model surface
the hyperparameter-tuning and economic-cost evaluator stages consume.

Sprint 3 prompt 3.3.b adds `run_tuning`, an Optuna-driven hyperparameter
sweep over `LightGBMFraudModel`. Each trial is logged to MLflow as a
nested run; best params land in `configs/model_best_params.yaml` for
downstream consumers.

Sprint 3 prompt 3.4.a adds `FraudNetModel` (Model B): a PyTorch
entity-embedding network for `card1` / `addr1` / `DeviceInfo`, trained
with focal loss against the 3.5 % fraud base rate. Diversity model;
shadow-deployable per CLAUDE.md §3.

Sprint 3 prompt 3.4.b adds `FraudGNNModel` (Model C): a 3-layer
GraphSAGE network operating over the bipartite transaction-entity
graph, trained with focal loss + neighbor sampling. Batch-only per
CLAUDE.md §3 — its outputs feed Model A as features in Sprint 5.
"""

from __future__ import annotations

from fraud_engine.models.baseline import BaselineResult, Variant, train_baseline
from fraud_engine.models.gnn_model import FraudGNN, FraudGNNModel
from fraud_engine.models.lightgbm_model import LightGBMFraudModel
from fraud_engine.models.neural_model import FocalLoss, FraudNet, FraudNetModel
from fraud_engine.models.tuning import SEARCH_SPACE_KEYS, run_tuning

__all__ = [
    "BaselineResult",
    "FocalLoss",
    "FraudGNN",
    "FraudGNNModel",
    "FraudNet",
    "FraudNetModel",
    "LightGBMFraudModel",
    "SEARCH_SPACE_KEYS",
    "Variant",
    "run_tuning",
    "train_baseline",
]
