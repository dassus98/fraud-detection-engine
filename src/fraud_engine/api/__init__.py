"""FastAPI serving layer.

Sprint 5 prompt 5.1.a: Pydantic v2 request/response schemas — the typed
contract between API clients and the fraud-detection service.

Sprint 5 prompt 5.1.b: `RedisFeatureStore` — async client over Redis
for entity-keyed online feature lookup.

Sprint 5 prompt 5.1.c: `FeatureService` — orchestrator combining
Tier-1 inline + Redis entity + Postgres batch features into a single
DataFrame for the model, with per-source degraded-mode fallback to
population defaults.

Sprint 5 prompt 5.1.d: `InferenceService` — loads the production
LightGBM model + isotonic calibrator at startup, exposes
`predict(features)` with calibrated probability + threshold-based
decision, supports atomic mid-session model reload.

Routes, SHAP integration, shadow mode, and prediction logging are
populated by later 5.x prompts.
"""

from __future__ import annotations

from fraud_engine.api.feature_service import FeatureService, FeatureVector
from fraud_engine.api.inference import InferenceResult, InferenceService
from fraud_engine.api.redis_store import RedisFeatureStore
from fraud_engine.api.schemas import (
    Card4Literal,
    Card6Literal,
    DecisionLiteral,
    DependencyStatusLiteral,
    HealthResponse,
    HealthStatusLiteral,
    PredictionResponse,
    ProductCodeLiteral,
    ReadyResponse,
    ReadyStatusLiteral,
    Reason,
    ReasonDirectionLiteral,
    RequestMetadata,
    TransactionRequest,
)

__all__ = [
    "Card4Literal",
    "Card6Literal",
    "DecisionLiteral",
    "DependencyStatusLiteral",
    "FeatureService",
    "FeatureVector",
    "HealthResponse",
    "HealthStatusLiteral",
    "InferenceResult",
    "InferenceService",
    "PredictionResponse",
    "ProductCodeLiteral",
    "ReadyResponse",
    "ReadyStatusLiteral",
    "Reason",
    "ReasonDirectionLiteral",
    "RedisFeatureStore",
    "RequestMetadata",
    "TransactionRequest",
]
