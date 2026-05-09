"""FastAPI serving layer.

Sprint 5 prompt 5.1.a: Pydantic v2 request/response schemas — the typed
contract between API clients and the fraud-detection service.

Sprint 5 prompt 5.1.b: `RedisFeatureStore` — async client over Redis
for entity-keyed online feature lookup. The FastAPI route (Sprint
5.1.c) will hold one instance for the process lifetime.

Routes, SHAP integration, shadow mode, and prediction logging are
populated by later 5.x prompts.
"""

from __future__ import annotations

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
    "HealthResponse",
    "HealthStatusLiteral",
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
