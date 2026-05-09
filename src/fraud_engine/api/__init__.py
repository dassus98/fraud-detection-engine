"""FastAPI serving layer.

Sprint 5 prompt 5.1.a: Pydantic v2 request/response schemas — the typed
contract between API clients and the fraud-detection service. Routes,
Redis wiring, SHAP integration, shadow mode, and prediction logging are
populated by later 5.x prompts. Schemas are the load-bearing first step
because every later prompt builds against this typed surface.
"""

from __future__ import annotations

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
    "RequestMetadata",
    "TransactionRequest",
]
