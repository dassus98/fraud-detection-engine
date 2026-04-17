"""Configuration surface — Pydantic settings read from .env."""

from __future__ import annotations

from fraud_engine.config.settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]
