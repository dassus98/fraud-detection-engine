"""Tests for fraud_engine.config.settings.

Business rationale:
    Settings is the single source of truth that every downstream module
    reads from. A regression here (wrong default, missing validator, or
    broken cache) silently propagates into feature pipelines, training,
    and the API. These tests lock the configuration surface against
    drift.

Trade-offs considered:
    - Tests build `Settings` with `_env_file=None` so the per-test
      result does not depend on whatever `.env` happens to be on disk.
      An alternative would have been a session-scoped monkeypatch, but
      per-call isolation is simpler and makes each test self-contained.
    - We cover the spec's five required cases plus a handful for the
      extensions this repo's Settings carries over the spec (log-level
      normalisation, temporal-split ordering) because those validators
      already existed and deserve regression coverage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from fraud_engine.config.settings import Settings, get_settings


def _build(**overrides: Any) -> Settings:
    """Construct Settings without reading the repo's .env.

    Passing `_env_file=None` overrides the `env_file=".env"` declared in
    `SettingsConfigDict` so tests do not pick up whatever the developer
    happens to have in their local .env.

    Args:
        **overrides: Field overrides passed straight to Settings().

    Returns:
        A fresh Settings instance bound to the supplied overrides.
    """
    return Settings(_env_file=None, **overrides)


# ---------------------------------------------------------------------
# Defaults — confirms the documented contract in .env.example
# ---------------------------------------------------------------------


class TestDefaults:
    """Defaults must match the values documented in .env.example."""

    def test_defaults_load_without_error(self) -> None:
        settings = _build()
        assert settings.seed == 42
        assert settings.log_level == "INFO"
        assert settings.fraud_cost_usd == 450.0
        assert settings.fp_cost_usd == 35.0
        assert settings.tp_cost_usd == 5.0
        assert settings.decision_threshold == 0.5
        assert settings.api_host == "0.0.0.0"
        assert settings.api_port == 8000

    def test_service_defaults_match_env_example(self) -> None:
        """Infra defaults align with .env.example so dev bring-up is one-step."""
        settings = _build()
        assert settings.redis_url == "redis://localhost:6379/0"
        assert settings.postgres_url == "postgresql://fraud:fraud@localhost:5432/fraud"
        assert settings.mlflow_tracking_uri == "./mlruns"
        assert settings.mlflow_experiment_name == "fraud-detection"
        assert settings.mlflow_port == 5000
        assert settings.prometheus_port == 9090
        assert settings.grafana_port == 3000


# ---------------------------------------------------------------------
# Cost validation — negatives must fail
# ---------------------------------------------------------------------


class TestCostValidation:
    """All three economic cost fields reject negatives."""

    @pytest.mark.parametrize("field", ["fraud_cost_usd", "fp_cost_usd", "tp_cost_usd"])
    def test_negative_cost_raises(self, field: str) -> None:
        with pytest.raises(ValidationError):
            _build(**{field: -1.0})

    @pytest.mark.parametrize("field", ["fraud_cost_usd", "fp_cost_usd", "tp_cost_usd"])
    def test_zero_cost_allowed(self, field: str) -> None:
        """Zero is a valid edge case (e.g. free-to-investigate TP)."""
        settings = _build(**{field: 0.0})
        assert getattr(settings, field) == 0.0


# ---------------------------------------------------------------------
# Decision threshold — must stay in [0, 1]
# ---------------------------------------------------------------------


class TestThresholdValidation:
    def test_threshold_below_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            _build(decision_threshold=-0.01)

    def test_threshold_above_one_raises(self) -> None:
        with pytest.raises(ValidationError):
            _build(decision_threshold=1.01)

    @pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
    def test_threshold_boundary_accepted(self, value: float) -> None:
        settings = _build(decision_threshold=value)
        assert settings.decision_threshold == value


# ---------------------------------------------------------------------
# Derived paths
# ---------------------------------------------------------------------


class TestDerivedPaths:
    def test_derived_paths_rooted_in_data_dir(self, tmp_path: Path) -> None:
        settings = _build(data_dir=tmp_path)
        assert settings.raw_dir == tmp_path / "raw"
        assert settings.interim_dir == tmp_path / "interim"
        assert settings.processed_dir == tmp_path / "processed"

    def test_derived_paths_track_data_dir_changes(self, tmp_path: Path) -> None:
        """The properties recompute — they don't cache a stale root."""
        first = tmp_path / "first"
        settings = _build(data_dir=first)
        assert settings.raw_dir == first / "raw"


# ---------------------------------------------------------------------
# Log-level validator (repo extension)
# ---------------------------------------------------------------------


class TestLogLevelValidator:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("debug", "DEBUG"),
            ("INFO", "INFO"),
            ("Warning", "WARNING"),
            ("error", "ERROR"),
            ("CRITICAL", "CRITICAL"),
        ],
    )
    def test_log_level_normalised(self, value: str, expected: str) -> None:
        settings = _build(log_level=value)
        assert settings.log_level == expected

    def test_invalid_log_level_raises(self) -> None:
        with pytest.raises(ValidationError):
            _build(log_level="TRACE")


# ---------------------------------------------------------------------
# Temporal split validator (repo extension — Sprint 1)
# ---------------------------------------------------------------------


class TestTemporalSplitValidator:
    def test_val_end_must_exceed_train_end(self) -> None:
        with pytest.raises(ValidationError):
            _build(train_end_dt=10_000_000, val_end_dt=10_000_000)

    def test_val_end_below_train_end_raises(self) -> None:
        with pytest.raises(ValidationError):
            _build(train_end_dt=10_000_000, val_end_dt=9_000_000)


# ---------------------------------------------------------------------
# get_settings() cache behaviour
# ---------------------------------------------------------------------


class TestGetSettingsCache:
    def test_returns_same_instance(self) -> None:
        """lru_cache guarantees one Settings per process."""
        get_settings.cache_clear()
        try:
            first = get_settings()
            second = get_settings()
            assert first is second
        finally:
            # Hygiene: clear the cache so neighbouring tests are not poisoned
            # by whatever this test's .env happened to resolve to.
            get_settings.cache_clear()


# ---------------------------------------------------------------------
# ensure_directories (side-effecting helper)
# ---------------------------------------------------------------------


class TestEnsureDirectories:
    def test_creates_full_tree(self, tmp_path: Path) -> None:
        settings = _build(
            data_dir=tmp_path / "data",
            models_dir=tmp_path / "models",
            logs_dir=tmp_path / "logs",
        )
        settings.ensure_directories()
        assert settings.data_dir.is_dir()
        assert settings.raw_dir.is_dir()
        assert settings.interim_dir.is_dir()
        assert settings.processed_dir.is_dir()
        assert settings.models_dir.is_dir()
        assert settings.logs_dir.is_dir()

    def test_idempotent(self, tmp_path: Path) -> None:
        settings = _build(
            data_dir=tmp_path / "data",
            models_dir=tmp_path / "models",
            logs_dir=tmp_path / "logs",
        )
        settings.ensure_directories()
        settings.ensure_directories()  # second call must not raise
        assert settings.data_dir.is_dir()
