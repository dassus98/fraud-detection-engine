"""Unit tests for `fraud_engine.utils.mlflow_setup`.

Each test points MLflow at a tmp-path backend via `MLFLOW_TRACKING_URI`
so no run or experiment ever lands in the real `mlruns/` tree. The
`get_settings` lru_cache is cleared around every test so the
monkeypatched URI is actually read.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import mlflow
import pandas as pd
import pytest

from fraud_engine.config.settings import Settings, get_settings
from fraud_engine.utils.mlflow_setup import (
    configure_mlflow,
    log_dataframe_stats,
    log_economic_metrics,
    setup_experiment,
)


@pytest.fixture
def mlflow_tmp(
    mock_settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[str]:
    """Redirect MLflow to a file-backed tracking URI under `tmp_path`.

    Yields the tracking URI (a `file:` URL). Teardown ends any still-
    active run so neighbouring tests don't inherit open-run state, and
    clears the settings cache so the next test re-reads its own env.
    """
    tracking_root = tmp_path / "mlruns"
    tracking_uri = tracking_root.resolve().as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    get_settings.cache_clear()

    yield tracking_uri

    if mlflow.active_run() is not None:
        mlflow.end_run()
    get_settings.cache_clear()
    _ = mock_settings  # keep fixture active for logs_dir redirection.


class TestConfigureMLflow:
    """Contract tests for `configure_mlflow`."""

    def test_sets_tracking_uri(self, mlflow_tmp: str) -> None:
        configure_mlflow()
        assert mlflow.get_tracking_uri() == mlflow_tmp


class TestSetupExperiment:
    """Contract tests for `setup_experiment`."""

    def test_creates_and_returns_id(self, mlflow_tmp: str) -> None:
        configure_mlflow()
        first = setup_experiment("unit-experiment-a")
        second = setup_experiment("unit-experiment-a")
        assert first == second
        assert first  # MLflow IDs are non-empty strings.
        _ = mlflow_tmp

    def test_defaults_to_settings_name(self, mlflow_tmp: str) -> None:
        configure_mlflow()
        exp_id = setup_experiment()
        exp = mlflow.get_experiment(exp_id)
        assert exp.name == get_settings().mlflow_experiment_name
        _ = mlflow_tmp


class TestLogDataframeStats:
    """Contract tests for `log_dataframe_stats`."""

    def test_records_params_and_metrics(self, mlflow_tmp: str) -> None:
        configure_mlflow()
        exp_id = setup_experiment("unit-df-stats")

        df = pd.DataFrame(
            {
                "a": range(10),
                "b": [float(i) for i in range(10)],
                "c": ["x"] * 10,
            }
        )

        with mlflow.start_run(experiment_id=exp_id) as run:
            log_dataframe_stats(df, prefix="train")
            run_id = run.info.run_id

        data = mlflow.get_run(run_id).data
        assert data.params["train_rows"] == "10"
        assert data.params["train_cols"] == "3"
        assert "train_memory_mb" in data.params
        # dtype histogram is JSON; each dtype appears at least once.
        assert "int64" in data.params["train_dtypes"] or "Int64" in data.params["train_dtypes"]
        assert data.metrics["train_n_missing"] == 0.0
        assert data.metrics["train_n_duplicates"] == 0.0
        _ = mlflow_tmp


class TestLogEconomicMetrics:
    """Contract tests for `log_economic_metrics`."""

    def test_raises_outside_run(self, mlflow_tmp: str) -> None:
        configure_mlflow()
        # Defensive: explicitly assert no active run leaked from elsewhere.
        assert mlflow.active_run() is None
        with pytest.raises(RuntimeError, match="active MLflow run"):
            log_economic_metrics(0.1, 0.02, 1234.0)
        _ = mlflow_tmp

    def test_records_three_metrics(self, mlflow_tmp: str) -> None:
        configure_mlflow()
        exp_id = setup_experiment("unit-econ-metrics")

        with mlflow.start_run(experiment_id=exp_id) as run:
            log_economic_metrics(fn_rate=0.125, fp_rate=0.03, total_cost_usd=9876.5)
            run_id = run.info.run_id

        metrics = mlflow.get_run(run_id).data.metrics
        assert metrics["fn_rate"] == pytest.approx(0.125)
        assert metrics["fp_rate"] == pytest.approx(0.03)
        assert metrics["total_cost_usd"] == pytest.approx(9876.5)
        _ = mlflow_tmp
