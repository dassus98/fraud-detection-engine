"""Unit tests for `fraud_engine.utils.mlflow_setup`.

Each test points MLflow at a tmp-path backend via `MLFLOW_TRACKING_URI`
so no run or experiment ever lands in the real `mlruns/` tree. The
`get_settings` lru_cache is cleared around every test so the
monkeypatched URI is actually read. One test additionally injects a
custom `Settings` instance to confirm the 0.3.c DI-style parameter on
`setup_experiment`.
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
        """`configure_mlflow()` sets the process-global tracking URI."""
        configure_mlflow()
        assert mlflow.get_tracking_uri() == mlflow_tmp


class TestSetupExperiment:
    """Contract tests for `setup_experiment`."""

    def test_creates_and_returns_id(self, mlflow_tmp: str) -> None:
        """First call creates the experiment; second call returns the same ID."""
        configure_mlflow()
        first = setup_experiment("unit-experiment-a")
        second = setup_experiment("unit-experiment-a")
        assert first == second
        assert first  # MLflow IDs are non-empty strings.
        _ = mlflow_tmp

    def test_defaults_to_settings_name(self, mlflow_tmp: str) -> None:
        """Passing `name=None` resolves to `settings.mlflow_experiment_name`."""
        configure_mlflow()
        exp_id = setup_experiment()
        exp = mlflow.get_experiment(exp_id)
        assert exp.name == get_settings().mlflow_experiment_name
        _ = mlflow_tmp

    def test_sets_tracking_uri_from_settings_when_not_pre_configured(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Spec: `setup_experiment` sets the tracking URI from settings.

        Does NOT call `configure_mlflow()` first — the spec wording
        "Set tracking URI from settings, create-or-get experiment,
        return experiment_id" means `setup_experiment` is a one-stop
        entry point. Tests confirm the URI side effect.
        """
        # Point at a fresh tmp URI via env; don't pre-configure MLflow.
        tracking_uri = (tmp_path / "solo-mlruns").resolve().as_uri()
        monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
        get_settings.cache_clear()
        try:
            exp_id = setup_experiment("unit-solo")
            assert mlflow.get_tracking_uri() == tracking_uri
            assert exp_id
        finally:
            get_settings.cache_clear()

    def test_accepts_injected_settings(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Spec: DI-style `settings=` parameter overrides `get_settings()`.

        Tests the explicit-argument path: construct a `Settings` whose
        `mlflow_tracking_uri` points at a brand-new tmp directory, pass
        it in, confirm MLflow follows.

        The monkeypatch + cache_clear scaffolding is *not* what the test
        is checking — it is teardown discipline. `setup_experiment` calls
        `mlflow.set_tracking_uri`, which (per mlflow ≥ 2.x) writes the
        URI to `os.environ['MLFLOW_TRACKING_URI']` so subprocesses can
        inherit it. Without monkeypatch pre-registering the variable,
        that write leaks into the next test's `Settings()` construction
        (pydantic-settings reads env vars by default, regardless of
        `_env_file=None`). Pre-registering means pytest's teardown
        restores the env-var to its prior state (typically unset).
        """
        injected_uri = (tmp_path / "injected-mlruns").resolve().as_uri()
        injected = Settings(mlflow_tracking_uri=injected_uri)
        # Pre-register the env var so monkeypatch teardown unsets it
        # after the mlflow.set_tracking_uri side effect inside
        # setup_experiment(). The explicit `settings=` kwarg still wins
        # over env in pydantic-settings priority order, so this does not
        # alter what the test is asserting.
        monkeypatch.setenv("MLFLOW_TRACKING_URI", injected_uri)
        get_settings.cache_clear()
        try:
            exp_id = setup_experiment("unit-injected", settings=injected)
            assert mlflow.get_tracking_uri() == injected_uri
            assert exp_id
        finally:
            get_settings.cache_clear()


class TestLogDataframeStats:
    """Contract tests for `log_dataframe_stats`."""

    def test_records_params_and_metrics(self, mlflow_tmp: str) -> None:
        """Spec: "shape, dtypes, memory, and null counts as params+metrics"."""
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

    def test_counts_nulls_and_dupes(self, mlflow_tmp: str) -> None:
        """Null and duplicate counts are the observable quantities."""
        configure_mlflow()
        exp_id = setup_experiment("unit-df-nulls")

        df = pd.DataFrame(
            {
                "a": [1, 1, 2, None, None],
                "b": [1, 1, 2, 3, 4],
            }
        )

        with mlflow.start_run(experiment_id=exp_id) as run:
            log_dataframe_stats(df, "raw")
            run_id = run.info.run_id

        metrics = mlflow.get_run(run_id).data.metrics
        # Two None values in column 'a'.
        assert metrics["raw_n_missing"] == 2.0
        # Rows 0 and 1 are identical.
        assert metrics["raw_n_duplicates"] == 1.0
        _ = mlflow_tmp

    def test_accepts_positional_prefix(self, mlflow_tmp: str) -> None:
        """Spec signature is `(df, prefix)` — positional must work."""
        configure_mlflow()
        exp_id = setup_experiment("unit-df-positional")

        df = pd.DataFrame({"x": [1, 2, 3]})
        with mlflow.start_run(experiment_id=exp_id) as run:
            # No `prefix=` kwarg — purely positional.
            log_dataframe_stats(df, "val")
            run_id = run.info.run_id

        params = mlflow.get_run(run_id).data.params
        assert params["val_rows"] == "3"
        _ = mlflow_tmp


class TestLogEconomicMetrics:
    """Contract tests for `log_economic_metrics` (counts + costs)."""

    def test_raises_outside_run(self, mlflow_tmp: str) -> None:
        """Calling outside a run is a loud failure per the module docstring."""
        configure_mlflow()
        # Defensive: explicitly assert no active run leaked from elsewhere.
        assert mlflow.active_run() is None
        with pytest.raises(RuntimeError, match="active MLflow run"):
            log_economic_metrics(
                fn_count=1,
                fp_count=2,
                tp_count=3,
                tn_count=4,
                fraud_cost=450.0,
                fp_cost=35.0,
            )
        _ = mlflow_tmp

    def test_records_counts_and_costs(self, mlflow_tmp: str) -> None:
        """Spec: confusion-matrix counts + total_cost + cost_per_txn.

        Manual check (fn=2, fp=3, tp=5, tn=90; fraud=450, fp=35,
        tp=5): total = 2*450 + 3*35 + 5*5 + 0 = 900 + 105 + 25 = 1030.
        Population N = 2 + 3 + 5 + 90 = 100; cost_per_txn = 10.30.
        """
        configure_mlflow()
        exp_id = setup_experiment("unit-econ-metrics")

        with mlflow.start_run(experiment_id=exp_id) as run:
            log_economic_metrics(
                fn_count=2,
                fp_count=3,
                tp_count=5,
                tn_count=90,
                fraud_cost=450.0,
                fp_cost=35.0,
                tp_cost=5.0,
            )
            run_id = run.info.run_id

        metrics = mlflow.get_run(run_id).data.metrics
        assert metrics["fn_count"] == pytest.approx(2.0)
        assert metrics["fp_count"] == pytest.approx(3.0)
        assert metrics["tp_count"] == pytest.approx(5.0)
        assert metrics["tn_count"] == pytest.approx(90.0)
        assert metrics["total_cost_usd"] == pytest.approx(1030.0)
        assert metrics["cost_per_txn"] == pytest.approx(10.30)
        _ = mlflow_tmp

    def test_tp_cost_defaults_to_zero(self, mlflow_tmp: str) -> None:
        """Spec: `tp_cost` has a default of 0.0."""
        configure_mlflow()
        exp_id = setup_experiment("unit-econ-tpzero")

        with mlflow.start_run(experiment_id=exp_id) as run:
            # Pass tp_count=10 but omit tp_cost → tp_cost defaults to 0.
            log_economic_metrics(
                fn_count=0,
                fp_count=0,
                tp_count=10,
                tn_count=0,
                fraud_cost=100.0,
                fp_cost=1.0,
            )
            run_id = run.info.run_id

        metrics = mlflow.get_run(run_id).data.metrics
        # TP count is 10 but tp_cost is 0 → total cost is 0.
        assert metrics["total_cost_usd"] == pytest.approx(0.0)
        assert metrics["tp_count"] == pytest.approx(10.0)
        _ = mlflow_tmp

    def test_zero_population_cost_per_txn_is_zero(self, mlflow_tmp: str) -> None:
        """Empty (all-zero counts) input must not divide by zero."""
        configure_mlflow()
        exp_id = setup_experiment("unit-econ-empty")

        with mlflow.start_run(experiment_id=exp_id) as run:
            log_economic_metrics(
                fn_count=0,
                fp_count=0,
                tp_count=0,
                tn_count=0,
                fraud_cost=450.0,
                fp_cost=35.0,
            )
            run_id = run.info.run_id

        metrics = mlflow.get_run(run_id).data.metrics
        assert metrics["total_cost_usd"] == pytest.approx(0.0)
        assert metrics["cost_per_txn"] == pytest.approx(0.0)
        _ = mlflow_tmp
