"""Integration tests for `fraud_engine.models.tuning.run_tuning`.

Five test surfaces (matches the spec's "5-trial smoke" gate):

- 5-trial smoke completes on a synthetic 600-row frame; returned
  dict carries `best_params`, `best_value`, `n_trials`, `study_name`,
  `output_path`.
- Best params are written to the YAML output path; `yaml.safe_load`
  round-trips identical content; schema_version present.
- MLflow tracking shows 1 parent run + 5 nested trial runs, each
  trial run carrying a `val_auc` metric.
- `n_trials=0` raises `ValueError`.
- Every key in `SEARCH_SPACE_KEYS` appears in `best_params` (catches
  drift between the harness and the YAML output).

The fixture monkeypatches `MLFLOW_TRACKING_URI` to `tmp_path / "mlruns"`
so the test never writes to the real `./mlruns/`. Mirrors
`tests/unit/test_baseline.py::baseline_settings`.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
import yaml

from fraud_engine.config.settings import Settings, get_settings
from fraud_engine.models.tuning import SEARCH_SPACE_KEYS, run_tuning

pytestmark = pytest.mark.integration

_SMOKE_N_TRIALS: int = 5
_SMOKE_NUM_BOOST_ROUND: int = 30
_SMOKE_EARLY_STOPPING_ROUNDS: int = 5
_SMOKE_SEED: int = 42


# ---------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------


@pytest.fixture
def tuning_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Settings]:
    """Settings with fully isolated paths, including MLflow tracking URI.

    Mirrors `tests/unit/test_baseline.py::baseline_settings`. The
    cleared `get_settings` cache before/after ensures the singleton
    sees the monkeypatched env vars.
    """
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    logs_dir = tmp_path / "logs"
    mlruns = tmp_path / "mlruns"
    for sub in ("raw", "interim", "processed"):
        (data_dir / sub).mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("MODELS_DIR", str(models_dir))
    monkeypatch.setenv("LOGS_DIR", str(logs_dir))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", str(mlruns))
    monkeypatch.setenv("SEED", str(_SMOKE_SEED))
    monkeypatch.setenv("LOG_LEVEL", "WARNING")

    get_settings.cache_clear()
    settings = Settings()
    settings.ensure_directories()
    yield settings
    get_settings.cache_clear()
    mlflow.set_tracking_uri("./mlruns")


def _make_synthetic_xy(  # noqa: N802 — sklearn convention
    n_rows: int = 600,
    n_features: int = 5,
    seed: int = _SMOKE_SEED,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return a synthetic (X, y) pair with modelling signal.

    Mirrors `tests/unit/test_lightgbm_model.py::_make_synthetic_xy` so
    behaviour is consistent across the model wrapper + tuning harness
    test suites.
    """
    rng = np.random.default_rng(seed)
    cols = {f"f{i}": rng.normal(0, 1, size=n_rows).astype(np.float32) for i in range(n_features)}
    cols["amount"] = (np.exp(rng.normal(0, 1, size=n_rows)) * 50.0).astype(np.float32)
    x_df = pd.DataFrame(cols)
    fraud_logit = (
        0.9 * (x_df["amount"].to_numpy() / x_df["amount"].mean() - 1) + 0.5 * x_df["f0"].to_numpy()
    )
    fraud_prob = 1.0 / (1.0 + np.exp(-fraud_logit))
    y = (rng.uniform(0, 1, size=n_rows) < 0.20 * fraud_prob / fraud_prob.mean()).astype(np.int64)
    if y.sum() == 0:
        y[0] = 1
    if y.sum() == n_rows:
        y[0] = 0
    return x_df, y


def _train_val_split(
    x: pd.DataFrame, y: np.ndarray, train_frac: float = 0.7, seed: int = _SMOKE_SEED
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """Random train/val split. The harness itself is split-agnostic."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    cut = int(len(x) * train_frac)
    train_idx, val_idx = idx[:cut], idx[cut:]
    x_train = x.iloc[train_idx].reset_index(drop=True)
    x_val = x.iloc[val_idx].reset_index(drop=True)
    return x_train, y[train_idx], x_val, y[val_idx]


def _run_smoke(tmp_path: Path) -> dict[str, object]:
    """Helper: run the 5-trial smoke and return the result dict.

    Encapsulates the synthetic-data setup so each test reads as the
    intent it asserts on, not the boilerplate it shares.
    """
    x_df, y = _make_synthetic_xy()
    x_train, y_train, x_val, y_val = _train_val_split(x_df, y)
    output_path = tmp_path / "best_params.yaml"
    return run_tuning(
        x_train,
        y_train,
        x_val,
        y_val,
        n_trials=_SMOKE_N_TRIALS,
        study_name="tuning_smoke",
        output_path=output_path,
        random_state=_SMOKE_SEED,
        num_boost_round=_SMOKE_NUM_BOOST_ROUND,
        early_stopping_rounds=_SMOKE_EARLY_STOPPING_ROUNDS,
    )


# ---------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------


def test_run_tuning_5_trial_smoke_completes(
    tuning_settings: Settings,
    tmp_path: Path,
) -> None:
    """Spec-named: 5-trial smoke completes; returns expected dict shape."""
    result = _run_smoke(tmp_path)
    assert result["n_trials"] == _SMOKE_N_TRIALS
    assert result["study_name"] == "tuning_smoke"
    assert isinstance(result["best_params"], dict)
    assert len(result["best_params"]) > 0
    assert isinstance(result["best_value"], float)
    assert 0.0 <= result["best_value"] <= 1.0


def test_best_params_written_to_yaml_and_round_trips(
    tuning_settings: Settings,
    tmp_path: Path,
) -> None:
    """Spec-named: best params saved; YAML round-trips identical content."""
    result = _run_smoke(tmp_path)
    output_path = result["output_path"]
    assert isinstance(output_path, Path)
    assert output_path.is_file()

    payload = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["study_name"] == "tuning_smoke"
    assert payload["n_trials"] == _SMOKE_N_TRIALS
    assert isinstance(payload["best_value"], float)
    assert payload["best_value"] == pytest.approx(result["best_value"])
    assert isinstance(payload["best_params"], dict)
    assert payload["best_params"] == result["best_params"]


def test_mlflow_logs_one_parent_and_n_trial_runs(
    tuning_settings: Settings,
    tmp_path: Path,
) -> None:
    """1 parent study run + 5 nested trial runs; each trial logs `val_auc`.

    Uses `MlflowClient.search_runs` rather than the high-level
    `mlflow.search_runs` because the latter applies a default
    "tags.mlflow.parentRunId IS NULL" filter that hides nested runs;
    the client API exposes everything when given an explicit
    experiment id.
    """
    _run_smoke(tmp_path)

    settings = get_settings()
    experiment = mlflow.get_experiment_by_name(settings.mlflow_experiment_name)
    assert experiment is not None
    client = mlflow.MlflowClient()
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=1000,
    )

    # Expect 1 parent + 5 children = 6 runs total.
    expected_runs = 1 + _SMOKE_N_TRIALS
    assert len(runs) == expected_runs

    # Partition: parent has tag kind="study_parent"; children have a
    # parentRunId tag pointing at the parent.
    parents = [r for r in runs if r.data.tags.get("kind") == "study_parent"]
    trials = [r for r in runs if r.data.tags.get("kind") != "study_parent"]
    assert len(parents) == 1
    assert len(trials) == _SMOKE_N_TRIALS

    # Each trial run carries a `val_auc` metric in [0, 1].
    parent_run_id = parents[0].info.run_id
    for trial_run in trials:
        assert trial_run.data.tags.get("mlflow.parentRunId") == parent_run_id
        assert "val_auc" in trial_run.data.metrics
        val_auc = trial_run.data.metrics["val_auc"]
        assert 0.0 <= val_auc <= 1.0


def test_n_trials_zero_raises(
    tuning_settings: Settings,
    tmp_path: Path,
) -> None:
    """`n_trials=0` raises `ValueError` (no Optuna study created)."""
    x_df, y = _make_synthetic_xy(n_rows=100)
    x_train, y_train, x_val, y_val = _train_val_split(x_df, y)
    with pytest.raises(ValueError, match="n_trials"):
        run_tuning(
            x_train,
            y_train,
            x_val,
            y_val,
            n_trials=0,
            study_name="invalid",
            output_path=tmp_path / "best.yaml",
        )


def test_search_space_keys_all_appear_in_best_params(
    tuning_settings: Settings,
    tmp_path: Path,
) -> None:
    """Every key in `SEARCH_SPACE_KEYS` is sampled and lands in best_params.

    Catches drift between the harness's `_suggest_params` and the
    public `SEARCH_SPACE_KEYS` tuple (the test referenced by 3.3.d
    when reading the YAML to verify column shape).
    """
    result = _run_smoke(tmp_path)
    best_params: dict[str, object] = result["best_params"]  # type: ignore[assignment]
    for key in SEARCH_SPACE_KEYS:
        assert key in best_params, f"Search-space key {key!r} missing from best_params"
    # And no extra keys (the YAML output should mirror the search space).
    assert set(best_params.keys()) == set(SEARCH_SPACE_KEYS)
