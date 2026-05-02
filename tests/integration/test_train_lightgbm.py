"""Integration test for `scripts/train_lightgbm.py::train_pipeline`.

Spec gates:
- 5-trial smoke completes on a stratified subsample.
- Output files exist (model joblib + manifest, calibrator, report,
  latency PNG).
- Calibration log_loss does not regress beyond 1% drift vs baseline.
- Single-row inference latency is reasonable (smoke ceiling 100 ms;
  the production gate `<15 ms` is enforced in the production run,
  not the smoke).
- Saved model joblib loads back and `predict_proba(X)` returns
  shape `(n, 2)`.

Skip-gated on `data/processed/tier5_train.parquet` presence (the
script's input). Mirrors `tests/integration/test_tier5_e2e.py`'s
`MANIFEST.json` skip pattern.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest
from scripts.train_lightgbm import (
    _MODELS_SUBDIR,
    _NON_FEATURE_COLS,
    _SMOKE_EARLY_STOPPING_ROUNDS,
    _SMOKE_NUM_BOOST_ROUND,
    _SMOKE_SAMPLE_SIZE,
    TrainingResult,
    train_pipeline,
)

from fraud_engine.config.settings import Settings, get_settings
from fraud_engine.models.lightgbm_model import LightGBMFraudModel

pytestmark = pytest.mark.integration

_SMOKE_N_TRIALS: int = 3
_SMOKE_SEED: int = 42
_SMOKE_LATENCY_CEILING_MS: float = 100.0
_SMOKE_VAL_AUC_FLOOR: float = 0.5
_CAL_DRIFT_TOLERANCE: float = 0.01


def _processed_dir_has_tier5() -> bool:
    """True iff `data/processed/tier5_train.parquet` exists locally."""
    return (get_settings().processed_dir / "tier5_train.parquet").is_file()


@pytest.fixture(scope="module")
def isolated_settings(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[Settings]:
    """Settings with isolated MLFLOW_TRACKING_URI + MODELS_DIR.

    Module-scoped (was function-scoped pre-audit) so the upstream
    `smoke_result` fixture can also be module-scoped — one tuning
    sweep + final-fit per test FILE rather than per test, ~6× wall
    speedup on the 6-test suite. Keeps the real `data/processed/`
    dir for input parquets but redirects every output (models,
    mlruns, logs) to a tmp dir so the test doesn't pollute the
    repo's `models/` or `./mlruns/`.

    `pytest.MonkeyPatch().context()` is the pytest-recommended idiom
    for module-scoped env-var patching (the function-scoped
    `monkeypatch` fixture can't be used at module scope).
    """
    if not _processed_dir_has_tier5():
        pytest.skip(
            "data/processed/tier5_train.parquet not present — run "
            "`uv run python scripts/build_features_all_tiers.py` first."
        )

    real_data_dir = get_settings().data_dir
    tmp_path = tmp_path_factory.mktemp("integ_train_lightgbm")
    models_dir = tmp_path / "models"
    logs_dir = tmp_path / "logs"
    mlruns = tmp_path / "mlruns"

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("DATA_DIR", str(real_data_dir))  # keep real data
        mp.setenv("MODELS_DIR", str(models_dir))
        mp.setenv("LOGS_DIR", str(logs_dir))
        mp.setenv("MLFLOW_TRACKING_URI", str(mlruns))
        mp.setenv("SEED", str(_SMOKE_SEED))
        mp.setenv("LOG_LEVEL", "WARNING")

        get_settings.cache_clear()
        settings = Settings()
        settings.ensure_directories()
        yield settings
        get_settings.cache_clear()


@pytest.fixture(scope="module")
def smoke_result(
    isolated_settings: Settings,
    tmp_path_factory: pytest.TempPathFactory,
) -> TrainingResult:
    """Run the smoke pipeline once; share the result across assertions.

    Module-scoped: the (Optuna tuning + final fit + calibration +
    save + report) pipeline runs once for the entire test file, not
    once per test. 6-test wall went from ~70 s to ~12 s after this
    audit fix (was function-scoped, against the docstring's
    promise — the docstring said "share across tests" but the
    function scope made each test re-run the pipeline).
    """
    tmp_path = tmp_path_factory.mktemp("smoke_result")
    report_path = tmp_path / "model_a_training_report.md"
    figure_path = tmp_path / "figures" / "model_a_latency.png"
    return train_pipeline(
        settings=isolated_settings,
        n_trials=_SMOKE_N_TRIALS,
        skip_tuning=False,
        sample_size=_SMOKE_SAMPLE_SIZE,
        num_boost_round=_SMOKE_NUM_BOOST_ROUND,
        early_stopping_rounds=_SMOKE_EARLY_STOPPING_ROUNDS,
        random_state=_SMOKE_SEED,
        report_path=report_path,
        figure_path=figure_path,
    )


# ---------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------


def test_smoke_completes_with_non_catastrophic_auc(smoke_result: TrainingResult) -> None:
    """5-trial smoke runs to completion; val AUC > 0.5 (not catastrophic)."""
    assert smoke_result.val_auc > _SMOKE_VAL_AUC_FLOOR
    assert smoke_result.test_auc > _SMOKE_VAL_AUC_FLOOR
    assert smoke_result.n_trials == _SMOKE_N_TRIALS


def test_output_files_exist(smoke_result: TrainingResult) -> None:
    """Model + manifest + calibrator + report + figure all written."""
    assert smoke_result.model_path.is_file()
    assert smoke_result.calibrator_path.is_file()
    assert smoke_result.report_path.is_file()
    assert smoke_result.figure_path.is_file()
    # The manifest sidecar lives next to the model joblib.
    manifest = smoke_result.model_path.parent / "lightgbm_model_manifest.json"
    assert manifest.is_file()


def test_calibration_no_log_loss_regression(smoke_result: TrainingResult) -> None:
    """Calibrated val log_loss ≤ uncalibrated × (1 + tolerance)."""
    threshold = smoke_result.val_log_loss_uncalibrated * (1.0 + _CAL_DRIFT_TOLERANCE)
    assert smoke_result.val_log_loss_calibrated <= threshold


def test_inference_latency_under_smoke_ceiling(smoke_result: TrainingResult) -> None:
    """p95 single-row latency < 100 ms on the smoke (loose ceiling).

    The production gate (<15 ms) is enforced in the actual run, not
    the smoke. The smoke ceiling exists so a 10×-regression in
    inference cost surfaces in CI.
    """
    assert smoke_result.latency_p95_ms < _SMOKE_LATENCY_CEILING_MS
    # Sanity: p50 ≤ p95 ≤ p99.
    assert smoke_result.latency_p50_ms <= smoke_result.latency_p95_ms
    assert smoke_result.latency_p95_ms <= smoke_result.latency_p99_ms


def test_saved_model_round_trips(
    isolated_settings: Settings,
    smoke_result: TrainingResult,
) -> None:
    """Saved joblib loads back; `predict_proba(X)` returns `(n, 2)` ∈ [0, 1]."""
    models_dir = isolated_settings.models_dir / _MODELS_SUBDIR
    reloaded = LightGBMFraudModel.load(models_dir)
    # Predict on a small frame from the real val parquet.
    val = pd.read_parquet(isolated_settings.processed_dir / "tier5_val.parquet").head(20)
    feature_cols = [
        c
        for c in val.columns
        if c not in _NON_FEATURE_COLS
        and not pd.api.types.is_object_dtype(val[c])
        and not pd.api.types.is_string_dtype(val[c])
    ]
    proba = reloaded.predict_proba(val[feature_cols])
    assert proba.shape == (20, 2)
    assert (proba >= 0.0).all() and (proba <= 1.0).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)


def test_report_contains_expected_sections(smoke_result: TrainingResult) -> None:
    """Generated report carries the spec-required sections."""
    body = smoke_result.report_path.read_text(encoding="utf-8")
    assert "# Model A — LightGBM training report" in body
    assert "## Headline metrics" in body
    assert "## Acceptance gates" in body
    assert "## Inference latency" in body
    assert "## Best Optuna parameters" in body
    assert "## Trial history" in body
    assert "Top" in body and "feature importances" in body
    # Sanity: realised numbers from the result appear in the report.
    assert f"{smoke_result.val_auc:.4f}" in body
