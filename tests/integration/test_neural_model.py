"""Integration test for `scripts/train_neural.py::train_pipeline`.

Spec gates:
- Trains on a 50K stratified subsample to convergence (early stopping
  fires OR final-epoch val AUC matches the best within 1%).
- Output files exist (model joblib + manifest + report + curves PNG).
- Val AUC > 0.6 (not catastrophic; FraudNet is the diversity model
  so we don't gate against the 0.93 production target).
- Saved model joblib loads back; `predict_proba(X)` returns shape
  `(n, 2)` with row-wise unit sums.
- Focal loss reduces majority-class easy-positive loss vs BCE while
  preserving the cross-entropy ordering of (easy, medium, hard).
- OOV / unseen entity values handled without crashing.

Skip-gated on `data/processed/tier5_train.parquet` presence (mirrors
`tests/integration/test_train_lightgbm.py`).
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest
import torch
from scripts.train_neural import (
    _MODELS_SUBDIR,
    TrainingResult,
    _select_columns_for_fraudnet,
    train_pipeline,
)

from fraud_engine.config.settings import Settings, get_settings
from fraud_engine.models.neural_model import (
    FocalLoss,
    FraudNetModel,
)

pytestmark = pytest.mark.integration

_SMOKE_SAMPLE_SIZE: int = 50_000
_SMOKE_EPOCHS: int = 10
_SMOKE_PATIENCE: int = 3
_SMOKE_SEED: int = 42
_SMOKE_VAL_AUC_FLOOR: float = 0.6
_SMOKE_LATENCY_CEILING_MS: float = 100.0
_CONVERGENCE_TOLERANCE: float = 0.01


def _processed_dir_has_tier5() -> bool:
    """True iff `data/processed/tier5_train.parquet` exists locally."""
    return (get_settings().processed_dir / "tier5_train.parquet").is_file()


@pytest.fixture(scope="module")
def isolated_settings(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[Settings]:
    """Settings with isolated MLFLOW_TRACKING_URI + MODELS_DIR.

    Mirrors `test_train_lightgbm.isolated_settings`. Keeps the real
    `data/processed/` dir for input parquets but redirects every
    output (models, mlruns, logs) to a module-scoped tmp dir.

    Module-scoped (rather than the default function scope) so the
    expensive smoke fixture below trains once for the whole module
    instead of once per assertion. The pytest builtin `monkeypatch`
    fixture is function-scoped, so we use a manual `MonkeyPatch`
    context here to keep the lifetime aligned with this fixture.
    """
    if not _processed_dir_has_tier5():
        pytest.skip(
            "data/processed/tier5_train.parquet not present - run "
            "`uv run python scripts/build_features_all_tiers.py` first."
        )

    tmp_path = tmp_path_factory.mktemp("neural_smoke")
    monkeypatch = pytest.MonkeyPatch()
    real_data_dir = get_settings().data_dir
    models_dir = tmp_path / "models"
    logs_dir = tmp_path / "logs"
    mlruns = tmp_path / "mlruns"

    monkeypatch.setenv("DATA_DIR", str(real_data_dir))  # keep real data
    monkeypatch.setenv("MODELS_DIR", str(models_dir))
    monkeypatch.setenv("LOGS_DIR", str(logs_dir))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", str(mlruns))
    monkeypatch.setenv("SEED", str(_SMOKE_SEED))
    monkeypatch.setenv("LOG_LEVEL", "WARNING")

    get_settings.cache_clear()
    settings = Settings()
    settings.ensure_directories()
    try:
        yield settings
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()


@pytest.fixture(scope="module")
def smoke_result(
    isolated_settings: Settings,
    tmp_path_factory: pytest.TempPathFactory,
) -> TrainingResult:
    """Run the smoke pipeline once; share across assertions.

    50K stratified subsample + 10 epochs + patience 3 ~= 30-60 s on
    CPU. Module-scoped so the suite trains once total (default
    function scope would re-train per test using this fixture).
    """
    out_dir = tmp_path_factory.mktemp("neural_smoke_outputs")
    report_path = out_dir / "model_b_training_report.md"
    figure_path = out_dir / "figures" / "model_b_training_curves.png"
    return train_pipeline(
        settings=isolated_settings,
        sample_size=_SMOKE_SAMPLE_SIZE,
        epochs=_SMOKE_EPOCHS,
        early_stopping_patience=_SMOKE_PATIENCE,
        random_state=_SMOKE_SEED,
        report_path=report_path,
        figure_path=figure_path,
        # 100 latency samples is plenty for the smoke; 1000 (the
        # production default) is the dominant cost driver of the
        # fixture's wall-time and unnecessary at this scale.
        n_latency_samples=100,
    )


# ---------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------


def test_smoke_completes_with_non_catastrophic_auc(smoke_result: TrainingResult) -> None:
    """50K smoke clears the 0.6 val/test AUC floor."""
    assert smoke_result.val_auc > _SMOKE_VAL_AUC_FLOOR
    assert smoke_result.test_auc > _SMOKE_VAL_AUC_FLOOR


def test_training_converged(smoke_result: TrainingResult) -> None:
    """Early stopping fired OR final-epoch AUC matches best within 1%."""
    history = smoke_result.val_auc_history
    if smoke_result.early_stopped:
        return
    # Final-epoch AUC matches best (within tolerance).
    best = max(history)
    final = history[-1]
    assert abs(final - best) / max(best, 1e-9) <= _CONVERGENCE_TOLERANCE, (
        f"Training did not converge: final AUC {final:.4f} "
        f"differs from best {best:.4f} by more than "
        f"{_CONVERGENCE_TOLERANCE * 100:.0f}% and early stopping did "
        f"not fire (epochs_run={smoke_result.epochs_run})."
    )


def test_output_files_exist(smoke_result: TrainingResult) -> None:
    """Model + manifest + report + curves figure all written."""
    assert smoke_result.model_path.is_file()
    assert smoke_result.manifest_path.is_file()
    assert smoke_result.report_path.is_file()
    assert smoke_result.figure_path.is_file()


def test_saved_model_round_trips(
    isolated_settings: Settings,
    smoke_result: TrainingResult,
) -> None:
    """Saved model loads back; `predict_proba(X)` shape `(n, 2)` ~= in-memory."""
    models_dir = isolated_settings.models_dir / _MODELS_SUBDIR
    reloaded = FraudNetModel.load(models_dir)
    val = pd.read_parquet(isolated_settings.processed_dir / "tier5_val.parquet").head(20)
    feature_cols = _select_columns_for_fraudnet(val)
    proba = reloaded.predict_proba(val[feature_cols])
    assert proba.shape == (20, 2)
    assert (proba >= 0.0).all() and (proba <= 1.0).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)


def test_oov_entities_handled(
    isolated_settings: Settings,
    smoke_result: TrainingResult,
) -> None:
    """A row with a never-seen entity value scores in [0, 1] without crashing."""
    models_dir = isolated_settings.models_dir / _MODELS_SUBDIR
    reloaded = FraudNetModel.load(models_dir)
    val = pd.read_parquet(isolated_settings.processed_dir / "tier5_val.parquet").head(5)
    feature_cols = _select_columns_for_fraudnet(val)
    val_x = val[feature_cols].copy()
    # Inject a synthetic never-seen card1 value (outside any reasonable
    # IEEE-CIS range) into the first row.
    val_x.iloc[0, val_x.columns.get_loc("card1")] = 999_999_999
    # And a never-seen DeviceInfo string into the second row.
    val_x.iloc[1, val_x.columns.get_loc("DeviceInfo")] = "__never_seen_device__"
    proba = reloaded.predict_proba(val_x)
    assert proba.shape == (5, 2)
    assert (proba >= 0.0).all() and (proba <= 1.0).all()


def test_inference_latency_under_smoke_ceiling(smoke_result: TrainingResult) -> None:
    """p95 single-row latency < 100 ms on the smoke (loose ceiling).

    FraudNet is shadow-deployable; we don't gate on the Model A 15 ms
    budget. The ceiling exists so a 10x regression in inference cost
    surfaces in CI.
    """
    assert smoke_result.latency_p95_ms < _SMOKE_LATENCY_CEILING_MS
    # Sanity: p50 <= p95 <= p99.
    assert smoke_result.latency_p50_ms <= smoke_result.latency_p95_ms
    assert smoke_result.latency_p95_ms <= smoke_result.latency_p99_ms


# ---------------------------------------------------------------------
# Focal-loss math (unit-style, lives here alongside the rest so the
# whole suite shares the smoke fixture's wall-time budget).
# ---------------------------------------------------------------------


def test_focal_loss_reduces_easy_positive_loss_vs_bce() -> None:
    """Focal loss attenuates easy-positive loss; ordering preserved.

    Per Lin et al. 2017: FL down-weights well-classified examples by
    `(1 - p_t)^gamma`. For an easy positive `(p=0.95, y=1)`, FL must
    yield smaller loss than BCE. The cross-entropy ordering of
    (easy, medium, hard) examples must still hold.
    """
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    focal = FocalLoss(alpha=0.25, gamma=2.0)

    # Logit values chosen so sigmoid(x) ~= 0.95, 0.7, 0.5.
    logits = torch.tensor([2.944, 0.847, 0.0])
    targets = torch.tensor([1.0, 1.0, 1.0])

    bce_per_row = bce(logits, targets)
    bce_easy, bce_medium, bce_hard = (float(v) for v in bce_per_row)

    # Per-row focal: re-instantiate with the per-row mean over a single
    # element to get a comparable scalar loss.
    focal_easy = float(focal(logits[0:1], targets[0:1]))
    focal_medium = float(focal(logits[1:2], targets[1:2]))
    focal_hard = float(focal(logits[2:3], targets[2:3]))

    # Easy-example loss attenuated more than 50% under FL.
    assert focal_easy < bce_easy * 0.5
    # Cross-entropy ordering preserved (hardest > medium > easy).
    assert focal_hard > focal_medium > focal_easy


def test_focal_loss_handles_unbalanced_smoke() -> None:
    """Loss is finite + non-negative on a synthetic imbalanced batch."""
    rng = np.random.default_rng(0)
    logits = torch.from_numpy(rng.normal(size=512).astype(np.float32))
    # 3.5% positive base rate to mirror IEEE-CIS.
    targets = torch.from_numpy((rng.random(512) < 0.035).astype(np.float32))
    loss = float(FocalLoss()(logits, targets))
    assert np.isfinite(loss)
    assert loss >= 0.0


def test_focal_loss_reduction_is_mean() -> None:
    """Loss for a constant-target batch matches the mean of per-row losses.

    Locks in the `reduction="mean"` contract; a switch to "sum" would
    break the LR-schedule comparability the project relies on.
    """
    logits = torch.tensor([1.0, -1.0, 2.0, -0.5])
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
    focal = FocalLoss()
    aggregate = float(focal(logits, targets))
    per_row = [float(focal(logits[i : i + 1], targets[i : i + 1])) for i in range(4)]
    assert abs(aggregate - np.mean(per_row)) < 1e-6


# ---------------------------------------------------------------------
# Wall-time smoke (cheap; bounds the suite).
# ---------------------------------------------------------------------


def test_smoke_walltime_reasonable(smoke_result: TrainingResult) -> None:
    """Suite-shared `smoke_result` fixture should finish in well under
    the 10-minute pytest default. We don't assert on time directly
    (it would be flaky across CI / dev machines); instead we verify
    that the completed run looks well-formed: epochs_run > 0 and the
    learning curve is non-trivial."""
    assert smoke_result.epochs_run > 0
    assert len(smoke_result.val_auc_history) == smoke_result.epochs_run
    assert len(smoke_result.train_loss_history) == smoke_result.epochs_run
    # The model trained for at least one epoch's worth of work, by
    # any sane definition: every val AUC is in (0, 1) and at least
    # one train loss is positive.
    assert all(0.0 <= auc <= 1.0 for auc in smoke_result.val_auc_history)
    assert any(loss > 0.0 for loss in smoke_result.train_loss_history)
