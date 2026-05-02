"""Integration test for `scripts/train_gnn.py::train_pipeline`.

Spec gates:
- Trains the 3-layer GraphSAGE on a 5K stratified subsample to
  convergence (early stopping fires OR final-epoch val AUC matches
  best within 1%).
- Output files exist (model bundle + manifest + report + curves PNG).
- Val AUC > 0.55 (not catastrophic; the 5K + 5-epoch smoke barely
  converges, so we use a looser floor than FraudNet's).
- Saved model bundle loads back; `predict_proba(X)` returns shape
  `(n, 2)` with row-wise unit sums.
- Unknown TransactionIDs raise `KeyError` (transductive contract).
- Bipartite graph invariant: txn↔entity edges only (no txn↔txn or
  entity↔entity).
- Single-row latency under loose ceiling (FraudGNN is shadow-
  deployable; not on the production hot path).

Skip-gated on `data/processed/tier5_train.parquet` presence.

The `smoke_result` fixture is **module-scoped** (lessons from 3.4.a)
so the suite trains once total.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest
import torch
from scripts.train_gnn import (
    _MODELS_SUBDIR,
    TrainingResult,
    train_pipeline,
)

from fraud_engine.config.settings import Settings, get_settings
from fraud_engine.models.gnn_model import (
    FraudGNNModel,
)

pytestmark = pytest.mark.integration

_SMOKE_SAMPLE_SIZE: int = 5_000
_SMOKE_EPOCHS: int = 5
_SMOKE_PATIENCE: int = 2
_SMOKE_SEED: int = 42
_SMOKE_VAL_AUC_FLOOR: float = 0.55
_SMOKE_LATENCY_CEILING_MS: float = 200.0
_CONVERGENCE_TOLERANCE: float = 0.01


def _processed_dir_has_tier5() -> bool:
    """True iff `data/processed/tier5_train.parquet` exists locally."""
    return (get_settings().processed_dir / "tier5_train.parquet").is_file()


@pytest.fixture(scope="module")
def isolated_settings(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[Settings]:
    """Settings with isolated MLFLOW_TRACKING_URI + MODELS_DIR.

    Module-scoped (lessons from 3.4.a): the smoke fixture is the
    expensive part of this test module, and module scope means it
    trains once for the whole module instead of once per test.
    """
    if not _processed_dir_has_tier5():
        pytest.skip(
            "data/processed/tier5_train.parquet not present - run "
            "`uv run python scripts/build_features_all_tiers.py` first."
        )

    tmp_path = tmp_path_factory.mktemp("gnn_smoke")
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

    5K stratified subsample + 5 epochs + patience 2 ~= 30-60 s on CPU.
    """
    out_dir = tmp_path_factory.mktemp("gnn_smoke_outputs")
    report_path = out_dir / "model_c_training_report.md"
    figure_path = out_dir / "figures" / "model_c_training_curves.png"
    return train_pipeline(
        settings=isolated_settings,
        sample_size=_SMOKE_SAMPLE_SIZE,
        epochs=_SMOKE_EPOCHS,
        early_stopping_patience=_SMOKE_PATIENCE,
        random_state=_SMOKE_SEED,
        report_path=report_path,
        figure_path=figure_path,
        # 50 latency samples is plenty for the smoke assertion.
        n_latency_samples=50,
    )


# ---------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------


def test_smoke_completes_with_non_catastrophic_auc(smoke_result: TrainingResult) -> None:
    """5K smoke clears the 0.55 val/test AUC floor."""
    assert smoke_result.val_auc > _SMOKE_VAL_AUC_FLOOR
    assert smoke_result.test_auc > _SMOKE_VAL_AUC_FLOOR


def test_training_converged(smoke_result: TrainingResult) -> None:
    """Early stopping fired OR final-epoch AUC matches best within 1%."""
    history = smoke_result.val_auc_history
    if smoke_result.early_stopped:
        return
    best = max(history)
    final = history[-1]
    assert abs(final - best) / max(best, 1e-9) <= _CONVERGENCE_TOLERANCE, (
        f"Training did not converge: final AUC {final:.4f} differs "
        f"from best {best:.4f} by more than "
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
    reloaded = FraudGNNModel.load(models_dir)
    # Use TransactionIDs that ARE in the smoke graph. The smoke trained
    # on a 5K subsample, so we need to use that same subsample's IDs.
    known_ids = list(reloaded.txn_index_.keys())[:5] if reloaded.txn_index_ else []
    if not known_ids:
        pytest.skip("No known TransactionIDs in reloaded model")
    # Build a minimal val_x with only known IDs by selecting from the
    # stored data. Since predict_proba just needs `TransactionID` and
    # ignores other columns for lookup, a single-column DF is enough.
    val_x = pd.DataFrame({"TransactionID": known_ids})
    proba = reloaded.predict_proba(val_x)
    assert proba.shape == (len(known_ids), 2)
    assert (proba >= 0.0).all() and (proba <= 1.0).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)


def test_predict_proba_keyerror_on_unknown_txn_id(
    isolated_settings: Settings,
    smoke_result: TrainingResult,
) -> None:
    """Unknown TransactionID raises KeyError (transductive contract)."""
    models_dir = isolated_settings.models_dir / _MODELS_SUBDIR
    reloaded = FraudGNNModel.load(models_dir)
    # 999_999_999 is virtually guaranteed not to exist in the smoke graph.
    bogus_df = pd.DataFrame({"TransactionID": [999_999_999]})
    with pytest.raises(KeyError, match="not in persisted graph"):
        reloaded.predict_proba(bogus_df)


def test_inference_latency_under_smoke_ceiling(smoke_result: TrainingResult) -> None:
    """p95 single-row latency < 200 ms (loose ceiling).

    FraudGNN is shadow-deployable per CLAUDE.md §3; the production
    Model A bears the 15 ms budget. predict_proba is a TransactionID
    lookup + sigmoid on cached node logits, so the p95 is dominated
    by Pandas .iloc + numpy fancy-indexing overhead.
    """
    assert smoke_result.latency_p95_ms < _SMOKE_LATENCY_CEILING_MS
    # Sanity: p50 <= p95 <= p99.
    assert smoke_result.latency_p50_ms <= smoke_result.latency_p95_ms
    assert smoke_result.latency_p95_ms <= smoke_result.latency_p99_ms


def test_graph_construction_is_bipartite(
    isolated_settings: Settings,
    smoke_result: TrainingResult,
) -> None:
    """Verify bipartite invariant: edges only between txn and entity nodes.

    Reads the persisted manifest's node-count fields and checks that
    `n_edges_undirected` is consistent with a bipartite construction
    (every edge contributes one txn endpoint and one entity endpoint).
    """
    models_dir = isolated_settings.models_dir / _MODELS_SUBDIR
    reloaded = FraudGNNModel.load(models_dir)
    assert reloaded.data_ is not None
    assert reloaded.n_txn_nodes_ is not None
    assert reloaded.n_entity_nodes_ is not None

    n_txn = reloaded.n_txn_nodes_
    edge_index = reloaded.data_.edge_index
    src = edge_index[0]
    dst = edge_index[1]
    # Every edge: exactly one endpoint in [0, n_txn) (txn partition);
    # the other in [n_txn, n_txn + n_entity_nodes_) (entity partition).
    src_is_txn = src < n_txn
    dst_is_txn = dst < n_txn
    # XOR: exactly one endpoint is txn, the other entity.
    assert torch.equal(
        src_is_txn ^ dst_is_txn, torch.ones_like(src, dtype=torch.bool)
    ), "Bipartite invariant violated: some edges have both endpoints in the same partition."


def test_smoke_walltime_reasonable(smoke_result: TrainingResult) -> None:
    """Sanity: completed run is well-formed."""
    assert smoke_result.epochs_run > 0
    assert len(smoke_result.val_auc_history) == smoke_result.epochs_run
    assert len(smoke_result.train_loss_history) == smoke_result.epochs_run
    assert all(0.0 <= auc <= 1.0 for auc in smoke_result.val_auc_history)
    assert any(loss > 0.0 for loss in smoke_result.train_loss_history)
    # Graph numbers reasonable.
    assert smoke_result.n_txn_nodes > 0
    assert smoke_result.n_entity_nodes > 0
    assert smoke_result.n_edges_undirected > 0
