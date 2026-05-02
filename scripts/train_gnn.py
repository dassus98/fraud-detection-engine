"""Sprint 3 FraudGNN (Model C) training pipeline.

End-to-end: load `data/processed/tier5_*.parquet` -> stratified
subsample (default 50K to fit the spec's "trains on 50K to convergence"
target and the 4-hour cap) -> fit `FraudGNNModel` (3-layer GraphSAGE
+ neighbor sampling + focal loss + early stopping) -> evaluate on
val + test -> save the model bundle to `models/sprint3/fraudgnn/` ->
emit `reports/model_c_training_report.md` and a per-epoch
learning-curve PNG.

Business rationale:
    Model C is the diversity GNN. The full training pipeline produces
    the artefact Sprint 4's evaluator and Sprint 5's batch-feature
    pipeline load. The `--quick` flag is the same code path as
    production (only sample_size / max_epochs / patience differ),
    so the integration test exercises the production path.

Trade-offs considered:
    - **Default 50K subsample, not full data.** Spec authorises
      deferral if runtime > 4 h; defaulting to 50K (matching 3.4.a)
      keeps wall-time well under and matches the integration test's
      smoke target. `--full` flag exposed for users with stable
      training environments; documented as time-risky.
    - **No calibration in this script.** 3.3.c toolkit reusable;
      Sprint 4 ensemble blend is most legible on raw probabilities.
    - **Single MLflow run** (`model_c_train`); per-epoch metrics
      logged with `step=epoch_idx`.
    - **Latency reporting, not gating.** Per CLAUDE.md §3 FraudGNN
      is batch-only — Model A bears the 15 ms hot-path budget.

Usage:
    uv run python scripts/train_gnn.py            # default 50K subsample
    uv run python scripts/train_gnn.py --quick    # 5K smoke (~30-60 s)
    uv run python scripts/train_gnn.py --full     # full data (~30-90 min on CPU)
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import Any, Final

import click
import matplotlib

matplotlib.use("Agg")  # headless; no display server

import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402

from fraud_engine.config.settings import Settings, get_settings  # noqa: E402
from fraud_engine.evaluation.calibration import (  # noqa: E402
    brier_score,
    expected_calibration_error,
    log_loss,
)
from fraud_engine.models.gnn_model import FraudGNNModel  # noqa: E402
from fraud_engine.utils.logging import get_logger  # noqa: E402
from fraud_engine.utils.mlflow_setup import (  # noqa: E402
    configure_mlflow,
    setup_experiment,
)

_logger = get_logger(__name__)

# ---------------------------------------------------------------------
# Module constants.
# ---------------------------------------------------------------------

# Output paths.
_MODELS_SUBDIR: Final[str] = "sprint3/fraudgnn"
_TRAINING_REPORT_PATH: Final[Path] = (
    Path(__file__).resolve().parents[1] / "reports" / "model_c_training_report.md"
)
_TRAINING_CURVES_PATH: Final[Path] = (
    Path(__file__).resolve().parents[1] / "reports" / "figures" / "model_c_training_curves.png"
)

# Defaults (mirror gnn_model module constants).
_DEFAULT_EPOCHS: Final[int] = 20
_DEFAULT_BATCH_SIZE: Final[int] = 1024
_DEFAULT_LR: Final[float] = 1e-3
_DEFAULT_HIDDEN_DIM: Final[int] = 64
_DEFAULT_NUM_NEIGHBORS_STR: Final[str] = "10,10,10"
_DEFAULT_EARLY_STOPPING_PATIENCE: Final[int] = 5

# Default sample size (matches 3.4.a's spec target; --full overrides).
_DEFAULT_SAMPLE_SIZE: Final[int] = 50_000

# Latency measurement.
_LATENCY_N_SAMPLES: Final[int] = 100

# `--quick` overrides.
_SMOKE_SAMPLE_SIZE: Final[int] = 5_000
_SMOKE_EPOCHS: Final[int] = 5
_SMOKE_PATIENCE: Final[int] = 2

# MLflow run naming.
_MLFLOW_RUN_NAME: Final[str] = "model_c_train"

# Acceptance gates referenced in the auto-generated report.
_VAL_AUC_FLOOR: Final[float] = 0.5
_CONVERGENCE_TOLERANCE: Final[float] = 0.01

# Float-precision for AUC reporting.
_AUC_DIGITS: Final[int] = 4


# ---------------------------------------------------------------------
# Result dataclass.
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class TrainingResult:
    """Metric + artifact bundle returned by `train_pipeline`."""

    val_auc: float
    test_auc: float
    val_pr_auc: float
    test_pr_auc: float
    val_log_loss: float
    test_log_loss: float
    val_brier: float
    test_brier: float
    val_ece: float
    test_ece: float
    best_epoch: int
    epochs_run: int
    early_stopped: bool
    n_train_rows: int
    n_val_rows: int
    n_test_rows: int
    n_numeric: int
    n_txn_nodes: int
    n_entity_nodes: int
    n_edges_undirected: int
    n_params: int
    val_auc_history: tuple[float, ...]
    train_loss_history: tuple[float, ...]
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    model_path: Path
    manifest_path: Path
    report_path: Path
    figure_path: Path


# ---------------------------------------------------------------------
# Data loading + feature selection.
# ---------------------------------------------------------------------


def _load_split(processed_dir: Path, name: str) -> pd.DataFrame:
    """Read one tier-5 parquet (`train` / `val` / `test`)."""
    path = processed_dir / f"tier5_{name}.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"Expected tier-5 parquet at {path} - run "
            f"`uv run python scripts/build_features_all_tiers.py` first."
        )
    return pd.read_parquet(path)


def _stratified_subsample(df: pd.DataFrame, target_n: int, seed: int = 42) -> pd.DataFrame:
    """Stratified subsample to ~`target_n` rows by `isFraud`."""
    if len(df) <= target_n:
        return df.reset_index(drop=True)
    kept, _ = train_test_split(
        df,
        train_size=target_n,
        stratify=df["isFraud"],
        random_state=seed,
    )
    return kept.reset_index(drop=True)


# ---------------------------------------------------------------------
# Inference latency.
# ---------------------------------------------------------------------


def _measure_inference_latency(
    model: FraudGNNModel,
    sample_X: pd.DataFrame,  # noqa: N803
    n_samples: int = _LATENCY_N_SAMPLES,
    seed: int = 42,
) -> tuple[float, float, float, np.ndarray[Any, Any]]:
    """Time single-row `predict_proba` calls. Return p50/p95/p99 + raw array.

    Per the FraudGNNModel contract, predict_proba is a TransactionID
    lookup + sigmoid on the cached node logits — no per-call forward
    pass. The latency reflects the lookup cost (very small).
    """
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(sample_X), size=n_samples)
    latencies_ms: list[float] = []
    for idx in indices:
        row = sample_X.iloc[[idx]]
        t0 = time.perf_counter()
        _ = model.predict_proba(row)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(latencies_ms, dtype=np.float64)
    p50, p95, p99 = (float(v) for v in np.percentile(arr, [50, 95, 99]))
    return p50, p95, p99, arr


def _save_training_curves(
    val_auc_history: list[float],
    train_loss_history: list[float],
    out_path: Path,
) -> None:
    """Render val AUC + train loss per epoch on dual y-axes."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    epochs = list(range(1, len(val_auc_history) + 1))
    ax1.plot(epochs, val_auc_history, color="#3a72b0", marker="o", label="val AUC")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("val ROC-AUC", color="#3a72b0")
    ax1.tick_params(axis="y", labelcolor="#3a72b0")
    ax2 = ax1.twinx()
    ax2.plot(epochs, train_loss_history, color="#c14242", marker="x", label="train focal loss")
    ax2.set_ylabel("train focal loss", color="#c14242")
    ax2.tick_params(axis="y", labelcolor="#c14242")
    ax1.set_title(f"FraudGNN training curves (n_epochs = {len(epochs)})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Report rendering.
# ---------------------------------------------------------------------


def _render_training_report(  # noqa: PLR0915 — linear markdown builder
    *,
    result: TrainingResult,
    out_path: Path,
) -> None:
    """Write the auto-generated training report to `out_path`."""
    val_auc_floor_gate = result.val_auc > _VAL_AUC_FLOOR
    convergence_gate = result.early_stopped or (
        len(result.val_auc_history) > 1
        and abs(result.val_auc_history[-1] - max(result.val_auc_history))
        / max(result.val_auc_history)
        <= _CONVERGENCE_TOLERANCE
    )

    lines: list[str] = []
    lines.append("# Model C - FraudGNN training report")
    lines.append("")
    lines.append("- **Generated by:** `scripts/train_gnn.py`")
    lines.append(f"- **Train txns:** {result.n_train_rows:,}")
    lines.append(f"- **Val txns:** {result.n_val_rows:,}")
    lines.append(f"- **Test txns:** {result.n_test_rows:,}")
    lines.append(f"- **Numeric features:** {result.n_numeric:,}")
    lines.append(
        f"- **Total nodes (txn + entity):** {result.n_txn_nodes + result.n_entity_nodes:,}"
    )
    lines.append(f"- **Entity nodes:** {result.n_entity_nodes:,}")
    lines.append(f"- **Edges (undirected):** {result.n_edges_undirected:,}")
    lines.append(f"- **Trainable params:** {result.n_params:,}")
    lines.append("")
    lines.append("## Headline metrics")
    lines.append("")
    lines.append("| Metric | Val | Test |")
    lines.append("|---|---|---|")
    lines.append(
        f"| ROC-AUC | {result.val_auc:.{_AUC_DIGITS}f} | {result.test_auc:.{_AUC_DIGITS}f} |"
    )
    lines.append(
        f"| PR-AUC | {result.val_pr_auc:.{_AUC_DIGITS}f} "
        f"| {result.test_pr_auc:.{_AUC_DIGITS}f} |"
    )
    lines.append(f"| Log loss | {result.val_log_loss:.6f} | {result.test_log_loss:.6f} |")
    lines.append(f"| Brier | {result.val_brier:.6f} | {result.test_brier:.6f} |")
    lines.append(f"| ECE | {result.val_ece:.4f} | {result.test_ece:.4f} |")
    lines.append("")
    lines.append("## Acceptance gates")
    lines.append("")
    lines.append("| Gate | Status |")
    lines.append("|---|---|")
    lines.append(
        f"| Val AUC > 0.5 (catastrophic floor) | "
        f"{'PASS' if val_auc_floor_gate else 'FAIL'} (val_auc = {result.val_auc:.4f}) |"
    )
    lines.append(
        f"| Training converged (early-stop fired or AUC plateaued) | "
        f"{'PASS' if convergence_gate else 'FAIL'} "
        f"(early_stopped = {result.early_stopped}, "
        f"epochs_run = {result.epochs_run}, best_epoch = {result.best_epoch}) |"
    )
    lines.append("")
    lines.append("## Inference latency")
    lines.append("")
    lines.append(
        "Single-row `predict_proba` (TransactionID lookup -> cached "
        "node logit -> sigmoid). Per CLAUDE.md §3 FraudGNN is "
        "batch-only; no production-path latency budget."
    )
    lines.append("")
    lines.append("| Quantile | Latency (ms) |")
    lines.append("|---|---|")
    lines.append(f"| p50 | {result.latency_p50_ms:.3f} |")
    lines.append(f"| p95 | {result.latency_p95_ms:.3f} |")
    lines.append(f"| p99 | {result.latency_p99_ms:.3f} |")
    lines.append("")
    lines.append("## Architecture")
    lines.append("")
    lines.append("- 3-layer GraphSAGE (`SAGEConv`) + Linear head; per spec.")
    lines.append("- Mean aggregator; bipartite undirected graph (txn ↔ entity).")
    lines.append("- Neighbor sampling per layer via `NeighborLoader`.")
    lines.append("- Focal loss (α=0.25, γ=2.0) imported from `neural_model.FocalLoss`.")
    lines.append("- Float32 throughout the numeric pipeline.")
    lines.append("")
    lines.append("## Training curves")
    lines.append("")
    lines.append(
        f"- Best epoch: **{result.best_epoch}** (val AUC = "
        f"{max(result.val_auc_history) if result.val_auc_history else 0.0:.{_AUC_DIGITS}f})"
    )
    lines.append(f"- Epochs run: {result.epochs_run}")
    lines.append(f"- Early stopped: {result.early_stopped}")
    lines.append(f"- See `{result.figure_path}` for per-epoch curves.")
    lines.append("")
    lines.append("## Per-epoch history")
    lines.append("")
    lines.append("| Epoch | Train focal loss | Val ROC-AUC |")
    lines.append("|---|---|---|")
    for i, (loss, auc) in enumerate(
        zip(result.train_loss_history, result.val_auc_history, strict=True), start=1
    ):
        lines.append(f"| {i} | {loss:.6f} | {auc:.{_AUC_DIGITS}f} |")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Model: `{result.model_path}`")
    lines.append(f"- Manifest: `{result.manifest_path}`")
    lines.append(f"- This report: `{result.report_path}`")
    lines.append(f"- Training curves: `{result.figure_path}`")
    lines.append("")
    lines.append("## Out of scope (Sprint 4 territory)")
    lines.append("")
    lines.append("- Calibration of FraudGNN outputs (3.3.c toolkit reusable).")
    lines.append("- Ensemble with Models A + B; cost-curve evaluation.")
    lines.append("- Inductive scoring of new TransactionIDs (Sprint 5+).")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------


def train_pipeline(  # noqa: PLR0913, PLR0915 — single-file orchestration; the knobs match the CLI
    *,
    settings: Settings,
    epochs: int = _DEFAULT_EPOCHS,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    lr: float = _DEFAULT_LR,
    hidden_dim: int = _DEFAULT_HIDDEN_DIM,
    num_neighbors: tuple[int, ...] = (10, 10, 10),
    early_stopping_patience: int = _DEFAULT_EARLY_STOPPING_PATIENCE,
    sample_size: int | None = _DEFAULT_SAMPLE_SIZE,
    random_state: int | None = None,
    report_path: Path = _TRAINING_REPORT_PATH,
    figure_path: Path = _TRAINING_CURVES_PATH,
    n_latency_samples: int = _LATENCY_N_SAMPLES,
) -> TrainingResult:
    """End-to-end fit -> evaluate -> save -> report.

    Args:
        settings: Project settings (paths + seeds).
        epochs: Cap on training epochs. Default 20.
        batch_size: NeighborLoader root-node batch size. Default 1024.
        lr: Adam learning rate. Default 1e-3.
        hidden_dim: SAGEConv hidden width. Default 64.
        num_neighbors: Per-layer neighbor fan-out (length must equal
            num_layers). Default (10, 10, 10).
        early_stopping_patience: Halt after N epochs without val-AUC
            improvement. Default 5.
        sample_size: If not None, stratified-subsample train/val/test
            to ~this many rows (val/test get sample_size//5). Default
            50_000 (the spec target adapted from 3.4.a).
            Pass None to use the full data (`--full` flag).
        random_state: Seed for torch + numpy + the subsampling.
        report_path: Where to write the training-report markdown.
        figure_path: Where to write the training-curves PNG.
        n_latency_samples: How many single-row predict_proba calls to
            time. Default 100.

    Returns:
        TrainingResult bundle with all metrics + artifact paths.
    """
    seed = random_state if random_state is not None else settings.seed

    # --- Load + optionally subsample ---
    train = _load_split(settings.processed_dir, "train")
    val = _load_split(settings.processed_dir, "val")
    test = _load_split(settings.processed_dir, "test")
    if sample_size is not None:
        train = _stratified_subsample(train, sample_size, seed=seed)
        val = _stratified_subsample(val, max(sample_size // 5, 200), seed=seed)
        test = _stratified_subsample(test, max(sample_size // 5, 200), seed=seed)

    train_y = train["isFraud"].to_numpy()
    val_y = val["isFraud"].to_numpy()
    test_y = test["isFraud"].to_numpy()

    _logger.info(
        "train_gnn.loaded",
        n_train=len(train),
        n_val=len(val),
        n_test=len(test),
        sample_size=sample_size,
    )

    # --- Fit ---
    configure_mlflow()
    experiment_id = setup_experiment()
    mlflow.set_experiment(experiment_id=experiment_id)

    models_dir = settings.models_dir / _MODELS_SUBDIR
    models_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(experiment_id=experiment_id, run_name=_MLFLOW_RUN_NAME):
        mlflow.set_tag("stage", "sprint3_train_modelc")
        mlflow.set_tag("model_class", "FraudGNNModel")
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("lr", lr)
        mlflow.log_param("hidden_dim", hidden_dim)
        mlflow.log_param("num_neighbors", ",".join(str(n) for n in num_neighbors))
        mlflow.log_param("early_stopping_patience", early_stopping_patience)
        mlflow.log_param("n_train_rows", int(len(train)))
        mlflow.log_param("n_val_rows", int(len(val)))
        mlflow.log_param("n_test_rows", int(len(test)))
        mlflow.log_param("sample_size", sample_size if sample_size is not None else "full")

        model = FraudGNNModel(
            hidden_dim=hidden_dim,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            max_epochs=epochs,
            lr=lr,
            early_stopping_patience=early_stopping_patience,
            random_state=seed,
        )
        model.fit(train, train_y, val, val_y, X_test=test, y_test=test_y)

        # --- Per-epoch metrics into MLflow ---
        for epoch_idx, (loss, auc) in enumerate(
            zip(model.train_loss_history_, model.val_auc_history_, strict=True),
            start=1,
        ):
            mlflow.log_metric("train_focal_loss", float(loss), step=epoch_idx)
            mlflow.log_metric("val_auc_per_epoch", float(auc), step=epoch_idx)

        # --- Score val + test ---
        val_p = model.predict_proba(val)[:, 1]
        test_p = model.predict_proba(test)[:, 1]
        val_auc = float(roc_auc_score(val_y, val_p))
        test_auc = float(roc_auc_score(test_y, test_p))
        val_pr_auc = float(average_precision_score(val_y, val_p))
        test_pr_auc = float(average_precision_score(test_y, test_p))
        val_ll = log_loss(val_y, val_p)
        test_ll = log_loss(test_y, test_p)
        val_brier = brier_score(val_y, val_p)
        test_brier = brier_score(test_y, test_p)
        val_ece = expected_calibration_error(val_y, val_p)
        test_ece = expected_calibration_error(test_y, test_p)

        mlflow.log_metric("val_auc", val_auc)
        mlflow.log_metric("test_auc", test_auc)
        mlflow.log_metric("val_pr_auc", val_pr_auc)
        mlflow.log_metric("test_pr_auc", test_pr_auc)
        mlflow.log_metric("val_log_loss", val_ll)
        mlflow.log_metric("test_log_loss", test_ll)
        mlflow.log_metric("val_brier", val_brier)
        mlflow.log_metric("test_brier", test_brier)
        mlflow.log_metric("val_ece", val_ece)
        mlflow.log_metric("test_ece", test_ece)
        mlflow.log_metric("best_epoch", float(model.best_epoch_ or 0))
        mlflow.log_metric("epochs_run", float(len(model.val_auc_history_)))

        _logger.info(
            "train_gnn.metrics",
            val_auc=val_auc,
            test_auc=test_auc,
            val_pr_auc=val_pr_auc,
            best_epoch=model.best_epoch_,
            epochs_run=len(model.val_auc_history_),
            early_stopped=model.early_stopped_,
        )

        # --- Save ---
        model_path, manifest_path = model.save(models_dir)

        # --- Latency ---
        p50, p95, p99, _latencies = _measure_inference_latency(
            model, val, n_samples=n_latency_samples, seed=seed
        )
        mlflow.log_metric("inference_p50_ms", p50)
        mlflow.log_metric("inference_p95_ms", p95)
        mlflow.log_metric("inference_p99_ms", p99)

        # --- Curves figure + report ---
        _save_training_curves(model.val_auc_history_, model.train_loss_history_, figure_path)
        n_params = sum(
            p.numel() for p in (model.module_.parameters() if model.module_ is not None else [])
        )

        result = TrainingResult(
            val_auc=val_auc,
            test_auc=test_auc,
            val_pr_auc=val_pr_auc,
            test_pr_auc=test_pr_auc,
            val_log_loss=val_ll,
            test_log_loss=test_ll,
            val_brier=val_brier,
            test_brier=test_brier,
            val_ece=val_ece,
            test_ece=test_ece,
            best_epoch=int(model.best_epoch_ or 0),
            epochs_run=len(model.val_auc_history_),
            early_stopped=bool(model.early_stopped_),
            n_train_rows=int(len(train)),
            n_val_rows=int(len(val)),
            n_test_rows=int(len(test)),
            n_numeric=len(model.numeric_cols_ or []),
            n_txn_nodes=int(model.n_txn_nodes_ or 0),
            n_entity_nodes=int(model.n_entity_nodes_ or 0),
            n_edges_undirected=int(model.n_edges_ or 0),
            n_params=int(n_params),
            val_auc_history=tuple(model.val_auc_history_),
            train_loss_history=tuple(model.train_loss_history_),
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            latency_p99_ms=p99,
            model_path=model_path,
            manifest_path=manifest_path,
            report_path=report_path,
            figure_path=figure_path,
        )

        _render_training_report(result=result, out_path=report_path)
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(manifest_path))
        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(figure_path))

    return result


# ---------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------


def _parse_num_neighbors(value: str) -> tuple[int, ...]:
    """Parse comma-separated `--num-neighbors` value, e.g. `10,10,10`."""
    return tuple(int(v.strip()) for v in value.split(",") if v.strip())


@click.command()
@click.option("--epochs", type=int, default=_DEFAULT_EPOCHS, show_default=True)
@click.option("--batch-size", type=int, default=_DEFAULT_BATCH_SIZE, show_default=True)
@click.option("--lr", type=float, default=_DEFAULT_LR, show_default=True)
@click.option("--hidden-dim", type=int, default=_DEFAULT_HIDDEN_DIM, show_default=True)
@click.option(
    "--num-neighbors",
    type=str,
    default=_DEFAULT_NUM_NEIGHBORS_STR,
    show_default=True,
    help="Comma-separated per-layer fan-out, e.g. '10,10,10'.",
)
@click.option(
    "--early-stopping-patience",
    type=int,
    default=_DEFAULT_EARLY_STOPPING_PATIENCE,
    show_default=True,
)
@click.option(
    "--quick",
    is_flag=True,
    default=False,
    help="Run a smoke: 5K stratified subsample + 5 epochs.",
)
@click.option(
    "--full",
    is_flag=True,
    default=False,
    help="Use the full dataset. Documented as time-risky (may exceed 4 h on CPU).",
)
def main(  # noqa: PLR0913 — Click decorator-driven; each kwarg is a CLI flag
    epochs: int,
    batch_size: int,
    lr: float,
    hidden_dim: int,
    num_neighbors: str,
    early_stopping_patience: int,
    quick: bool,
    full: bool,
) -> None:
    """Run the FraudGNN training pipeline."""
    if quick and full:
        raise click.UsageError("--quick and --full are mutually exclusive")
    settings = get_settings()
    settings.ensure_directories()

    sample_size: int | None = _DEFAULT_SAMPLE_SIZE
    if quick:
        sample_size = _SMOKE_SAMPLE_SIZE
        epochs = _SMOKE_EPOCHS
        early_stopping_patience = _SMOKE_PATIENCE
    elif full:
        sample_size = None  # use entire dataset

    nn_tuple = _parse_num_neighbors(num_neighbors)

    result = train_pipeline(
        settings=settings,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        hidden_dim=hidden_dim,
        num_neighbors=nn_tuple,
        early_stopping_patience=early_stopping_patience,
        sample_size=sample_size,
    )

    click.echo("\ntrain_gnn: COMPLETE")
    click.echo(f"  val_auc          = {result.val_auc:.{_AUC_DIGITS}f}")
    click.echo(f"  test_auc         = {result.test_auc:.{_AUC_DIGITS}f}")
    click.echo(f"  val_pr_auc       = {result.val_pr_auc:.{_AUC_DIGITS}f}")
    click.echo(f"  best_epoch       = {result.best_epoch} / {result.epochs_run}")
    click.echo(f"  early_stopped    = {result.early_stopped}")
    click.echo(f"  n_txn / n_entity = {result.n_txn_nodes:,} / {result.n_entity_nodes:,}")
    click.echo(f"  n_edges          = {result.n_edges_undirected:,}")
    click.echo(
        f"  latency p50/p95/p99 = "
        f"{result.latency_p50_ms:.3f}/{result.latency_p95_ms:.3f}/"
        f"{result.latency_p99_ms:.3f} ms"
    )
    click.echo(f"  model            = {result.model_path}")
    click.echo(f"  report           = {result.report_path}")


if __name__ == "__main__":
    main()
