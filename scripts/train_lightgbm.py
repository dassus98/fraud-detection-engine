"""Sprint 3 LightGBM training pipeline (the canonical entry point).

End-to-end: load `data/processed/tier5_*.parquet` → run a 100-trial
Optuna sweep (3.3.b harness) → fit a final `LightGBMFraudModel`
(3.3.a) on train with early stopping on val → calibrate via
`select_calibration_method` (3.3.c) → evaluate on val + test →
save the model + calibrator → time inference latency → emit
`reports/model_a_training_report.md` and a latency-histogram PNG.
Single canonical script; downstream sprints (Sprint 4 cost-curve,
Sprint 5 serving) read from `models/sprint3/lightgbm_model.joblib`.

Business rationale:
    Sprint 3's Tier-5 default-hparam val AUC of 0.7689 sits well
    below the spec's 0.93 target. The shuffled-labels leak gates
    confirmed the feature pipeline is correct, so the gap is a
    tuning problem. This script runs the canonical sweep + final
    fit + calibration in one shot, producing the model artifact
    every downstream sprint reads from.

Trade-offs considered:
    - **Two top-level MLflow runs (tuning + final-fit) instead of
      one nested hierarchy.** `run_tuning` from 3.3.b already opens
      its own `mlflow.start_run` for the tuning study; nesting that
      under a wrapping outer run would require an API change to
      3.3.b. Two top-level runs in the same experiment make the UI
      "Compare runs" view immediately productive without breaking
      the 3.3.b contract.
    - **`run_tuning` runs every invocation by default.** The
      `--skip-tuning` flag reads the existing
      `configs/model_best_params.yaml` instead — useful when
      iterating on the final-fit / calibration / report steps
      without re-running a 50-90 min sweep.
    - **`--quick` flag (n_trials=5 + 5k row subset)** for the
      integration test smoke. The full run is gated by `--n-trials`
      override (default 100) without a separate code path.
    - **Latency measurement uses single-row `predict_proba +
      calibrator.transform`** — the full Sprint-5 inference path.
      Measuring just `predict_proba` would understate; measuring
      both gives the operationally relevant p50 / p95 / p99.
    - **Acceptance gates are reported, not enforced.** The script
      runs to completion regardless of whether val AUC ≥ 0.93;
      the report carries the realised number and flags whether
      the gate is met. The integration test enforces only the
      catastrophic floor (val AUC > 0.5 on the smoke subset).

Usage:
    uv run python scripts/train_lightgbm.py                  # full sweep
    uv run python scripts/train_lightgbm.py --skip-tuning    # reuse YAML
    uv run python scripts/train_lightgbm.py --quick          # smoke
    uv run python scripts/train_lightgbm.py --n-trials 50    # half sweep
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import Any, Final, cast

import click
import joblib
import matplotlib

matplotlib.use("Agg")  # headless; no display server

import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402

from fraud_engine.config.settings import Settings, get_settings  # noqa: E402
from fraud_engine.evaluation.calibration import (  # noqa: E402
    Calibrator,
    brier_score,
    expected_calibration_error,
    log_loss,
    select_calibration_method,
)
from fraud_engine.models.lightgbm_model import LightGBMFraudModel  # noqa: E402
from fraud_engine.models.tuning import run_tuning  # noqa: E402
from fraud_engine.utils.logging import get_logger  # noqa: E402
from fraud_engine.utils.mlflow_setup import (  # noqa: E402
    configure_mlflow,
    setup_experiment,
)

_logger = get_logger(__name__)

# ---------------------------------------------------------------------
# Module constants.
# ---------------------------------------------------------------------

# Output filenames + paths.
_MODELS_SUBDIR: Final[str] = "sprint3"
_MODEL_FILENAME: Final[str] = "lightgbm_model.joblib"
_CALIBRATOR_FILENAME: Final[str] = "calibrator.joblib"
_BEST_PARAMS_YAML: Final[str] = "model_best_params.yaml"
_TRAINING_REPORT_PATH: Final[Path] = (
    Path(__file__).resolve().parents[1] / "reports" / "model_a_training_report.md"
)
_LATENCY_FIGURE_PATH: Final[Path] = (
    Path(__file__).resolve().parents[1] / "reports" / "figures" / "model_a_latency.png"
)

# Tuning defaults; see 3.3.b plan + spec.
_DEFAULT_N_TRIALS: Final[int] = 100
_DEFAULT_NUM_BOOST_ROUND: Final[int] = 1000
_DEFAULT_EARLY_STOPPING_ROUNDS: Final[int] = 50

# Latency measurement.
_LATENCY_N_SAMPLES: Final[int] = 1000
_LATENCY_P95_BUDGET_MS: Final[float] = 15.0

# Smoke (`--quick`) overrides.
_SMOKE_N_TRIALS: Final[int] = 5
_SMOKE_SAMPLE_SIZE: Final[int] = 5_000
_SMOKE_NUM_BOOST_ROUND: Final[int] = 30
_SMOKE_EARLY_STOPPING_ROUNDS: Final[int] = 5

# Acceptance gates.
_VAL_AUC_GATE: Final[float] = 0.93

# Non-feature columns (mirrors `build_features_all_tiers.py`).
_NON_FEATURE_COLS: Final[frozenset[str]] = frozenset(
    {"TransactionID", "TransactionDT", "isFraud", "timestamp"}
)

# Number of feature-importance rows in the report.
_TOP_FEATURE_COUNT: Final[int] = 50

# Number of top trials to render in the trial-history table.
_TOP_TRIALS_COUNT: Final[int] = 10

# MLflow run / experiment naming.
_MLFLOW_RUN_NAME_TUNE: Final[str] = "model_a_tuning"
_MLFLOW_RUN_NAME_FIT: Final[str] = "model_a_train"

# Float-precision for AUC reporting.
_AUC_DIGITS: Final[int] = 4


# ---------------------------------------------------------------------
# Result dataclass.
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class TrainingResult:
    """Bundle returned by `train_pipeline` for downstream / test consumption.

    Captures every metric + path the report renders. Frozen + slotted
    so consumers can't accidentally mutate it.
    """

    best_params: dict[str, Any]
    best_value: float
    n_trials: int
    val_auc: float
    val_pr_auc: float
    val_log_loss_uncalibrated: float
    val_log_loss_calibrated: float
    val_brier_uncalibrated: float
    val_brier_calibrated: float
    val_ece_uncalibrated: float
    val_ece_calibrated: float
    test_auc: float
    test_pr_auc: float
    test_log_loss_uncalibrated: float
    test_log_loss_calibrated: float
    test_brier_uncalibrated: float
    test_brier_calibrated: float
    test_ece_uncalibrated: float
    test_ece_calibrated: float
    calibration_method: str
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    n_features: int
    n_train_rows: int
    n_val_rows: int
    n_test_rows: int
    model_path: Path
    calibrator_path: Path
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
            f"Expected tier-5 parquet at {path} — run "
            f"`uv run python scripts/build_features_all_tiers.py` first."
        )
    return pd.read_parquet(path)


def _select_features(df: pd.DataFrame) -> list[str]:
    """Return the LightGBM-ingestable subset of columns.

    Mirrors `scripts/build_features_all_tiers.py:_select_lgbm_features`
    exactly — drop non-feature columns and any object/string-dtype
    columns (provider/tld would need explicit categorical-feature
    enumeration; not in scope here).
    """
    return [
        col
        for col in df.columns
        if col not in _NON_FEATURE_COLS
        and not pd.api.types.is_object_dtype(df[col])
        and not pd.api.types.is_string_dtype(df[col])
    ]


# ---------------------------------------------------------------------
# Tuning + best-params YAML I/O.
# ---------------------------------------------------------------------


def _resolve_best_params_yaml() -> Path:
    """Path to `configs/model_best_params.yaml` resolved against project root."""
    return Path(__file__).resolve().parents[1] / "configs" / _BEST_PARAMS_YAML


def _read_best_params_yaml(path: Path) -> tuple[dict[str, Any], float, int, str]:
    """Return ``(best_params, best_value, n_trials, study_name)`` from the YAML."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return (
        dict(payload["best_params"]),
        float(payload["best_value"]),
        int(payload["n_trials"]),
        str(payload["study_name"]),
    )


# ---------------------------------------------------------------------
# Inference + latency.
# ---------------------------------------------------------------------


def _measure_inference_latency(
    model: LightGBMFraudModel,
    calibrator: Calibrator,
    sample_X: pd.DataFrame,  # noqa: N803 — sklearn convention; matches LightGBMFraudModel.predict_proba signature
    n_samples: int = _LATENCY_N_SAMPLES,
    seed: int = 42,
) -> tuple[float, float, float, np.ndarray[Any, Any]]:
    """Time `predict_proba → calibrator.transform` over `n_samples` single rows.

    Operates against the FULL inference path Sprint 5 will deploy.
    Returns p50 / p95 / p99 in milliseconds plus the raw latency
    array (for the histogram).
    """
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(sample_X), size=n_samples)
    latencies_ms: list[float] = []
    for idx in indices:
        row = sample_X.iloc[[idx]]
        t0 = time.perf_counter()
        proba = model.predict_proba(row)[:, 1]
        _ = calibrator.transform(proba)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(latencies_ms, dtype=np.float64)
    p50, p95, p99 = (float(v) for v in np.percentile(arr, [50, 95, 99]))
    return p50, p95, p99, arr


def _save_latency_histogram(latencies_ms: np.ndarray[Any, Any], out_path: Path) -> None:
    """Render the latency distribution + p95 vertical line + 15 ms budget line."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(latencies_ms, bins=60, color="#3a72b0", edgecolor="white")
    p95 = float(np.percentile(latencies_ms, 95))
    ax.axvline(p95, color="#c14242", linestyle="--", label=f"p95 = {p95:.2f} ms")
    ax.axvline(
        _LATENCY_P95_BUDGET_MS,
        color="#3a8b3a",
        linestyle=":",
        label=f"budget = {_LATENCY_P95_BUDGET_MS:.0f} ms",
    )
    ax.set_xlabel("single-row inference latency (ms)")
    ax.set_ylabel("count")
    ax.set_title(
        f"Model A inference latency  "
        f"(n = {len(latencies_ms):,};  predict_proba + calibrator.transform)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Feature-importance interpretation.
# ---------------------------------------------------------------------


def _interpret_feature(name: str) -> str:  # noqa: PLR0911, PLR0912 — exhaustive if-elif chain by design; one branch per recognised column-name pattern, kept linear for readability over fan-out into a dispatch dict
    """Map column name → 1-line business interpretation for the report.

    Looks for tier-specific suffixes / prefixes; falls back to "raw"
    for unrecognised columns. Best-effort; not exhaustive.
    """
    if name.endswith("_target_enc"):
        return "Tier-2 OOF target encoding"
    if "_fraud_v_ewm_lambda_" in name:
        return "Tier-4 EWM fraud-weighted velocity (OOF-safe)"
    if "_v_ewm_lambda_" in name:
        return "Tier-4 EWM velocity"
    if name.startswith("entity_degree_"):
        return "Tier-5 entity degree (training-graph hubness)"
    if name == "fraud_neighbor_rate":
        return "Tier-5 OOF fraud rate of 1-hop graph neighbours"
    if name == "pagerank_score":
        return "Tier-5 graph-pagerank score"
    if name == "clustering_coefficient":
        return "Tier-5 bipartite clustering coefficient"
    if name == "connected_component_size":
        return "Tier-5 connected-component size"
    if name == "is_coldstart_card1":
        return "Tier-3 cold-start flag for card1"
    if name in {"amt_zscore_vs_card1_history", "time_since_last_txn_zscore"}:
        return "Tier-3 behavioural deviation z-score"
    if name in {"addr_change_flag", "device_change_flag"}:
        return "Tier-3 behavioural deviation flag"
    if name == "hour_deviation":
        return "Tier-3 hour-of-day deviation from card history"
    if (
        name.endswith("_velocity_1h")
        or name.endswith("_velocity_24h")
        or name.endswith("_velocity_7d")
    ):
        return "Tier-2 per-entity velocity counter"
    if (
        name.endswith("_amt_mean_30d")
        or name.endswith("_amt_std_30d")
        or name.endswith("_amt_max_30d")
    ):
        return "Tier-2 historical-amount stats (30 d rolling)"
    if name.startswith("is_null_"):
        return "Tier-1 missingness indicator"
    if name in {"log_amount", "amount_decile"}:
        return "Tier-1 amount transform"
    if name in {
        "hour_of_day",
        "is_business_hours",
        "hour_sin",
        "hour_cos",
        "day_of_week",
        "is_weekend",
    }:
        return "Tier-1 time feature"
    if name.startswith("P_emaildomain_") or name.startswith("R_emaildomain_"):
        return "Tier-1 email-domain feature"
    if name.startswith("V"):
        return "Vesta-engineered V feature"
    if name.startswith("C"):
        return "Vesta-engineered C feature"
    if name.startswith("D"):
        return "Vesta-engineered D feature"
    if name.startswith("M"):
        return "Vesta-engineered M flag"
    if name.startswith("id_"):
        return "Identity / device feature"
    if name in {
        "TransactionAmt",
        "ProductCD",
        "card1",
        "card2",
        "card3",
        "card4",
        "card5",
        "card6",
        "addr1",
        "addr2",
        "dist1",
        "dist2",
    }:
        return "Raw transaction column"
    return "(uncategorised)"


# ---------------------------------------------------------------------
# Report rendering.
# ---------------------------------------------------------------------


def _render_training_report(  # noqa: PLR0915 — linear markdown builder; splitting into sub-helpers fragments the report layout without clarifying it
    result: TrainingResult,
    feature_importance: pd.DataFrame,
    trial_history: list[dict[str, Any]],
    out_path: Path,
) -> None:
    """Emit `reports/model_a_training_report.md`.

    Carries the spec-required sections: best params, trial history,
    metrics before+after calibration, top-50 feature importance with
    business interpretation, latency distribution. Acceptance gates
    are flagged green / red inline.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    val_auc_gate = result.val_auc >= _VAL_AUC_GATE
    p95_gate = result.latency_p95_ms < _LATENCY_P95_BUDGET_MS
    cal_gate = result.val_log_loss_calibrated <= result.val_log_loss_uncalibrated * 1.01

    lines: list[str] = []
    lines.append("# Model A — LightGBM training report")
    lines.append("")
    lines.append("- **Generated by:** `scripts/train_lightgbm.py`")
    lines.append(f"- **Train rows:** {result.n_train_rows:,}")
    lines.append(f"- **Val rows:** {result.n_val_rows:,}")
    lines.append(f"- **Test rows:** {result.n_test_rows:,}")
    lines.append(f"- **Features:** {result.n_features:,}")
    lines.append("")
    lines.append("## Headline metrics")
    lines.append("")
    lines.append("| Metric | Val | Test |")
    lines.append("|---|---|---|")
    lines.append(
        f"| ROC-AUC | {result.val_auc:.{_AUC_DIGITS}f} | {result.test_auc:.{_AUC_DIGITS}f} |"
    )
    lines.append(
        f"| PR-AUC | {result.val_pr_auc:.{_AUC_DIGITS}f} | {result.test_pr_auc:.{_AUC_DIGITS}f} |"
    )
    lines.append(
        f"| Log loss (uncalibrated) | {result.val_log_loss_uncalibrated:.6f} "
        f"| {result.test_log_loss_uncalibrated:.6f} |"
    )
    lines.append(
        f"| Log loss (calibrated) | {result.val_log_loss_calibrated:.6f} "
        f"| {result.test_log_loss_calibrated:.6f} |"
    )
    lines.append(
        f"| Brier (uncalibrated) | {result.val_brier_uncalibrated:.6f} "
        f"| {result.test_brier_uncalibrated:.6f} |"
    )
    lines.append(
        f"| Brier (calibrated) | {result.val_brier_calibrated:.6f} "
        f"| {result.test_brier_calibrated:.6f} |"
    )
    lines.append(
        f"| ECE (uncalibrated) | {result.val_ece_uncalibrated:.6f} "
        f"| {result.test_ece_uncalibrated:.6f} |"
    )
    lines.append(
        f"| ECE (calibrated) | {result.val_ece_calibrated:.6f} "
        f"| {result.test_ece_calibrated:.6f} |"
    )
    lines.append("")
    lines.append(f"**Calibration method chosen:** `{result.calibration_method}`")
    lines.append("")
    lines.append("## Acceptance gates")
    lines.append("")
    lines.append(
        f"- {'✅' if val_auc_gate else '❌'} Val AUC ≥ {_VAL_AUC_GATE} "
        f"(realised: {result.val_auc:.{_AUC_DIGITS}f})"
    )
    lines.append(
        f"- {'✅' if p95_gate else '❌'} Inference p95 < {_LATENCY_P95_BUDGET_MS} ms "
        f"(realised: {result.latency_p95_ms:.2f} ms)"
    )
    lines.append(
        f"- {'✅' if cal_gate else '❌'} Calibration doesn't hurt val log loss "
        f"({result.val_log_loss_calibrated:.6f} vs {result.val_log_loss_uncalibrated:.6f} baseline)"
    )
    lines.append("")
    lines.append("## Inference latency")
    lines.append("")
    lines.append(
        f"Single-row `predict_proba → calibrator.transform`, "
        f"n = {_LATENCY_N_SAMPLES:,} random rows from val:"
    )
    lines.append("")
    lines.append("| Quantile | Latency (ms) |")
    lines.append("|---|---|")
    lines.append(f"| p50 | {result.latency_p50_ms:.2f} |")
    lines.append(f"| p95 | {result.latency_p95_ms:.2f} |")
    lines.append(f"| p99 | {result.latency_p99_ms:.2f} |")
    lines.append("")
    lines.append(f"![inference latency histogram](figures/{result.figure_path.name})")
    lines.append("")
    lines.append("## Best Optuna parameters")
    lines.append("")
    lines.append(
        f"From `configs/{_BEST_PARAMS_YAML}` — best Optuna val AUC: "
        f"**{result.best_value:.{_AUC_DIGITS}f}** over "
        f"{result.n_trials} trials."
    )
    lines.append("")
    lines.append("```yaml")
    lines.append(
        yaml.safe_dump(result.best_params, sort_keys=True, default_flow_style=False).rstrip()
    )
    lines.append("```")
    lines.append("")
    lines.append(f"## Trial history (top {min(_TOP_TRIALS_COUNT, len(trial_history))})")
    lines.append("")
    if trial_history:
        lines.append("| Rank | Trial | Val AUC | Params (selected) |")
        lines.append("|---|---|---|---|")
        for rank, trial in enumerate(trial_history[:_TOP_TRIALS_COUNT], start=1):
            params_short = ", ".join(
                f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in trial["params"].items()
                if k in {"num_leaves", "learning_rate", "max_depth", "min_child_samples"}
            )
            lines.append(
                f"| {rank} | {trial['number']} "
                f"| {trial['value']:.{_AUC_DIGITS}f} "
                f"| {params_short} |"
            )
    else:
        lines.append("_(trial history unavailable — see MLflow tuning run for details)_")
    lines.append("")
    lines.append(f"## Top {_TOP_FEATURE_COUNT} feature importances (gain)")
    lines.append("")
    lines.append("| Rank | Feature | Gain | Tier / interpretation |")
    lines.append("|---|---|---|---|")
    top = feature_importance.head(_TOP_FEATURE_COUNT)
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        lines.append(
            f"| {rank} | `{row['feature']}` "
            f"| {row['importance']:.0f} "
            f"| {_interpret_feature(str(row['feature']))} |"
        )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Model: `{result.model_path}`")
    lines.append(f"- Calibrator: `{result.calibrator_path}`")
    lines.append(f"- This report: `{result.report_path}`")
    lines.append(f"- Latency histogram: `{result.figure_path}`")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------


def train_pipeline(  # noqa: PLR0913, PLR0915 — single-file orchestration; eight knobs match the CLI surface
    *,
    settings: Settings,
    n_trials: int = _DEFAULT_N_TRIALS,
    skip_tuning: bool = False,
    sample_size: int | None = None,
    num_boost_round: int = _DEFAULT_NUM_BOOST_ROUND,
    early_stopping_rounds: int = _DEFAULT_EARLY_STOPPING_ROUNDS,
    random_state: int | None = None,
    report_path: Path = _TRAINING_REPORT_PATH,
    figure_path: Path = _LATENCY_FIGURE_PATH,
) -> TrainingResult:
    """End-to-end train → tune → calibrate → evaluate → save → report.

    Args:
        settings: Project settings (paths + seeds).
        n_trials: Number of Optuna trials. Default 100.
        skip_tuning: If True, skip the sweep; read existing
            `configs/model_best_params.yaml` instead.
        sample_size: If not None, stratified-subsample the train +
            val + test frames to roughly this many rows total. Used
            by the `--quick` smoke and the integration test.
        num_boost_round: Per-trial + final-fit boosting cap.
        early_stopping_rounds: Per-trial + final-fit early-stop patience.
        random_state: Seed for the Optuna sampler + the smoke
            subsample. Defaults to `settings.seed`.
        report_path: Where to write the training-report markdown.
        figure_path: Where to write the latency-histogram PNG.

    Returns:
        TrainingResult bundle with all metrics + artefact paths.
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

    feature_cols = _select_features(train)
    train_x = train[feature_cols]
    train_y = train["isFraud"].to_numpy()
    val_x = val[feature_cols]
    val_y = val["isFraud"].to_numpy()
    test_x = test[feature_cols]
    test_y = test["isFraud"].to_numpy()

    _logger.info(
        "train_lightgbm.loaded",
        n_train=len(train_x),
        n_val=len(val_x),
        n_test=len(test_x),
        n_features=len(feature_cols),
        sample_size=sample_size,
    )

    # --- Tune (or read YAML) ---
    yaml_path = _resolve_best_params_yaml()
    if not skip_tuning:
        _logger.info("train_lightgbm.tune_start", n_trials=n_trials)
        run_tuning(
            train_x,
            train_y,
            val_x,
            val_y,
            n_trials=n_trials,
            study_name=_MLFLOW_RUN_NAME_TUNE,
            output_path=yaml_path,
            random_state=seed,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
        )
    else:
        _logger.info("train_lightgbm.tune_skipped", yaml=str(yaml_path))
    best_params, best_value, best_n_trials, _study_name = _read_best_params_yaml(yaml_path)

    # --- Final fit (separate top-level MLflow run) ---
    configure_mlflow()
    experiment_id = setup_experiment()
    mlflow.set_experiment(experiment_id=experiment_id)

    models_dir = settings.models_dir / _MODELS_SUBDIR
    models_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(experiment_id=experiment_id, run_name=_MLFLOW_RUN_NAME_FIT):
        mlflow.set_tag("stage", "sprint3_train_final")
        for k, v in best_params.items():
            mlflow.log_param(f"best_{k}", v)
        mlflow.log_param("n_trials", best_n_trials)
        mlflow.log_param("n_train_rows", int(len(train_x)))
        mlflow.log_param("n_val_rows", int(len(val_x)))
        mlflow.log_param("n_test_rows", int(len(test_x)))
        mlflow.log_param("n_features", int(len(feature_cols)))
        mlflow.log_param("sample_size", sample_size if sample_size is not None else "full")

        model = LightGBMFraudModel(
            params=best_params,
            random_state=seed,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
        )
        model.fit(train_x, train_y, val_x, val_y)

        # Score val + test, uncalibrated.
        val_p = model.predict_proba(val_x)[:, 1]
        test_p = model.predict_proba(test_x)[:, 1]
        val_auc = float(roc_auc_score(val_y, val_p))
        test_auc = float(roc_auc_score(test_y, test_p))
        val_pr_auc = float(average_precision_score(val_y, val_p))
        test_pr_auc = float(average_precision_score(test_y, test_p))
        val_ll_unc = log_loss(val_y, val_p)
        test_ll_unc = log_loss(test_y, test_p)
        val_brier_unc = brier_score(val_y, val_p)
        test_brier_unc = brier_score(test_y, test_p)
        val_ece_unc = expected_calibration_error(val_y, val_p)
        test_ece_unc = expected_calibration_error(test_y, test_p)

        # Calibrate on val; apply to val + test.
        cal_method, calibrator = select_calibration_method(val_y, val_p, random_state=seed)
        val_p_cal = calibrator.transform(val_p)
        test_p_cal = calibrator.transform(test_p)
        val_ll_cal = log_loss(val_y, val_p_cal)
        test_ll_cal = log_loss(test_y, test_p_cal)
        val_brier_cal = brier_score(val_y, val_p_cal)
        test_brier_cal = brier_score(test_y, test_p_cal)
        val_ece_cal = expected_calibration_error(val_y, val_p_cal)
        test_ece_cal = expected_calibration_error(test_y, test_p_cal)

        _logger.info(
            "train_lightgbm.metrics",
            val_auc=val_auc,
            test_auc=test_auc,
            val_pr_auc=val_pr_auc,
            test_pr_auc=test_pr_auc,
            val_ll_unc=val_ll_unc,
            val_ll_cal=val_ll_cal,
            calibration_method=cal_method,
        )

        mlflow.log_metric("val_auc", val_auc)
        mlflow.log_metric("test_auc", test_auc)
        mlflow.log_metric("val_pr_auc", val_pr_auc)
        mlflow.log_metric("test_pr_auc", test_pr_auc)
        mlflow.log_metric("val_log_loss_uncalibrated", val_ll_unc)
        mlflow.log_metric("val_log_loss_calibrated", val_ll_cal)
        mlflow.log_metric("test_log_loss_uncalibrated", test_ll_unc)
        mlflow.log_metric("test_log_loss_calibrated", test_ll_cal)
        mlflow.log_metric("val_brier_uncalibrated", val_brier_unc)
        mlflow.log_metric("val_brier_calibrated", val_brier_cal)
        mlflow.log_metric("test_brier_uncalibrated", test_brier_unc)
        mlflow.log_metric("test_brier_calibrated", test_brier_cal)
        mlflow.log_metric("val_ece_uncalibrated", val_ece_unc)
        mlflow.log_metric("val_ece_calibrated", val_ece_cal)
        mlflow.log_metric("test_ece_uncalibrated", test_ece_unc)
        mlflow.log_metric("test_ece_calibrated", test_ece_cal)
        mlflow.set_tag("calibration_method", cal_method)

        # Save model + calibrator. `LightGBMFraudModel.save` writes
        # `lightgbm_model.joblib` + manifest sidecar; we dump the
        # calibrator next to it as `calibrator.joblib` (separate file
        # so Sprint 5's serving stack can load them independently and
        # so the manifest stays calibrator-agnostic).
        model_path, _manifest_path = model.save(models_dir)
        cal_path = models_dir / _CALIBRATOR_FILENAME
        joblib.dump(calibrator, cal_path)

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(cal_path))

        # Latency.
        p50, p95, p99, latencies = _measure_inference_latency(model, calibrator, val_x)
        mlflow.log_metric("latency_p50_ms", p50)
        mlflow.log_metric("latency_p95_ms", p95)
        mlflow.log_metric("latency_p99_ms", p99)
        _save_latency_histogram(latencies, figure_path)
        mlflow.log_artifact(str(figure_path))

        # Feature importance + trial history (best effort — tuning's
        # study isn't carried back to us; we report top-N from the
        # final model, and leave trial details to MLflow's tuning run).
        feat_imp = model.feature_importance(importance_type="gain")
        trial_history: list[dict[str, Any]] = []  # Optuna study not in scope here.

        result = TrainingResult(
            best_params=best_params,
            best_value=best_value,
            n_trials=best_n_trials,
            val_auc=val_auc,
            val_pr_auc=val_pr_auc,
            val_log_loss_uncalibrated=val_ll_unc,
            val_log_loss_calibrated=val_ll_cal,
            val_brier_uncalibrated=val_brier_unc,
            val_brier_calibrated=val_brier_cal,
            val_ece_uncalibrated=val_ece_unc,
            val_ece_calibrated=val_ece_cal,
            test_auc=test_auc,
            test_pr_auc=test_pr_auc,
            test_log_loss_uncalibrated=test_ll_unc,
            test_log_loss_calibrated=test_ll_cal,
            test_brier_uncalibrated=test_brier_unc,
            test_brier_calibrated=test_brier_cal,
            test_ece_uncalibrated=test_ece_unc,
            test_ece_calibrated=test_ece_cal,
            calibration_method=cal_method,
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            latency_p99_ms=p99,
            n_features=int(len(feature_cols)),
            n_train_rows=int(len(train_x)),
            n_val_rows=int(len(val_x)),
            n_test_rows=int(len(test_x)),
            model_path=model_path,
            calibrator_path=cal_path,
            report_path=report_path,
            figure_path=figure_path,
        )

        _render_training_report(result, feat_imp, trial_history, report_path)
        mlflow.log_artifact(str(report_path))

    return result


def _stratified_subsample(df: pd.DataFrame, target_n: int, seed: int = 42) -> pd.DataFrame:
    """Stratified subsample to ~`target_n` rows by `isFraud`. Skip if df smaller."""
    if len(df) <= target_n:
        return df.reset_index(drop=True)
    kept, _ = train_test_split(
        df,
        train_size=target_n,
        stratify=df["isFraud"],
        random_state=seed,
    )
    # `cast` because `train_test_split`'s return type is too loose for
    # mypy to narrow back to `pd.DataFrame`; the runtime type is
    # guaranteed by the input.
    return cast(pd.DataFrame, kept.reset_index(drop=True))


# ---------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------


@click.command()
@click.option(
    "--n-trials",
    type=int,
    default=_DEFAULT_N_TRIALS,
    show_default=True,
    help="Number of Optuna trials in the tuning sweep.",
)
@click.option(
    "--skip-tuning",
    is_flag=True,
    default=False,
    help="Skip the sweep; read existing configs/model_best_params.yaml.",
)
@click.option(
    "--quick",
    is_flag=True,
    default=False,
    help=(
        "Smoke run: 5 trials + 5k stratified subsample + small "
        "boosting cap. For integration tests / iteration, NOT for "
        "production model selection."
    ),
)
def main(n_trials: int, skip_tuning: bool, quick: bool) -> None:
    """Sprint 3 LightGBM training pipeline. See module docstring."""
    settings = get_settings()
    settings.ensure_directories()

    if quick:
        result = train_pipeline(
            settings=settings,
            n_trials=_SMOKE_N_TRIALS,
            skip_tuning=skip_tuning,
            sample_size=_SMOKE_SAMPLE_SIZE,
            num_boost_round=_SMOKE_NUM_BOOST_ROUND,
            early_stopping_rounds=_SMOKE_EARLY_STOPPING_ROUNDS,
        )
    else:
        result = train_pipeline(
            settings=settings,
            n_trials=n_trials,
            skip_tuning=skip_tuning,
        )

    val_auc_gate = result.val_auc >= _VAL_AUC_GATE
    p95_gate = result.latency_p95_ms < _LATENCY_P95_BUDGET_MS
    cal_gate = result.val_log_loss_calibrated <= result.val_log_loss_uncalibrated * 1.01

    click.echo(click.style("train_lightgbm: COMPLETE", fg="green", bold=True))
    click.echo(
        f"  val_auc          = {result.val_auc:.{_AUC_DIGITS}f}  "
        f"({'PASS' if val_auc_gate else 'GAP'} vs {_VAL_AUC_GATE})"
    )
    click.echo(f"  test_auc         = {result.test_auc:.{_AUC_DIGITS}f}")
    click.echo(f"  val_pr_auc       = {result.val_pr_auc:.{_AUC_DIGITS}f}")
    click.echo(f"  val_ll_unc       = {result.val_log_loss_uncalibrated:.6f}")
    click.echo(
        f"  val_ll_cal       = {result.val_log_loss_calibrated:.6f}  "
        f"({'PASS' if cal_gate else 'REGRESSION'})"
    )
    click.echo(f"  cal_method       = {result.calibration_method}")
    click.echo(
        f"  latency p50/p95/p99 = {result.latency_p50_ms:.2f}/"
        f"{result.latency_p95_ms:.2f}/{result.latency_p99_ms:.2f} ms  "
        f"({'PASS' if p95_gate else 'GAP'} vs {_LATENCY_P95_BUDGET_MS:.0f} ms)"
    )
    click.echo(f"  model            = {result.model_path}")
    click.echo(f"  calibrator       = {result.calibrator_path}")
    click.echo(f"  report           = {result.report_path}")


if __name__ == "__main__":
    main()
