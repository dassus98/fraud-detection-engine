"""LightGBM baseline on raw IEEE-CIS features.

Two variants share a single training path:

1. `random` — stratified 80/20 split, seeded. Answers "what AUC does
   a vanilla classifier reach when temporal ordering is ignored?"
2. `temporal` — uses `temporal_split` so training rows predate
   validation rows. Answers "what AUC remains when the model can
   only see past transactions?"

The gap (random − temporal) is the leakage-risk signal Sprint 2's
feature engineering must watch. Both variants open an MLflow run
under `settings.mlflow_experiment_name` and persist the fitted
model to `settings.models_dir` with a content-addressed filename so
identical configurations collide on disk rather than proliferating.

Business rationale:
    A baseline is the discipline check for every later sprint.
    Without a frozen, reproducible number for "LightGBM on raw
    columns," any AUC lift from Sprint 2 (engineered features),
    Sprint 3 (Optuna tuning), or Sprint 4 (cost-optimised threshold)
    is unattributable — we cannot tell whether the gain came from
    the work or from noise. The temporal variant additionally
    mirrors production: the fraud model will only ever score
    transactions newer than anything it trained on, and the
    temporal-AUC is the honest number the hiring committee should
    see.

Trade-offs considered:
    - LightGBM's native categorical handling via
      `categorical_feature="auto"` is superior to one-hot for
      high-cardinality columns (card1, P_emaildomain). The
      `RawDataLoader._optimize` pass already sets `category` dtypes,
      so the sklearn-API call picks them up transparently. An
      alternative (manual encoding) would duplicate that work and
      risk drift.
    - Tuning is deliberately out of scope here — Sprint 3 owns
      Optuna. The baseline uses `settings.lgbm_defaults` verbatim
      so the comparison in Sprint 3's report is apples-to-apples.
    - `content_hash` is the SHA-256 of the joblib bytes, computed
      before writing. That guarantees two runs with the same data
      and seed produce the same file name — collisions are silent
      overwrites, which is the desired behaviour (rerunning a
      baseline should not spawn a second copy of the same model).
    - The test set from `temporal_split` is *not touched* here. It
      is frozen for Sprint 4's cost-curve evaluation — any peek at
      it now contaminates that evaluation.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import joblib
import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from fraud_engine.config.settings import Settings, get_settings
from fraud_engine.data.splits import (
    temporal_split,
    validate_no_overlap,
)
from fraud_engine.utils.logging import get_logger, log_call
from fraud_engine.utils.mlflow_setup import (
    configure_mlflow,
    log_dataframe_stats,
    setup_experiment,
)

_TIME_COLUMN: Final[str] = "TransactionDT"
_LABEL_COLUMN: Final[str] = "isFraud"
_KEY_COLUMN: Final[str] = "TransactionID"

# Columns excluded from the feature matrix. TransactionID is a unique
# row identifier (pure leakage if fed to the model); TransactionDT is
# the temporal axis used for splitting (pure leakage under the
# temporal variant, and a proxy for recency under the random one);
# isFraud is the label.
_NON_FEATURE_COLUMNS: Final[frozenset[str]] = frozenset({_TIME_COLUMN, _LABEL_COLUMN, _KEY_COLUMN})

# Number of feature-importance entries persisted as an MLflow artefact.
# 20 is large enough to cover all the expected-heavy columns
# (TransactionAmt, card1, addr1, the top V-features) on IEEE-CIS
# without bloating the artefact store.
_IMPORTANCE_TOP_K: Final[int] = 20

# Random-variant holdout fraction. 0.2 mirrors the temporal val-size
# on the default 4/1/1 split so AUC is measured on roughly-equal
# sample sizes between variants.
_RANDOM_TEST_SIZE: Final[float] = 0.2

Variant = Literal["random", "temporal"]


@dataclass(frozen=True)
class BaselineResult:
    """Return object from `train_baseline`.

    Attributes:
        variant: Which split strategy produced this result.
        model_path: Absolute path to the persisted LightGBM model.
        auc: ROC-AUC on the held-out set (val rows for `temporal`;
            the stratified 20% holdout for `random`).
        feature_importances: Top-20 features by LightGBM gain,
            descending. Values are native LightGBM gain scores (not
            normalised) so two models fit on the same data produce
            comparable numbers.
        content_hash: 64-char SHA-256 of the joblib bytes. Used to
            name the on-disk file and to assert reproducibility in
            the integration tests.
    """

    variant: Variant
    model_path: Path
    auc: float
    feature_importances: dict[str, float]
    content_hash: str


@log_call
def train_baseline(
    merged: pd.DataFrame,
    *,
    variant: Variant,
    settings: Settings | None = None,
    run_name: str | None = None,
) -> BaselineResult:
    """Fit a LightGBM baseline and log it to MLflow.

    Business rationale:
        See the module docstring. The one-paragraph version is: this
        number is the discipline check. Any later AUC claim in this
        repo is measured against the `temporal` variant of this
        function, and the gap to the `random` variant is the
        leakage-risk signal Sprint 2 must not widen.

    Trade-offs considered:
        - Variant dispatch is explicit (`Literal["random",
          "temporal"]`) rather than a generic split callable. The
          set of legitimate variants is closed — a third option
          would be a feature request, not a parameter — and mypy
          flags typos at import time.
        - The MLflow run always opens, even under the random variant
          where a split manifest would be misleading. The `variant`
          tag makes the distinction; logging both variants into the
          same experiment keeps the model-selection tree coherent.
        - Feature importances are logged as a JSON artefact rather
          than as individual metrics. MLflow's per-metric rendering
          is designed for time-series, not leaderboard-style
          rankings; JSON is easier to diff across runs.

    Args:
        merged: Merged IEEE-CIS frame (output of
            `RawDataLoader.load_merged`). Must contain
            `TransactionID`, `TransactionDT`, `isFraud`.
        variant: Either `"random"` (stratified 80/20) or
            `"temporal"` (`temporal_split` on `settings.train_end_dt`
            / `val_end_dt`).
        settings: Override for the Settings singleton. Tests pass a
            monkeypatched instance so models land under `tmp_path`.
        run_name: Human-friendly MLflow run name. Defaults to
            `baseline_{variant}`.

    Returns:
        A `BaselineResult` summarising the fitted model.

    Raises:
        KeyError: If `merged` is missing a required column.
        ValueError: If `variant` is not one of `"random"` /
            `"temporal"`, or if the temporal split fails.
    """
    effective_settings = settings or get_settings()
    _require_core_columns(merged)

    feature_cols = _select_feature_columns(merged.columns)
    label = merged[_LABEL_COLUMN].astype(np.int64)

    if variant == "random":
        x_train, x_val, y_train, y_val = train_test_split(
            merged[feature_cols],
            label,
            test_size=_RANDOM_TEST_SIZE,
            stratify=label,
            random_state=effective_settings.seed,
        )
        split_manifest: dict[str, Any] | None = None
    elif variant == "temporal":
        splits = temporal_split(merged, settings=effective_settings)
        validate_no_overlap(splits)
        x_train = splits.train[feature_cols]
        y_train = splits.train[_LABEL_COLUMN].astype(np.int64)
        x_val = splits.val[feature_cols]
        y_val = splits.val[_LABEL_COLUMN].astype(np.int64)
        split_manifest = splits.manifest
    else:
        raise ValueError(f"variant={variant!r} is not supported; expected 'random' or 'temporal'")

    configure_mlflow()
    experiment_id = setup_experiment()

    effective_run_name = run_name or f"baseline_{variant}"
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=effective_run_name,
    ) as mlflow_run:
        mlflow.set_tag("variant", variant)
        mlflow.set_tag("stage", "sprint1_baseline")

        for key, value in effective_settings.lgbm_defaults.items():
            mlflow.log_param(f"lgbm_{key}", value)
        mlflow.log_param("seed", effective_settings.seed)
        mlflow.log_param("n_features", len(feature_cols))

        if split_manifest is not None:
            for key, value in split_manifest.items():
                mlflow.log_param(f"split_{key}", value)

        log_dataframe_stats(x_train, prefix="train")
        log_dataframe_stats(x_val, prefix="val")

        classifier = LGBMClassifier(
            **effective_settings.lgbm_defaults,
            random_state=effective_settings.seed,
            verbose=-1,
        )
        classifier.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            categorical_feature="auto",
        )

        val_proba = classifier.predict_proba(x_val)[:, 1]
        auc = float(roc_auc_score(y_val, val_proba))
        mlflow.log_metric("auc", auc)

        importances = _top_k_importances(
            classifier,
            feature_cols,
            k=_IMPORTANCE_TOP_K,
        )
        _log_importance_artefact(importances)

        content_hash = _sha256_joblib(classifier)
        model_path = _write_model(
            classifier,
            models_dir=effective_settings.models_dir,
            variant=variant,
            content_hash=content_hash,
        )
        mlflow.log_param("model_path", str(model_path))
        mlflow.log_param("content_hash", content_hash)
        mlflow.log_artifact(str(model_path))

        get_logger(__name__).info(
            "baseline.trained",
            variant=variant,
            auc=auc,
            n_train=int(len(x_train)),
            n_val=int(len(x_val)),
            content_hash=content_hash,
            mlflow_run_id=mlflow_run.info.run_id,
        )

    return BaselineResult(
        variant=variant,
        model_path=model_path,
        auc=auc,
        feature_importances=importances,
        content_hash=content_hash,
    )


def _require_core_columns(df: pd.DataFrame) -> None:
    """Fail loudly if `df` lacks the columns every variant needs.

    Raising `KeyError` matches the splitter's contract and keeps the
    failure mode uniform between the random and temporal branches.
    """
    missing = {_TIME_COLUMN, _LABEL_COLUMN, _KEY_COLUMN} - set(df.columns)
    if missing:
        raise KeyError(f"train_baseline requires columns {sorted(missing)} in the merged frame")


def _select_feature_columns(all_columns: pd.Index[str]) -> list[str]:
    """Return every column that is neither a key, label, nor time axis."""
    return [c for c in all_columns if c not in _NON_FEATURE_COLUMNS]


def _top_k_importances(
    classifier: LGBMClassifier,
    feature_cols: list[str],
    *,
    k: int,
) -> dict[str, float]:
    """Return the top-`k` features by gain, descending.

    Sorted because downstream consumers (MLflow artefact, sprint
    report) care about the ranking, not just the set of columns.
    """
    gains = classifier.booster_.feature_importance(importance_type="gain")
    ranked = sorted(
        zip(feature_cols, gains, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    return {name: float(score) for name, score in ranked[:k]}


def _log_importance_artefact(importances: dict[str, float]) -> None:
    """Persist feature importances as a JSON artefact on the MLflow run.

    Uses a NamedTemporaryFile because `mlflow.log_artifact` only
    accepts paths, not in-memory buffers.
    """
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        delete=False,
        encoding="utf-8",
    ) as handle:
        json.dump(importances, handle, indent=2, sort_keys=False)
        tmp_path = Path(handle.name)
    try:
        mlflow.log_artifact(str(tmp_path), artifact_path="feature_importances")
    finally:
        tmp_path.unlink(missing_ok=True)


def _sha256_joblib(obj: Any) -> str:
    """Return a deterministic SHA-256 of `obj` as serialised by joblib.

    Using a NamedTemporaryFile keeps the implementation dependency-
    free: joblib writes to a path, we read the bytes back, and hash.
    """
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as handle:
        joblib.dump(obj, handle.name)
        tmp_path = Path(handle.name)
    try:
        return hashlib.sha256(tmp_path.read_bytes()).hexdigest()
    finally:
        tmp_path.unlink(missing_ok=True)


def _write_model(
    classifier: LGBMClassifier,
    *,
    models_dir: Path,
    variant: Variant,
    content_hash: str,
) -> Path:
    """Persist `classifier` under `baseline_{variant}_{hash[:12]}.joblib`."""
    models_dir.mkdir(parents=True, exist_ok=True)
    target = models_dir / f"baseline_{variant}_{content_hash[:12]}.joblib"
    joblib.dump(classifier, target)
    return target


__all__ = [
    "BaselineResult",
    "Variant",
    "train_baseline",
]
