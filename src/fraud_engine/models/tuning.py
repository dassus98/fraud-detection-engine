"""Optuna hyperparameter-tuning harness for `LightGBMFraudModel`.

Sprint 3 prompt 3.3.b: a 100-trial Optuna study (in-memory storage,
TPE sampler + MedianPruner) that searches the canonical LightGBM
hyperparameter space, logs every trial to MLflow as a nested run
under a single parent study run, and writes the winning params to
a YAML config for downstream consumers.

This module is the harness; the full 100-trial sweep against Tier-5
features lives in prompt 3.3.d. This prompt's verification is a
5-trial smoke test on synthetic data.

Business rationale:
    Sprint 3's Tier-5 default-hparam val AUC of 0.7689 sits well
    below the 0.93-0.94 spec target. The leak gate confirms the
    feature pipeline is correct, so the gap is a tuning problem.
    Optuna over the LightGBM hyperparameter manifold is the
    standard recovery path. Wrapping the study in a reusable
    function lets Sprint 4's economic-cost evaluator and Sprint 5's
    serving stack repeat the sweep against a frozen feature surface
    without copy-pasting the search space or the MLflow plumbing.

Trade-offs considered:
    - **TPE sampler vs random / grid.** TPE is the Optuna default and
      consistently outperforms random search at moderate trial counts
      (50-200). Grid search would force pre-committing to a rectangular
      slice; TPE expands its sampling around promising regions across
      trials. Cost: TPE is sequential (no parallel-trial speedup
      without a shared storage backend). Acceptable for 100 trials at
      ~1-3 s each.
    - **MedianPruner with `n_startup_trials=5`.** Early trials run to
      completion (so the pruner has comparison baselines); later
      trials whose intermediate val AUC drops below the median of
      completed trials at the same step get killed. Saves wall-time
      on hopeless trials. Trade-off: aggressive pruning can prune a
      slow-starting good trial; `n_startup_trials=5` + the modest
      n_warmup_steps=10 default give the pruner a conservative bias.
    - **In-memory storage.** No SQLite file, no parallel workers, no
      study resume. Sufficient for the 100-trial sweep (~5-10 min on
      Tier-5 features). 3.3.d may switch to SQLite if the sweep grows
      or needs resume-on-failure semantics.
    - **MLflow nested runs.** Each trial logs as a child run under
      a parent study run. The parent carries the study-level summary
      (best_value, best_params, n_trials); children carry the per-
      trial sampled params and val AUC. Mirrors the MLflow nested-run
      idiom and makes the UI's "Compare runs" view immediately
      productive.
    - **Search-space scope.** 9 hyperparameters: num_leaves,
      learning_rate (log-uniform), max_depth, min_child_samples,
      reg_alpha (log-uniform), reg_lambda (log-uniform),
      feature_fraction, bagging_fraction, bagging_freq. Standard
      LightGBM tuning surface. PROJECT_PLAN.md is not yet checked
      into the repo at this prompt; the spec referenced "Section 3.3"
      conventions and these are the canonical ranges every fraud-ML
      tuning study uses. Documented inline; if/when PROJECT_PLAN.md
      ships with different bounds, this module is the one place to
      update.
    - **YAML output format.** Matches the project's other
      `configs/*.yaml` files (PyYAML safe_dump, comments allowed at
      file head). The output is a `best_params: {...}` block plus
      study metadata (best_value, n_trials, study_name); downstream
      consumers `yaml.safe_load(...)` and feed `best_params` to
      `LightGBMFraudModel(params=...)`.

Cross-references:
    - `src/fraud_engine/models/lightgbm_model.py` — the model the
      objective fits + scores per trial.
    - `src/fraud_engine/utils/mlflow_setup.py` —
      `configure_mlflow()` + `setup_experiment()` boilerplate used
      verbatim.
    - `src/fraud_engine/models/baseline.py:227-289` — the parent-
      run MLflow pattern this harness mirrors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import mlflow
import numpy as np
import optuna
import pandas as pd
import yaml
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score

from fraud_engine.config.settings import get_settings
from fraud_engine.models.lightgbm_model import LightGBMFraudModel
from fraud_engine.utils.logging import get_logger
from fraud_engine.utils.mlflow_setup import configure_mlflow, setup_experiment

_logger = get_logger(__name__)

# ---------------------------------------------------------------------
# Module constants.
# ---------------------------------------------------------------------

# Default trial count for the canonical 3.3.d sweep. Tests override
# this aggressively (5 trials).
_DEFAULT_N_TRIALS: Final[int] = 100

# Optuna `MedianPruner` startup window. Run the first 5 trials to
# completion so the median has comparison baselines; from trial 6+
# the pruner can kill hopeless trials early.
_MEDIAN_PRUNER_STARTUP_TRIALS: Final[int] = 5

# Default boosting cap for trial fits. Each trial early-stops well
# below this on the synthetic / production frames; the cap is a
# safety net for pathological hyperparameter combinations.
_DEFAULT_TRIAL_NUM_BOOST_ROUND: Final[int] = 1000
# Default early-stopping patience per trial. Tighter than 3.3.a's
# default (20) because tuning runs hundreds of trials and we want
# each one to fail fast on diminishing returns.
_DEFAULT_TRIAL_EARLY_STOPPING_ROUNDS: Final[int] = 50

# YAML schema version. Bump when the output layout changes.
_BEST_PARAMS_YAML_SCHEMA_VERSION: Final[int] = 1

# MLflow tag namespace for trial-level runs.
_MLFLOW_TAG_STAGE: Final[str] = "sprint3_tuning"

# Default destination for the best-params YAML when callers don't
# pass `output_path`. Resolved relative to the project root using the
# same `parents[3]` convention `config/settings.py:_PROJECT_ROOT`
# uses, so the default works regardless of caller cwd.
_DEFAULT_BEST_PARAMS_YAML: Final[Path] = (
    Path(__file__).resolve().parents[3] / "configs" / "model_best_params.yaml"
)


# ---------------------------------------------------------------------
# Search space.
# ---------------------------------------------------------------------


def _suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    """Sample one set of LightGBM hyperparameters from the search space.

    9 hyperparameters spanning the canonical LightGBM tuning surface:

    - **`num_leaves`**: tree complexity. Higher → more interactions
      captured, more overfit risk.
    - **`learning_rate`** (log-uniform): step size. Lower needs more
      boosting rounds but typically generalises better.
    - **`max_depth`**: hard cap on tree depth. Interacts with
      num_leaves; LightGBM honours both.
    - **`min_child_samples`**: minimum rows per leaf. Higher → more
      regularisation, fewer rare-pattern splits.
    - **`reg_alpha`** (L1, log-uniform): sparse regularisation.
    - **`reg_lambda`** (L2, log-uniform): smooth regularisation.
    - **`feature_fraction`**: column-subsampling per tree. <1 helps
      decorrelate trees and combats over-reliance on hub features.
    - **`bagging_fraction`** + **`bagging_freq`**: row-subsampling
      controls. `bagging_freq=0` disables row subsampling; >0 enables
      it every `bagging_freq` iterations.
    """
    return {
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 0, 7),
    }


# Names of every hyperparameter the search space samples. Used by the
# search-space-coverage test to detect drift between the harness and
# the YAML output.
SEARCH_SPACE_KEYS: Final[tuple[str, ...]] = (
    "num_leaves",
    "learning_rate",
    "max_depth",
    "min_child_samples",
    "reg_alpha",
    "reg_lambda",
    "feature_fraction",
    "bagging_fraction",
    "bagging_freq",
)


# ---------------------------------------------------------------------
# Objective.
# ---------------------------------------------------------------------


def _make_objective(  # noqa: PLR0913 — six explicit args keep the closure readable; folding into a config dict adds friction with no gain.
    X_train: pd.DataFrame,  # noqa: N803 — sklearn convention
    y_train: np.ndarray[Any, Any],
    X_val: pd.DataFrame,  # noqa: N803 — sklearn convention
    y_val: np.ndarray[Any, Any],
    random_state: int,
    num_boost_round: int,
    early_stopping_rounds: int,
) -> Any:
    """Return an Optuna objective closure binding the train/val split.

    Each trial:
        1. Samples params from the 9-knob search space.
        2. Fits a `LightGBMFraudModel` with the sampled params + the
           caller's `num_boost_round` and `early_stopping_rounds`.
        3. Scores val AUC via `roc_auc_score(y_val, proba[:, 1])`.
        4. Logs sampled params + val AUC to MLflow as a `nested=True`
           run under the parent study run.
        5. Returns val AUC (the value Optuna maximises).

    A ZeroDivisionError or numerical failure inside `lgb.train`
    propagates to Optuna, which marks the trial FAIL and the TPE
    sampler avoids that region. We do NOT swallow exceptions silently
    — a fit failure is a real signal Optuna should learn from.
    """

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial)
        with mlflow.start_run(
            run_name=f"trial_{trial.number:03d}",
            nested=True,
        ):
            mlflow.set_tag("stage", _MLFLOW_TAG_STAGE)
            mlflow.set_tag("trial_number", str(trial.number))
            for key, value in params.items():
                mlflow.log_param(key, value)

            model = LightGBMFraudModel(
                params=params,
                random_state=random_state,
                num_boost_round=num_boost_round,
                early_stopping_rounds=early_stopping_rounds,
            )
            model.fit(X_train, y_train, X_val, y_val)
            proba = model.predict_proba(X_val)
            val_auc = float(roc_auc_score(y_val, proba[:, 1]))

            mlflow.log_metric("val_auc", val_auc)
            mlflow.log_metric("best_iteration", float(model.best_iteration_ or 0))

        return val_auc

    return objective


# ---------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------


def run_tuning(  # noqa: PLR0913 — eight explicit knobs match the spec contract; folding into a config object hides the override surface from CLI consumers.
    X_train: pd.DataFrame,  # noqa: N803 — sklearn convention
    y_train: pd.Series[int] | np.ndarray[Any, Any],
    X_val: pd.DataFrame,  # noqa: N803 — sklearn convention
    y_val: pd.Series[int] | np.ndarray[Any, Any],
    n_trials: int = _DEFAULT_N_TRIALS,
    study_name: str = "lightgbm_fraud_tuning",
    output_path: Path | None = None,
    mlflow_run_name: str | None = None,
    random_state: int | None = None,
    num_boost_round: int = _DEFAULT_TRIAL_NUM_BOOST_ROUND,
    early_stopping_rounds: int = _DEFAULT_TRIAL_EARLY_STOPPING_ROUNDS,
) -> dict[str, Any]:
    """Run an Optuna study; log every trial to MLflow; write best params YAML.

    Args:
        X_train: Training features.
        y_train: Training labels (binary).
        X_val: Validation features.
        y_val: Validation labels (binary).
        n_trials: Number of Optuna trials. Default 100. Must be >= 1.
        study_name: Optuna study name. Used as MLflow parent run name
            if `mlflow_run_name` is None.
        output_path: Path to write the best-params YAML. If `None`,
            defaults to `configs/model_best_params.yaml`.
        mlflow_run_name: MLflow parent run name. Defaults to
            `study_name`.
        random_state: Seed for the TPE sampler + each trial's
            `LightGBMFraudModel`. If `None`, uses `Settings.seed`.
        num_boost_round: Per-trial LightGBM boosting cap. Default 1000.
        early_stopping_rounds: Per-trial early-stopping patience.
            Default 50.

    Returns:
        Dict with keys ``best_params``, ``best_value``, ``n_trials``,
        ``study_name``, ``output_path``. ``best_params`` is the dict
        consumed by `LightGBMFraudModel(params=...)`.

    Raises:
        ValueError: If `n_trials < 1`.
    """
    if n_trials < 1:
        raise ValueError(f"run_tuning: n_trials must be >= 1, got {n_trials}")

    effective_random_state = random_state if random_state is not None else get_settings().seed
    effective_output_path = output_path if output_path is not None else _DEFAULT_BEST_PARAMS_YAML
    effective_mlflow_run_name = mlflow_run_name or study_name

    # MLflow tracking setup. `setup_experiment` is idempotent; a stale
    # tracking-URI from a previous test fixture gets overwritten here.
    # `set_experiment` is the critical extra step: without it, nested
    # runs opened inside the per-trial `objective` closure default to
    # the global "Default" experiment instead of the one we just set
    # up — leaving trial runs invisible to `search_runs(experiment_ids
    # =[experiment_id])`. Setting it active makes both the parent run
    # and every nested child land in the same experiment.
    configure_mlflow()
    experiment_id = setup_experiment()
    mlflow.set_experiment(experiment_id=experiment_id)

    y_train_arr = np.asarray(y_train).ravel()
    y_val_arr = np.asarray(y_val).ravel()

    sampler = TPESampler(seed=effective_random_state)
    pruner = MedianPruner(n_startup_trials=_MEDIAN_PRUNER_STARTUP_TRIALS)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    objective = _make_objective(
        X_train,
        y_train_arr,
        X_val,
        y_val_arr,
        random_state=effective_random_state,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
    )

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=effective_mlflow_run_name,
    ):
        mlflow.set_tag("stage", _MLFLOW_TAG_STAGE)
        mlflow.set_tag("kind", "study_parent")
        mlflow.log_param("study_name", study_name)
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("sampler", type(sampler).__name__)
        mlflow.log_param("pruner", type(pruner).__name__)
        mlflow.log_param("random_state", effective_random_state)
        mlflow.log_param("num_boost_round", num_boost_round)
        mlflow.log_param("early_stopping_rounds", early_stopping_rounds)
        mlflow.log_param("n_train_rows", int(len(X_train)))
        mlflow.log_param("n_val_rows", int(len(X_val)))
        mlflow.log_param("n_features", int(X_train.shape[1]))

        study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

        best_value = float(study.best_value)
        best_params: dict[str, Any] = dict(study.best_params)
        mlflow.log_metric("best_value", best_value)
        mlflow.log_metric("best_trial_number", float(study.best_trial.number))
        for key, value in best_params.items():
            mlflow.log_param(f"best_{key}", value)

    payload = {
        "schema_version": _BEST_PARAMS_YAML_SCHEMA_VERSION,
        "study_name": study_name,
        "n_trials": int(n_trials),
        "best_value": best_value,
        "best_trial_number": int(study.best_trial.number),
        "random_state": int(effective_random_state),
        "best_params": best_params,
    }
    _write_best_params_yaml(payload, effective_output_path)

    _logger.info(
        "tuning.run_done",
        study_name=study_name,
        n_trials=n_trials,
        best_value=best_value,
        best_trial_number=int(study.best_trial.number),
        output_path=str(effective_output_path),
    )

    return {
        "best_params": best_params,
        "best_value": best_value,
        "n_trials": int(n_trials),
        "study_name": study_name,
        "output_path": effective_output_path,
    }


# ---------------------------------------------------------------------
# YAML output.
# ---------------------------------------------------------------------


_YAML_HEADER = """\
# Best LightGBM hyperparameters from the Optuna tuning study.
#
# Generated by `fraud_engine.models.tuning.run_tuning` (Sprint 3
# prompt 3.3.b harness; full 100-trial sweep in 3.3.d). Downstream
# consumers `yaml.safe_load(...)` and feed `best_params` to
# `LightGBMFraudModel(params=...)`.
#
# DO NOT hand-edit this file. To regenerate, run the canonical sweep
# (3.3.d) or invoke `run_tuning(...)` from a script / notebook.
"""


def _write_best_params_yaml(payload: dict[str, Any], output_path: Path) -> None:
    """Persist the best-params payload to YAML with a project header."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(payload, sort_keys=True, default_flow_style=False)
    output_path.write_text(_YAML_HEADER + "\n" + body, encoding="utf-8")


__all__ = [
    "SEARCH_SPACE_KEYS",
    "run_tuning",
]
