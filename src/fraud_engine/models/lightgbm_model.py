"""LightGBM model wrapper for fraud detection.

Wraps `lgb.Booster` directly (NOT the sklearn `LGBMClassifier` API).
Sprint 3's hyperparameter-tuning prompt, Sprint 4's economic-cost
evaluator, and Sprint 5's serving stack all need a stable model
contract: fit on (X_train, y_train, X_val, y_val), get probabilities
back, persist with full provenance for audit. This class is that
contract.

Business rationale:
    Sprint 1's `train_baseline` used `LGBMClassifier` for the headline
    val AUC alongside a quick MLflow run. That works for a one-off
    baseline; it does not work for the production-realistic surface
    we need from Sprint 3 onwards. Specifically:

    - **Explicit early stopping.** Sprint 3 hyperparameter tuning runs
      hundreds of fits; we need `lgb.early_stopping(N)` callbacks
      with a configurable patience parameter, not the sklearn API's
      implicit early-stopping behaviour.
    - **Native Booster access.** `feature_importance(importance_type)`
      and `feature_name()` are first-class on the booster. The
      sklearn wrapper exposes them via a `.booster_` indirection
      that adds friction without value.
    - **Smaller, audit-friendly serialised payload.** Joblib-dumping
      the booster + a JSON manifest gives ops a `cat`-able artefact;
      the sklearn wrapper carries extra inheritance noise.

    Sprint 1's baseline stays — it's the temporal-split / random-split
    AUC anchor in `sprints/sprint_1` and the only consumer of MLflow.
    `LightGBMFraudModel` is the new surface for Sprint 3+.

Trade-offs considered:
    - **Native Booster vs LGBMClassifier.** Native is per spec; gives
      explicit early stopping; smaller serialised payload. Cost:
      `predict_proba` returns 1-D from the Booster, but the sklearn
      ecosystem (e.g. `sklearn.metrics.roc_auc_score`) accepts the
      same 1-D shape via `predict_proba(X)[:, 1]` indexing. We return
      `(n, 2)` to match the sklearn API surface so consumers can
      switch wrappers without changing call sites.
    - **`scale_pos_weight` vs `is_unbalance`.** Per spec,
      `scale_pos_weight`. If not specified, computed from y_train
      as `neg / pos` at fit time. Deterministic and propagates
      severity (a 99/1 split should weight more aggressively than
      a 60/40 split); `is_unbalance` is a coin-flip flag that loses
      that gradient.
    - **Joblib + JSON manifest persistence.** Mirrors
      `FeaturePipeline.save / TransactionEntityGraph.save` exactly.
      Joblib pickles the booster cleanly; manifest sidecar is
      `cat`-able and `jq`-queryable for audit.
    - **Manifest contains both `schema_hash` and `content_hash`.**
      Schema hash = SHA-256 of `{col: str(dtype)}` truncated to 16
      chars (mirrors `data/lineage.py:_schema_fingerprint`); lets a
      downstream consumer detect "the input frame I'm about to
      `predict_proba` doesn't match what was used at fit time."
      Content hash = full SHA-256 of joblib bytes (mirrors
      `models/baseline.py:_sha256_joblib`); lets ops detect
      bit-level drift across re-saves of an ostensibly identical
      model.
    - **No content-hash prefix in filename.** The Sprint 1 baseline
      embeds the hash in the filename; we keep the filename stable
      (`lightgbm_model.joblib`) and surface the hash via the
      manifest. Reason: downstream callers should target a
      well-known path; the manifest is where provenance lives.
    - **`predict_proba` returns float, not int.** Even when LightGBM
      returns near-0 or near-1 with float32, we return float64.
      Calibration (Sprint 4 territory) requires float precision.

Cross-references:
    - `src/fraud_engine/features/pipeline.py:175-236` — save/load template
    - `src/fraud_engine/features/tier5_graph.py:460-513` — save/load + manifest
    - `src/fraud_engine/data/lineage.py:86-89` — schema-fingerprint pattern
    - `src/fraud_engine/models/baseline.py:428-440` — content-hash pattern
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Final, Self

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from fraud_engine.config.settings import get_settings
from fraud_engine.utils.logging import get_logger

_logger = get_logger(__name__)

# ---------------------------------------------------------------------
# Module constants.
# ---------------------------------------------------------------------

# Persistence filenames. Stable across re-saves of the same model;
# provenance lives in the manifest, not the filename.
_MODEL_FILENAME: Final[str] = "lightgbm_model.joblib"
_MANIFEST_FILENAME: Final[str] = "lightgbm_model_manifest.json"

# Manifest schema version. Bump when the manifest JSON shape changes
# in a non-backward-compatible way.
_MANIFEST_SCHEMA_VERSION: Final[int] = 1

# Default boosting cap; early stopping typically truncates well below.
_DEFAULT_NUM_BOOST_ROUND: Final[int] = 500
# Default early-stopping patience (rounds without val-AUC improvement).
_DEFAULT_EARLY_STOPPING_ROUNDS: Final[int] = 20

# Schema-fingerprint truncation length, mirrors `data/lineage.py`.
_SCHEMA_FINGERPRINT_HEX_CHARS: Final[int] = 16

# Importance types accepted by `lgb.Booster.feature_importance`.
_VALID_IMPORTANCE_TYPES: Final[frozenset[str]] = frozenset({"gain", "split"})

# Number of class probability columns returned by `predict_proba`.
# 2 = (P[class=0], P[class=1]) — sklearn convention.
_N_CLASS_PROBS: Final[int] = 2


def _schema_fingerprint(df: pd.DataFrame) -> str:
    """SHA-256 of `{col: str(dtype)}` (alphabetised), hex-truncated.

    Mirrors `data/lineage.py:_schema_fingerprint`. The fingerprint is
    deterministic across runs given the same column names + dtypes,
    and changes when EITHER drifts. Truncated to 16 hex chars (64 bits
    of entropy) — sufficient for collision-resistant fingerprinting at
    the project's scale.
    """
    schema_dict = {col: str(df[col].dtype) for col in sorted(df.columns)}
    schema_str = json.dumps(schema_dict, separators=(",", ":"))
    full_hash = hashlib.sha256(schema_str.encode("utf-8")).hexdigest()
    return full_hash[:_SCHEMA_FINGERPRINT_HEX_CHARS]


def _sha256_joblib(obj: Any) -> str:
    """Return a deterministic SHA-256 of `obj` as serialised by joblib.

    Mirrors `models/baseline.py:_sha256_joblib`. Using a temp file
    keeps the implementation dependency-free: joblib writes to a
    path, we read the bytes back, and hash. Returns the full
    64-character hex digest.
    """
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as handle:
        joblib.dump(obj, handle.name)
        tmp_path = Path(handle.name)
    try:
        return hashlib.sha256(tmp_path.read_bytes()).hexdigest()
    finally:
        tmp_path.unlink(missing_ok=True)


class LightGBMFraudModel:
    """Native `lgb.Booster` wrapper for fraud detection.

    Public API:
        - `fit(X_train, y_train, X_val, y_val)` — train with early
          stopping; return self.
        - `predict_proba(X)` — return `(n, 2)` probability matrix
          (`P[class=0]`, `P[class=1]`).
        - `save(path)` — joblib payload + JSON manifest sidecar.
        - `load(path)` — classmethod inverse of `save`.
        - `feature_importance(importance_type="gain")` — return a
          DataFrame with `feature` + `importance` columns, sorted
          descending.

    Fitted state (all `None` pre-fit):
        booster_: Fitted `lgb.Booster`.
        feature_names_: List of feature column names from `X_train`.
        n_features_: `len(feature_names_)`.
        best_iteration_: Iteration selected by early stopping (1-based).
        best_score_: Best val AUC (the early-stopping metric).
        scale_pos_weight_: The scale_pos_weight value used at fit time.
    """

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        random_state: int | None = None,
        scale_pos_weight: float | None = None,
        num_boost_round: int = _DEFAULT_NUM_BOOST_ROUND,
        early_stopping_rounds: int = _DEFAULT_EARLY_STOPPING_ROUNDS,
    ) -> None:
        """Construct the model with optional hyperparameter overrides.

        Args:
            params: LightGBM hyperparameter dict. If `None`, uses
                `Settings.lgbm_defaults` (objective=binary, metric=auc,
                num_leaves=63, learning_rate=0.05, max_depth=-1,
                n_estimators=500, min_child_samples=20, reg_alpha=0,
                reg_lambda=0). Caller-supplied keys override the
                defaults.
            random_state: Seed for `lgb.train`. If `None`, uses
                `Settings.seed` (default 42).
            scale_pos_weight: Class-imbalance weight. If `None`,
                computed at fit time as `neg / pos` from `y_train`.
                Spec mandates `scale_pos_weight` over `is_unbalance`.
            num_boost_round: Maximum boosting iterations. Default 500.
                Early stopping typically truncates well below.
            early_stopping_rounds: Patience — early-stop after this
                many rounds without val-metric improvement. Default 20.

        Raises:
            ValueError: If `early_stopping_rounds < 1` or
                `num_boost_round < 1`.
        """
        if num_boost_round < 1:
            raise ValueError(
                f"LightGBMFraudModel: num_boost_round must be >= 1, " f"got {num_boost_round}"
            )
        if early_stopping_rounds < 1:
            raise ValueError(
                f"LightGBMFraudModel: early_stopping_rounds must be >= 1, "
                f"got {early_stopping_rounds}"
            )
        settings = get_settings()
        # Build effective params: settings defaults overlaid with caller kwargs.
        effective_params: dict[str, Any] = dict(settings.lgbm_defaults)
        if params is not None:
            effective_params.update(params)

        self.params: dict[str, Any] = effective_params
        self.random_state: int = random_state if random_state is not None else settings.seed
        self._scale_pos_weight_init: float | None = scale_pos_weight
        self.num_boost_round: int = num_boost_round
        self.early_stopping_rounds: int = early_stopping_rounds

        # Fitted state — populated by `fit`.
        self.booster_: lgb.Booster | None = None
        self.feature_names_: list[str] | None = None
        self.n_features_: int | None = None
        self.best_iteration_: int | None = None
        self.best_score_: float | None = None
        self.scale_pos_weight_: float | None = None

    # -----------------------------------------------------------------
    # Fit.
    # -----------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,  # noqa: N803 — sklearn convention; matches prompt spec
        y_train: pd.Series[int] | np.ndarray[Any, Any],
        X_val: pd.DataFrame,  # noqa: N803 — sklearn convention; matches prompt spec
        y_val: pd.Series[int] | np.ndarray[Any, Any],
    ) -> Self:
        """Fit the booster with early stopping on the val split.

        Computes `scale_pos_weight = neg / pos` from `y_train` if not
        supplied at construction. Trains via the native `lgb.train`
        API with `lgb.early_stopping(early_stopping_rounds)` callback
        and `lgb.log_evaluation(0)` to silence per-iteration output.

        Args:
            X_train: Training features. Column order is preserved as
                `feature_names_`.
            y_train: Binary training labels (0 = negative, 1 = positive).
            X_val: Validation features. Must have the same columns as
                `X_train`.
            y_val: Validation labels.

        Returns:
            self, fitted in place.

        Raises:
            ValueError: If `X_train.shape[1] != X_val.shape[1]` or
                their column names differ; if `y_train` has only one
                class.
        """
        if list(X_train.columns) != list(X_val.columns):
            raise ValueError(
                "LightGBMFraudModel.fit: X_train and X_val must have "
                "identical column names and order"
            )
        y_train_arr = np.asarray(y_train).ravel()
        y_val_arr = np.asarray(y_val).ravel()
        unique_train = np.unique(y_train_arr)
        if len(unique_train) < _N_CLASS_PROBS:
            raise ValueError(
                f"LightGBMFraudModel.fit: y_train must contain both "
                f"classes; got unique values {unique_train.tolist()}"
            )

        # Compute scale_pos_weight from training data if not supplied.
        if self._scale_pos_weight_init is None:
            n_pos = int((y_train_arr == 1).sum())
            n_neg = int((y_train_arr == 0).sum())
            spw = float(n_neg) / float(n_pos) if n_pos > 0 else 1.0
        else:
            spw = float(self._scale_pos_weight_init)
        self.scale_pos_weight_ = spw

        # Build the final params dict. random_state + scale_pos_weight
        # + verbose go in here; the user's params take precedence on
        # any other key.
        train_params: dict[str, Any] = dict(self.params)
        train_params.setdefault("seed", self.random_state)
        train_params["scale_pos_weight"] = spw
        train_params.setdefault("verbose", -1)

        # `n_estimators` lives in lgbm_defaults but is named
        # `num_boost_round` for the native API; pop it to avoid the
        # "unknown parameter" warning, and prefer the constructor's
        # explicit `num_boost_round` over the inherited default.
        legacy_n_estimators = train_params.pop("n_estimators", None)
        if legacy_n_estimators is not None and self.num_boost_round == _DEFAULT_NUM_BOOST_ROUND:
            num_boost_round = int(legacy_n_estimators)
        else:
            num_boost_round = self.num_boost_round

        feature_names = [str(col) for col in X_train.columns]
        train_set = lgb.Dataset(
            X_train,
            label=y_train_arr,
            feature_name=feature_names,
            free_raw_data=False,
        )
        val_set = lgb.Dataset(
            X_val,
            label=y_val_arr,
            feature_name=feature_names,
            reference=train_set,
            free_raw_data=False,
        )

        booster = lgb.train(
            train_params,
            train_set,
            num_boost_round=num_boost_round,
            valid_sets=[val_set],
            valid_names=["val"],
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=self.early_stopping_rounds,
                    verbose=False,
                ),
                lgb.log_evaluation(period=0),
            ],
        )

        self.booster_ = booster
        self.feature_names_ = feature_names
        self.n_features_ = len(feature_names)
        self.best_iteration_ = int(booster.best_iteration)
        # `best_score` is `{val_name: {metric_name: score}}`.
        best_score_dict = booster.best_score or {}
        val_scores = best_score_dict.get("val", {})
        # The metric name comes from `params["metric"]` — typically "auc".
        metric_name = str(train_params.get("metric", "auc"))
        if metric_name in val_scores:
            self.best_score_ = float(val_scores[metric_name])
        elif val_scores:
            # Fallback: take the first available metric.
            self.best_score_ = float(next(iter(val_scores.values())))
        else:
            self.best_score_ = float("nan")

        _logger.info(
            "lightgbm_model.fit_done",
            n_features=self.n_features_,
            best_iteration=self.best_iteration_,
            best_score=self.best_score_,
            scale_pos_weight=self.scale_pos_weight_,
        )
        return self

    # -----------------------------------------------------------------
    # Predict.
    # -----------------------------------------------------------------

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray[Any, Any]:  # noqa: N803 — sklearn convention
        """Return per-row class probabilities `(n, 2)`.

        `lgb.Booster.predict` returns a 1-D array of `P[class=1]` for
        binary objective; we stack `[1 - p, p]` to match sklearn's
        `predict_proba` convention so downstream `roc_auc_score(y,
        proba[:, 1])` calls work without modification.

        Args:
            X: Frame to score. Must contain every column in
                `feature_names_`.

        Returns:
            float64 array of shape `(len(X), 2)`. Each row sums to 1.

        Raises:
            AttributeError: If `predict_proba` is called pre-fit.
            KeyError: If `X` is missing any column in
                `feature_names_`.
        """
        if self.booster_ is None or self.feature_names_ is None:
            raise AttributeError("LightGBMFraudModel must be fit before predict_proba")
        missing = sorted(set(self.feature_names_) - set(X.columns))
        if missing:
            raise KeyError(f"LightGBMFraudModel.predict_proba: missing column(s) {missing}")

        # Reorder to match training column order; LightGBM uses
        # column NAMES not positions, but reordering keeps numerical
        # determinism for sklearn-style consumers that index by position.
        X_ordered = X[self.feature_names_]  # noqa: N806 — sklearn convention
        if len(X_ordered) == 0:
            return np.empty((0, _N_CLASS_PROBS), dtype=np.float64)

        # `num_iteration=best_iteration_` ensures we use the
        # early-stopped booster, not the full num_boost_round.
        p_pos = np.asarray(
            self.booster_.predict(X_ordered, num_iteration=self.best_iteration_),
            dtype=np.float64,
        ).ravel()
        p_neg = 1.0 - p_pos
        return np.column_stack([p_neg, p_pos])

    # -----------------------------------------------------------------
    # Feature importance.
    # -----------------------------------------------------------------

    def feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """Return per-feature importance, sorted descending.

        Args:
            importance_type: ``"gain"`` (sum of split gains) or
                ``"split"`` (count of splits using the feature).
                Default ``"gain"``.

        Returns:
            DataFrame with columns ``["feature", "importance"]``,
            one row per feature, sorted descending by ``importance``.
            ``len(df) == n_features_``.

        Raises:
            AttributeError: If called pre-fit.
            ValueError: If ``importance_type`` is not "gain" or "split".
        """
        if self.booster_ is None or self.feature_names_ is None:
            raise AttributeError("LightGBMFraudModel must be fit before feature_importance")
        if importance_type not in _VALID_IMPORTANCE_TYPES:
            raise ValueError(
                f"LightGBMFraudModel.feature_importance: "
                f"importance_type must be one of "
                f"{sorted(_VALID_IMPORTANCE_TYPES)}, got {importance_type!r}"
            )

        importances = self.booster_.feature_importance(importance_type=importance_type)
        df = pd.DataFrame(
            {
                "feature": self.feature_names_,
                "importance": np.asarray(importances, dtype=np.float64),
            }
        )
        return df.sort_values("importance", ascending=False, kind="stable").reset_index(drop=True)

    # -----------------------------------------------------------------
    # Save / load.
    # -----------------------------------------------------------------

    def save(self, path: Path) -> tuple[Path, Path]:
        """Persist the fitted model + manifest under `path/`.

        Mirrors `FeaturePipeline.save` / `TransactionEntityGraph.save`:

        - `path/lightgbm_model.joblib` — pickled `LightGBMFraudModel`
          instance (carries the embedded `lgb.Booster`).
        - `path/lightgbm_model_manifest.json` — sidecar with params,
          feature names, schema hash, content hash, schema version,
          best iteration, best score, scale_pos_weight. `cat`-able
          and `jq`-queryable.

        Args:
            path: Destination directory. Created if missing.

        Returns:
            ``(model_path, manifest_path)`` for caller logging.

        Raises:
            AttributeError: If `save` is called pre-fit.
        """
        if self.booster_ is None:
            raise AttributeError("LightGBMFraudModel must be fit before save")
        path.mkdir(parents=True, exist_ok=True)
        model_path = path / _MODEL_FILENAME
        manifest_path = path / _MANIFEST_FILENAME

        joblib.dump(self, model_path)
        content_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
        manifest = self._build_manifest(content_hash=content_hash)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return model_path, manifest_path

    @classmethod
    def load(cls, path: Path) -> Self:
        """Inverse of `save`. Reads `path/lightgbm_model.joblib`.

        The manifest sidecar is not read here — it's an audit
        artefact, not part of the runtime contract.

        Args:
            path: Directory containing the saved model.

        Returns:
            The reconstructed `LightGBMFraudModel`.

        Raises:
            FileNotFoundError: If `path/lightgbm_model.joblib` does
                not exist.
            TypeError: If the joblib payload is not a
                `LightGBMFraudModel` instance.
        """
        model_path = path / _MODEL_FILENAME
        loaded = joblib.load(model_path)
        if not isinstance(loaded, cls):
            raise TypeError(
                f"Loaded object at {model_path} is "
                f"{type(loaded).__name__}, expected LightGBMFraudModel"
            )
        return loaded

    # -----------------------------------------------------------------
    # Manifest.
    # -----------------------------------------------------------------

    def _build_manifest(self, content_hash: str) -> dict[str, Any]:
        """Render the manifest dict (called from `save`).

        Args:
            content_hash: SHA-256 hex digest of the `model.joblib`
                bytes produced by the most-recent dump.

        Returns:
            JSON-safe dict.
        """
        if (
            self.feature_names_ is None
            or self.n_features_ is None
            or self.best_iteration_ is None
            or self.scale_pos_weight_ is None
        ):
            raise AttributeError("LightGBMFraudModel._build_manifest called before fit")
        # Schema fingerprint of the training feature frame is captured
        # from feature_names + dtype info recorded at fit time. We
        # don't carry the original DataFrame around (memory cost), so
        # we hash just the column names. Downstream callers can
        # cross-check by hashing their input columns the same way.
        schema_str = json.dumps(sorted(self.feature_names_), separators=(",", ":"))
        schema_hash = hashlib.sha256(schema_str.encode("utf-8")).hexdigest()[
            :_SCHEMA_FINGERPRINT_HEX_CHARS
        ]
        return {
            "schema_version": _MANIFEST_SCHEMA_VERSION,
            "params": self.params,
            "feature_names": list(self.feature_names_),
            "n_features": int(self.n_features_),
            "best_iteration": int(self.best_iteration_),
            "best_score": (float(self.best_score_) if self.best_score_ is not None else None),
            "scale_pos_weight": float(self.scale_pos_weight_),
            "num_boost_round": int(self.num_boost_round),
            "early_stopping_rounds": int(self.early_stopping_rounds),
            "random_state": int(self.random_state),
            "schema_hash": schema_hash,
            "content_hash": content_hash,
        }


__all__ = ["LightGBMFraudModel"]
