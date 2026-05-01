"""Unit tests for `fraud_engine.models.lightgbm_model.LightGBMFraudModel`.

Five contract surfaces:

- `TestFit`: training contract (fitted state populated; scale_pos_weight
  computed from data; explicit override; column-mismatch raises;
  one-class y_train raises; early stopping converges).
- `TestPredictProba`: shape `(n, 2)`, rows sum to 1, values in [0, 1];
  pre-fit raises `AttributeError`; missing columns raise `KeyError`;
  empty input returns `(0, 2)`.
- `TestSaveLoad`: joblib + manifest round-trip; manifest has expected
  keys + types; load reproduces predictions bit-for-bit; load rejects
  wrong object type; load on missing path raises `FileNotFoundError`.
- `TestFeatureImportance`: DataFrame with `feature` + `importance`
  columns, sorted descending; both `gain` and `split` modes work;
  invalid `importance_type` raises `ValueError`; pre-fit raises.
- `TestErrorHandling`: invalid construction args raise `ValueError`.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from fraud_engine.models.lightgbm_model import LightGBMFraudModel

# ---------------------------------------------------------------------
# Synthetic-data helpers.
# ---------------------------------------------------------------------


def _make_synthetic_xy(  # noqa: N802 — sklearn convention
    n_rows: int = 600,
    n_features: int = 5,
    fraud_rate: float = 0.20,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return a synthetic (X, y) pair with modelling signal.

    Fraud is correlated with a high-amount tail + minority of features
    so LightGBM can fit non-trivial AUC on a tiny frame.
    """
    rng = np.random.default_rng(seed)
    cols = {f"f{i}": rng.normal(0, 1, size=n_rows).astype(np.float32) for i in range(n_features)}
    cols["amount"] = (np.exp(rng.normal(0, 1, size=n_rows)) * 50.0).astype(np.float32)
    x_df = pd.DataFrame(cols)
    fraud_logit = (
        0.9 * (x_df["amount"].to_numpy() / x_df["amount"].mean() - 1) + 0.5 * x_df["f0"].to_numpy()
    )
    fraud_prob = 1.0 / (1.0 + np.exp(-fraud_logit))
    y = (rng.uniform(0, 1, size=n_rows) < fraud_rate * fraud_prob / fraud_prob.mean()).astype(
        np.int64
    )
    # Guarantee at least one positive and one negative.
    if y.sum() == 0:
        y[0] = 1
    if y.sum() == n_rows:
        y[0] = 0
    return x_df, y


def _train_val_split(
    x: pd.DataFrame, y: np.ndarray, train_frac: float = 0.7, seed: int = 42
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """Random train/val split. The wrapper itself is split-agnostic."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    cut = int(len(x) * train_frac)
    train_idx, val_idx = idx[:cut], idx[cut:]
    x_train = x.iloc[train_idx].reset_index(drop=True)
    x_val = x.iloc[val_idx].reset_index(drop=True)
    return x_train, y[train_idx], x_val, y[val_idx]


def _fitted_model() -> (
    tuple[LightGBMFraudModel, pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]
):
    """Fit a small model on synthetic data; return model + train/val frames.

    Trim the boosting cap to keep test wall-time low.
    """
    x_df, y = _make_synthetic_xy()
    x_train, y_train, x_val, y_val = _train_val_split(x_df, y)
    model = LightGBMFraudModel(num_boost_round=50, early_stopping_rounds=5)
    model.fit(x_train, y_train, x_val, y_val)
    return model, x_train, y_train, x_val, y_val


# ---------------------------------------------------------------------
# `TestFit`: training contract.
# ---------------------------------------------------------------------


class TestFit:
    """Training contract: fitted state populated, deterministic, leak-free."""

    def test_fit_populates_fitted_state(self) -> None:
        """After fit, all `_` attributes are populated with expected types."""
        model, x_train, _, _, _ = _fitted_model()
        assert isinstance(model.booster_, lgb.Booster)
        assert model.feature_names_ == list(x_train.columns)
        assert model.n_features_ == x_train.shape[1]
        assert model.best_iteration_ is not None
        assert model.best_iteration_ >= 1
        assert model.best_score_ is not None
        assert 0.0 <= model.best_score_ <= 1.0
        assert model.scale_pos_weight_ is not None
        assert model.scale_pos_weight_ > 0.0

    def test_fit_computes_scale_pos_weight_from_data(self) -> None:
        """When not supplied, scale_pos_weight = (neg / pos) on y_train."""
        x_df, y = _make_synthetic_xy(n_rows=400, fraud_rate=0.10, seed=7)
        x_train, y_train, x_val, y_val = _train_val_split(x_df, y)
        model = LightGBMFraudModel(num_boost_round=20, early_stopping_rounds=5)
        model.fit(x_train, y_train, x_val, y_val)
        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        expected = float(n_neg) / float(n_pos)
        assert model.scale_pos_weight_ == pytest.approx(expected)

    def test_fit_uses_explicit_scale_pos_weight_override(self) -> None:
        """`scale_pos_weight=5.0` at construction wins over data-derived value."""
        x_df, y = _make_synthetic_xy()
        x_train, y_train, x_val, y_val = _train_val_split(x_df, y)
        model = LightGBMFraudModel(
            num_boost_round=20,
            early_stopping_rounds=5,
            scale_pos_weight=5.0,
        )
        model.fit(x_train, y_train, x_val, y_val)
        assert model.scale_pos_weight_ == 5.0

    def test_fit_column_mismatch_raises(self) -> None:
        """X_train and X_val column lists must match exactly."""
        x_df, y = _make_synthetic_xy()
        x_train, y_train, x_val, y_val = _train_val_split(x_df, y)
        x_val_wrong = x_val.rename(columns={"f0": "renamed"})
        model = LightGBMFraudModel(num_boost_round=10, early_stopping_rounds=5)
        with pytest.raises(ValueError, match="column names"):
            model.fit(x_train, y_train, x_val_wrong, y_val)

    def test_fit_one_class_y_train_raises(self) -> None:
        """y_train must contain both 0 and 1."""
        x_df, _ = _make_synthetic_xy(n_rows=100)
        y_zeros = np.zeros(len(x_df), dtype=np.int64)
        x_train = x_df.iloc[:70]
        x_val = x_df.iloc[70:]
        y_train = y_zeros[:70]
        y_val = y_zeros[70:]
        model = LightGBMFraudModel(num_boost_round=10, early_stopping_rounds=5)
        with pytest.raises(ValueError, match="both"):
            model.fit(x_train, y_train, x_val, y_val)


# ---------------------------------------------------------------------
# `TestPredictProba`: shape + invariants.
# ---------------------------------------------------------------------


class TestPredictProba:
    """`predict_proba(X)` returns `(n, 2)` float64 with rows summing to 1."""

    def test_predict_proba_shape_and_invariants(self) -> None:
        """Shape `(len(X), 2)`, values in [0, 1], rows sum to 1."""
        model, _, _, x_val, _ = _fitted_model()
        proba = model.predict_proba(x_val)
        assert proba.shape == (len(x_val), 2)
        assert proba.dtype == np.float64
        assert (proba >= 0.0).all() and (proba <= 1.0).all()
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)

    def test_predict_proba_before_fit_raises(self) -> None:
        """Pre-fit `predict_proba` raises `AttributeError`."""
        model = LightGBMFraudModel()
        x_df, _ = _make_synthetic_xy()
        with pytest.raises(AttributeError, match="fit"):
            model.predict_proba(x_df)

    def test_predict_proba_missing_columns_raises(self) -> None:
        """Missing required columns raise `KeyError`."""
        model, _, _, x_val, _ = _fitted_model()
        x_drop = x_val.drop(columns=["f0"])
        with pytest.raises(KeyError, match="f0"):
            model.predict_proba(x_drop)

    def test_predict_proba_empty_dataframe_returns_empty_2d(self) -> None:
        """Predict on 0-row X → `(0, 2)` empty array."""
        model, _, _, x_val, _ = _fitted_model()
        empty = x_val.iloc[0:0]
        proba = model.predict_proba(empty)
        assert proba.shape == (0, 2)
        assert proba.dtype == np.float64

    def test_predict_proba_handles_reordered_columns(self) -> None:
        """Reordered input columns produce identical predictions."""
        model, _, _, x_val, _ = _fitted_model()
        proba_in_order = model.predict_proba(x_val)
        reordered = x_val[list(reversed(x_val.columns))]
        proba_reordered = model.predict_proba(reordered)
        np.testing.assert_allclose(proba_in_order, proba_reordered, atol=1e-12)


# ---------------------------------------------------------------------
# `TestSaveLoad`: joblib + manifest round-trip.
# ---------------------------------------------------------------------


class TestSaveLoad:
    """Persistence: joblib payload + JSON manifest sidecar."""

    def test_save_writes_model_and_manifest(self, tmp_path: Path) -> None:
        """Both `lightgbm_model.joblib` and `lightgbm_model_manifest.json` are written."""
        model, _, _, _, _ = _fitted_model()
        model_path, manifest_path = model.save(tmp_path)

        assert model_path.is_file()
        assert manifest_path.is_file()
        assert model_path.name == "lightgbm_model.joblib"
        assert manifest_path.name == "lightgbm_model_manifest.json"

    def test_manifest_has_expected_shape(self, tmp_path: Path) -> None:
        """Manifest carries every required key with correct types."""
        model, x_train, _, _, _ = _fitted_model()
        _, manifest_path = model.save(tmp_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        assert manifest["schema_version"] == 1
        assert isinstance(manifest["params"], dict)
        assert manifest["params"]["objective"] == "binary"
        assert manifest["feature_names"] == list(x_train.columns)
        assert manifest["n_features"] == x_train.shape[1]
        assert isinstance(manifest["best_iteration"], int)
        assert manifest["best_iteration"] >= 1
        assert isinstance(manifest["best_score"], float)
        assert isinstance(manifest["scale_pos_weight"], float)
        assert isinstance(manifest["random_state"], int)
        # Schema hash: 16-char hex (mirrors data/lineage.py).
        assert isinstance(manifest["schema_hash"], str)
        assert len(manifest["schema_hash"]) == 16
        assert all(c in "0123456789abcdef" for c in manifest["schema_hash"])
        # Content hash: 64-char hex sha256.
        assert isinstance(manifest["content_hash"], str)
        assert len(manifest["content_hash"]) == 64
        assert all(c in "0123456789abcdef" for c in manifest["content_hash"])

    def test_load_round_trip_predicts_identically(self, tmp_path: Path) -> None:
        """save → load → predict_proba on same X yields bit-identical output."""
        model, _, _, x_val, _ = _fitted_model()
        proba_before = model.predict_proba(x_val)
        model.save(tmp_path)
        reloaded = LightGBMFraudModel.load(tmp_path)
        proba_after = reloaded.predict_proba(x_val)
        np.testing.assert_array_equal(proba_before, proba_after)

    def test_load_rejects_wrong_object_type(self, tmp_path: Path) -> None:
        """`load` raises `TypeError` if the joblib payload isn't a `LightGBMFraudModel`."""
        joblib.dump({"not": "a model"}, tmp_path / "lightgbm_model.joblib")
        with pytest.raises(TypeError, match="expected LightGBMFraudModel"):
            LightGBMFraudModel.load(tmp_path)

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        """`load` on a directory with no `lightgbm_model.joblib` raises `FileNotFoundError`."""
        with pytest.raises(FileNotFoundError):
            LightGBMFraudModel.load(tmp_path / "nope")

    def test_save_before_fit_raises(self, tmp_path: Path) -> None:
        """`save` raises `AttributeError` if the model has not been fit."""
        model = LightGBMFraudModel()
        with pytest.raises(AttributeError, match="fit"):
            model.save(tmp_path)


# ---------------------------------------------------------------------
# `TestFeatureImportance`: DataFrame contract.
# ---------------------------------------------------------------------


class TestFeatureImportance:
    """`feature_importance` returns a sorted DataFrame with expected columns."""

    def test_returns_dataframe_with_correct_columns_and_length(self) -> None:
        """DataFrame with `feature` + `importance` columns; one row per feature."""
        model, x_train, _, _, _ = _fitted_model()
        df = model.feature_importance(importance_type="gain")
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["feature", "importance"]
        assert len(df) == x_train.shape[1]
        # Every training-feature appears exactly once.
        assert set(df["feature"]) == set(x_train.columns)

    def test_sorted_descending_by_importance(self) -> None:
        """Output is sorted descending; first row has the max importance."""
        model, _, _, _, _ = _fitted_model()
        df = model.feature_importance(importance_type="gain")
        importances = df["importance"].to_numpy()
        assert (importances[:-1] >= importances[1:]).all()

    def test_supports_both_gain_and_split(self) -> None:
        """Both `"gain"` and `"split"` modes return non-empty DataFrames."""
        model, _, _, _, _ = _fitted_model()
        gain_df = model.feature_importance(importance_type="gain")
        split_df = model.feature_importance(importance_type="split")
        assert len(gain_df) > 0
        assert len(split_df) > 0
        assert set(gain_df["feature"]) == set(split_df["feature"])

    def test_invalid_importance_type_raises(self) -> None:
        """Unknown `importance_type` raises `ValueError`."""
        model, _, _, _, _ = _fitted_model()
        with pytest.raises(ValueError, match="importance_type"):
            model.feature_importance(importance_type="permutation")

    def test_before_fit_raises(self) -> None:
        """Pre-fit `feature_importance` raises `AttributeError`."""
        model = LightGBMFraudModel()
        with pytest.raises(AttributeError, match="fit"):
            model.feature_importance()


# ---------------------------------------------------------------------
# `TestErrorHandling`: construction argument validation.
# ---------------------------------------------------------------------


class TestErrorHandling:
    """Construction-arg validation: bad config raises `ValueError` early."""

    def test_invalid_num_boost_round_raises(self) -> None:
        """`num_boost_round=0` raises at construction (fail-fast)."""
        with pytest.raises(ValueError, match="num_boost_round"):
            LightGBMFraudModel(num_boost_round=0)

    def test_invalid_early_stopping_rounds_raises(self) -> None:
        """`early_stopping_rounds=0` raises at construction."""
        with pytest.raises(ValueError, match="early_stopping_rounds"):
            LightGBMFraudModel(early_stopping_rounds=0)

    def test_params_override_settings_defaults(self) -> None:
        """Caller-supplied `params` override `Settings.lgbm_defaults`."""
        model = LightGBMFraudModel(params={"num_leaves": 7})
        assert model.params["num_leaves"] == 7
        # Other defaults are still present.
        assert model.params["objective"] == "binary"
        assert model.params["metric"] == "auc"
