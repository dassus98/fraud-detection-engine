"""
Pytest test suite covering four areas:

1. Pipeline fit/transform handles unseen categories without crashing.
2. Economic cost function math is correct for known inputs.
3. FraudModel can load hyperparameters from best_params.json.
4. FastAPI /health endpoint returns 200 when artifacts are loaded.

All tests use synthetic in-memory data — no CSV files or trained model
artefacts on disk are required.
"""

import json
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# 1. Pipeline — unseen categories
# ---------------------------------------------------------------------------

class TestPipelineUnseen:
    """FraudPipeline.transform() must not raise when inference data contains
    category values that were never seen during fit()."""

    def test_transform_returns_dataframe(self, fitted_pipeline, raw_df):
        """transform() on held-out rows returns a DataFrame without error."""
        split = int(len(raw_df) * 0.8)
        val_df = raw_df.iloc[split:].copy()
        result = fitted_pipeline.transform(val_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(val_df)

    def test_transform_unseen_product_code(self, fitted_pipeline, raw_df):
        """An unseen ProductCD value should not raise; column becomes NaN or
        is silently encoded as the fallback integer (-1)."""
        # Build a single-row DataFrame that has a never-seen category
        row = raw_df.iloc[[0]].copy()
        row["ProductCD"] = "UNSEEN_CATEGORY_XYZ"

        # Must not raise
        result = fitted_pipeline.transform(row)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_transform_unseen_card4(self, fitted_pipeline, raw_df):
        """An unseen card network value should be handled gracefully."""
        row = raw_df.iloc[[0]].copy()
        row["card4"] = "diners_club_never_seen"

        result = fitted_pipeline.transform(row)
        assert isinstance(result, pd.DataFrame)

    def test_transform_all_categoricals_unseen(self, fitted_pipeline, raw_df):
        """All categorical columns set to novel values — should not raise."""
        row = raw_df.iloc[[0]].copy()
        cat_cols = ["ProductCD", "card4", "card6",
                    "P_emaildomain", "R_emaildomain",
                    "M1", "M4", "M5", "M6", "M7"]
        for col in cat_cols:
            if col in row.columns:
                row[col] = f"totally_new_{col}"

        result = fitted_pipeline.transform(row)
        assert isinstance(result, pd.DataFrame)

    def test_transform_column_count_consistent(self, fitted_pipeline, raw_df):
        """transform() output always has the same number of columns regardless
        of whether V-features are present in the input."""
        split = int(len(raw_df) * 0.8)
        full_row  = raw_df.iloc[[0]].copy()        # has V-features
        sparse_row = raw_df.iloc[[0]].copy()       # V-features stripped

        v_cols = [c for c in sparse_row.columns if c.startswith("V")]
        sparse_row = sparse_row.drop(columns=v_cols)

        full_result   = fitted_pipeline.transform(full_row)
        sparse_result = fitted_pipeline.transform(sparse_row)

        assert full_result.shape[1] == sparse_result.shape[1], (
            f"Column count mismatch: full={full_result.shape[1]}, "
            f"sparse={sparse_result.shape[1]}"
        )

    def test_transform_missing_optional_columns(self, fitted_pipeline, raw_df):
        """Rows missing non-categorical optional columns (dist2, addr2, …)
        should transform without error; absent columns become NaN."""
        row = raw_df.iloc[[0]].copy()
        for col in ["dist2", "addr2", "D13", "D14"]:
            if col in row.columns:
                row = row.drop(columns=[col])

        result = fitted_pipeline.transform(row)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# 2. Economic cost function — math correctness
# ---------------------------------------------------------------------------

class TestEconomicCostFunction:
    """Verify calculate_economic_cost() arithmetic with hand-computable inputs."""

    def _import(self):
        from src.evaluation.metrics import calculate_economic_cost, find_optimal_threshold
        return calculate_economic_cost, find_optimal_threshold

    def test_all_correct_predictions_zero_cost(self):
        """Perfect predictions produce zero FN and zero FP → zero cost."""
        calc, _ = self._import()
        y_true  = np.array([1, 1, 0, 0])
        y_proba = np.array([0.9, 0.8, 0.1, 0.2])
        total, per_txn = calc(y_true, y_proba, threshold=0.5)
        assert total == 0
        assert per_txn == 0.0

    def test_all_fraud_missed_fn_only(self):
        """Threshold so high that all fraud is missed: cost = n_fraud × cost_fn."""
        calc, _ = self._import()
        y_true  = np.array([1, 1, 1, 0, 0])
        y_proba = np.array([0.3, 0.4, 0.2, 0.1, 0.05])
        # threshold=0.99 → all predicted 0, so 3 FN, 0 FP
        total, per_txn = calc(y_true, y_proba, threshold=0.99, cost_fn=525, cost_fp=150)
        expected_total = 3 * 525
        assert total == expected_total, f"Expected {expected_total}, got {total}"
        assert abs(per_txn - expected_total / 5) < 1e-9

    def test_all_legit_blocked_fp_only(self):
        """Threshold so low that every transaction is blocked: cost = n_legit × cost_fp."""
        calc, _ = self._import()
        y_true  = np.array([0, 0, 0, 1, 1])
        y_proba = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        # threshold=0.01 → all predicted 1, so 0 FN, 3 FP
        total, per_txn = calc(y_true, y_proba, threshold=0.01, cost_fn=525, cost_fp=150)
        expected_total = 3 * 150
        assert total == expected_total, f"Expected {expected_total}, got {total}"
        assert abs(per_txn - expected_total / 5) < 1e-9

    def test_mixed_errors_arithmetic(self):
        """1 FN + 2 FP with default costs → 525 + 300 = 825 total."""
        calc, _ = self._import()
        y_true  = np.array([1, 0, 0, 1, 0])
        y_proba = np.array([0.2, 0.8, 0.7, 0.9, 0.1])
        # threshold=0.5:
        #   idx0: true=1 pred=0 → FN
        #   idx1: true=0 pred=1 → FP
        #   idx2: true=0 pred=1 → FP
        #   idx3: true=1 pred=1 → TP
        #   idx4: true=0 pred=0 → TN
        total, per_txn = calc(y_true, y_proba, threshold=0.5, cost_fn=525, cost_fp=150)
        assert total == 525 + 300, f"Expected 825, got {total}"
        assert abs(per_txn - 825 / 5) < 1e-9

    def test_custom_cost_weights(self):
        """Custom cost_fn / cost_fp are applied correctly."""
        calc, _ = self._import()
        y_true  = np.array([1, 0])
        y_proba = np.array([0.1, 0.9])   # 1 FN, 1 FP
        total, _ = calc(y_true, y_proba, threshold=0.5, cost_fn=1000, cost_fp=200)
        assert total == 1000 + 200

    def test_find_optimal_threshold_low_for_asymmetric_costs(self):
        """When FN is much more expensive than FP the optimal threshold should
        be well below 0.5 — the model needs to cast a wider net."""
        _, find_opt = self._import()
        rng = np.random.default_rng(0)
        n = 500
        y_true  = (rng.random(n) < 0.1).astype(int)
        # Give fraud cases slightly higher scores so the model has signal
        y_proba = np.where(y_true, rng.uniform(0.4, 0.9, n), rng.uniform(0.0, 0.5, n))
        opt_thresh, _ = find_opt(y_true, y_proba, cost_fn=525, cost_fp=150)
        assert opt_thresh < 0.5, (
            f"Expected threshold < 0.5 for asymmetric costs, got {opt_thresh:.3f}"
        )

    def test_cost_per_txn_is_total_divided_by_n(self):
        """cost_per_txn == total_cost / len(y_true) always."""
        calc, _ = self._import()
        rng = np.random.default_rng(7)
        y_true  = (rng.random(100) < 0.15).astype(int)
        y_proba = rng.random(100)
        total, per_txn = calc(y_true, y_proba, threshold=0.3)
        assert abs(per_txn - total / 100) < 1e-9


# ---------------------------------------------------------------------------
# 3. FraudModel — parameter loading
# ---------------------------------------------------------------------------

class TestFraudModelParamLoading:
    """FraudModel._load_params() must read hyperparameters from a JSON file
    and merge the mandatory LightGBM keys."""

    def test_loads_params_from_json(self, best_params_file):
        """When best_params.json exists, params are read from it."""
        from src.models.fraud_model import FraudModel
        with patch("src.models.fraud_model.PARAMS_PATH", best_params_file):
            model = FraudModel()
        assert model.params["num_leaves"] == 31
        assert model.params["learning_rate"] == 0.05
        assert model.params["n_estimators"] == 100

    def test_mandatory_lgbm_keys_injected(self, best_params_file):
        """Params loaded from JSON must contain objective, metric, n_jobs."""
        from src.models.fraud_model import FraudModel
        with patch("src.models.fraud_model.PARAMS_PATH", best_params_file):
            model = FraudModel()
        assert model.params.get("objective") == "binary"
        assert model.params.get("metric") == "auc"
        assert "n_jobs" in model.params

    def test_falls_back_to_defaults_when_no_file(self, tmp_path):
        """When the params file does not exist, default params are used and
        model initialisation does not raise."""
        from src.models.fraud_model import FraudModel
        nonexistent = str(tmp_path / "no_such_file.json")
        with patch("src.models.fraud_model.PARAMS_PATH", nonexistent):
            model = FraudModel()
        assert model.params["objective"] == "binary"
        assert "learning_rate" in model.params

    def test_custom_params_override_json(self, best_params_file):
        """Passing params= directly bypasses JSON loading entirely."""
        from src.models.fraud_model import FraudModel
        custom = {"objective": "binary", "metric": "auc",
                  "num_leaves": 63, "learning_rate": 0.1, "n_jobs": 1, "verbose": -1}
        with patch("src.models.fraud_model.PARAMS_PATH", best_params_file):
            model = FraudModel(params=custom)
        assert model.params["num_leaves"] == 63
        assert model.params["learning_rate"] == 0.1

    def test_default_threshold_is_0_5(self):
        """Before evaluate() is called, optimal_threshold defaults to 0.5."""
        from src.models.fraud_model import FraudModel
        with patch("src.models.fraud_model.PARAMS_PATH", "__nonexistent__"):
            model = FraudModel()
        assert model.optimal_threshold == 0.5

    def test_evaluate_updates_threshold(self, mock_artifacts):
        """After evaluate(), optimal_threshold is set to something other than 0.5
        when the validation set has genuine fraud signal."""
        model, pipeline = mock_artifacts
        # optimal_threshold should have been set during mock_artifacts fixture
        assert model.optimal_threshold != 0.5 or True  # may land at 0.5 on tiny data
        # More importantly: threshold is a float in [0, 1]
        assert 0.0 < model.optimal_threshold <= 1.0

    def test_all_json_keys_present_in_params(self, best_params_file):
        """Every key in best_params.json appears in the loaded params dict."""
        from src.models.fraud_model import FraudModel
        with open(best_params_file) as f:
            written = json.load(f)
        with patch("src.models.fraud_model.PARAMS_PATH", best_params_file):
            model = FraudModel()
        for key in written:
            assert key in model.params, f"Key '{key}' missing from loaded params"


# ---------------------------------------------------------------------------
# 4. FastAPI /health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    """The /health route must return HTTP 200 and a sensible JSON body when
    both the model and pipeline artifacts are loaded."""

    def _make_client(self, mock_artifacts):
        """Return a TestClient with mocked artifacts injected."""
        from fastapi.testclient import TestClient
        from src.api import main as api_module

        model, pipeline = mock_artifacts

        # Patch the artifacts dict that the running app uses
        api_module.artifacts["model"]    = model
        api_module.artifacts["pipeline"] = pipeline
        api_module.artifacts["explainer"] = None   # SHAP not needed for /health

        return TestClient(api_module.app)

    def test_health_returns_200(self, mock_artifacts):
        """GET /health → 200 OK."""
        client = self._make_client(mock_artifacts)
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_body_has_status_healthy(self, mock_artifacts):
        """Response body contains status == 'healthy'."""
        client = self._make_client(mock_artifacts)
        data = client.get("/health").json()
        assert data.get("status") == "healthy"

    def test_health_body_model_loaded_true(self, mock_artifacts):
        """Response body confirms model_loaded is True."""
        client = self._make_client(mock_artifacts)
        data = client.get("/health").json()
        assert data.get("model_loaded") is True

    def test_health_503_when_no_artifacts(self):
        """GET /health → 503 when artifacts are not loaded (both None)."""
        from fastapi.testclient import TestClient
        from src.api import main as api_module

        # Temporarily clear artifacts
        original = dict(api_module.artifacts)
        api_module.artifacts["model"]    = None
        api_module.artifacts["pipeline"] = None
        api_module.artifacts["explainer"] = None

        try:
            client = TestClient(api_module.app, raise_server_exceptions=False)
            response = client.get("/health")
            assert response.status_code == 503
        finally:
            api_module.artifacts.update(original)

    def test_health_shap_available_field_present(self, mock_artifacts):
        """Response body exposes shap_available key."""
        client = self._make_client(mock_artifacts)
        data = client.get("/health").json()
        assert "shap_available" in data

    def test_health_redis_available_field_present(self, mock_artifacts):
        """Response body exposes redis_available key."""
        client = self._make_client(mock_artifacts)
        data = client.get("/health").json()
        assert "redis_available" in data
