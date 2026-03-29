"""
Shared pytest fixtures.

All fixtures create synthetic data in memory — no CSV files are required
to run these tests.
"""
import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.config import SELECTED_FEATURES


# ---------------------------------------------------------------------------
# Minimal DataFrame that looks like the real training data
# ---------------------------------------------------------------------------

# Categorical columns that FraudPipeline encodes
_CAT_COLS = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
             "M1", "M4", "M5", "M6", "M7"]

# A small subset of V-features so VFeatureCleaner has something to work with
_V_COLS = [f"V{i}" for i in range(1, 21)]   # V1–V20 (no real collinearity at this scale)


def _make_transactions(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Build a synthetic transaction DataFrame with the columns the pipeline expects.
    The fraud rate is ~10% (intentionally high so metrics fixtures are non-trivial).
    """
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        "TransactionDT": np.arange(n),
        "TransactionAmt": rng.uniform(10, 1000, n),
        "ProductCD":  rng.choice(["W", "H", "C", "S", "R"], n),
        "card1":  rng.integers(1000, 9999, n),
        "card2":  rng.uniform(100, 500, n),
        "card3":  rng.uniform(100, 200, n),
        "card4":  rng.choice(["visa", "mastercard", "amex", "discover"], n),
        "card5":  rng.uniform(100, 250, n),
        "card6":  rng.choice(["debit", "credit"], n),
        "addr1":  rng.uniform(100, 500, n),
        "addr2":  rng.uniform(10, 100, n),
        "dist1":  rng.uniform(0, 100, n),
        "dist2":  rng.uniform(0, 100, n),
        "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", "hotmail.com"], n),
        "R_emaildomain": rng.choice(["gmail.com", "yahoo.com", "hotmail.com", None], n),
        "C1":  rng.uniform(0, 5, n),
        "C2":  rng.uniform(0, 5, n),
        "C5":  rng.uniform(0, 3, n),
        "C8":  rng.uniform(0, 5, n),
        "C9":  rng.uniform(0, 3, n),
        "C12": rng.uniform(0, 3, n),
        "D1":  rng.uniform(0, 300, n),
        "D3":  rng.uniform(0, 100, n),
        "D4":  rng.uniform(0, 100, n),
        "D5":  rng.uniform(0, 100, n),
        "D8":  rng.uniform(0, 100, n),
        "D10": rng.uniform(0, 100, n),
        "D11": rng.uniform(0, 100, n),
        "D13": rng.uniform(0, 100, n),
        "D14": rng.uniform(0, 100, n),
        "D15": rng.uniform(0, 100, n),
        "M1":  rng.choice(["T", "F", None], n),
        "M4":  rng.choice(["M0", "M1", "M2", None], n),
        "M5":  rng.choice(["T", "F", None], n),
        "M6":  rng.choice(["T", "F", None], n),
        "M7":  rng.choice(["T", "F", None], n),
        "isFraud": (rng.random(n) < 0.10).astype(int),
    })

    # Add a handful of V-features with mild random values
    for col in _V_COLS:
        df[col] = rng.uniform(0, 1, n)

    return df


@pytest.fixture(scope="session")
def raw_df():
    """Full synthetic DataFrame (200 rows) — represents the raw CSV."""
    return _make_transactions(n=200)


@pytest.fixture(scope="session")
def fitted_pipeline(raw_df):
    """A FraudPipeline fitted on the synthetic training split."""
    from src.pipeline import FraudPipeline
    split = int(len(raw_df) * 0.8)
    train_df = raw_df.iloc[:split]
    pipeline = FraudPipeline()
    pipeline.fit(train_df)
    return pipeline


@pytest.fixture(scope="session")
def best_params_file():
    """
    Write a minimal best_params.json to a temp file and yield its path.
    Used to test FraudModel param loading without touching the real models/ dir.
    """
    params = {
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20,
        "learning_rate": 0.05,
        "n_estimators": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(params, f)
        path = f.name

    yield path

    os.unlink(path)


@pytest.fixture(scope="session")
def mock_artifacts(raw_df, fitted_pipeline):
    """
    Train a tiny LightGBM model on synthetic data and return
    (model, pipeline) — ready to be injected into the FastAPI artifacts dict.
    """
    from src.models.fraud_model import FraudModel

    split = int(len(raw_df) * 0.8)
    train_df = raw_df.iloc[:split].copy()
    val_df   = raw_df.iloc[split:].copy()
    y_train  = train_df["isFraud"]
    y_val    = val_df["isFraud"]

    X_train = fitted_pipeline.transform(train_df)
    X_val   = fitted_pipeline.transform(val_df)

    model = FraudModel()
    model.train(X_train, y_train, X_val, y_val)
    model.evaluate(X_val, y_val)

    return model, fitted_pipeline
