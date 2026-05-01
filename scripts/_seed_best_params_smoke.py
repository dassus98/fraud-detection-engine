"""One-shot seeder for `configs/model_best_params.yaml`.

Runs the 5-trial Optuna smoke against deterministic synthetic data
to populate `configs/model_best_params.yaml` with a real (small)
`best_params` block. Sprint 3 prompt 3.3.b user-elected option (b)
"5-trial smoke result for traceability" — the YAML carries the
output of this script as a placeholder until 3.3.d runs the full
100-trial sweep against Tier-5 features.

Run once; commit the resulting `configs/model_best_params.yaml`.

Usage:
    MLFLOW_TRACKING_URI=/tmp/seed_mlruns \
        uv run python scripts/_seed_best_params_smoke.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fraud_engine.models.tuning import run_tuning

_N_ROWS = 600
_N_FEATURES = 5
_FRAUD_RATE = 0.20
_SEED = 42
_OUTPUT_PATH = Path("configs") / "model_best_params.yaml"


def _make_synthetic_xy() -> tuple[pd.DataFrame, np.ndarray]:
    """Synthetic (X, y) pair mirroring the test fixture."""
    rng = np.random.default_rng(_SEED)
    cols = {f"f{i}": rng.normal(0, 1, size=_N_ROWS).astype(np.float32) for i in range(_N_FEATURES)}
    cols["amount"] = (np.exp(rng.normal(0, 1, size=_N_ROWS)) * 50.0).astype(np.float32)
    x_df = pd.DataFrame(cols)
    fraud_logit = (
        0.9 * (x_df["amount"].to_numpy() / x_df["amount"].mean() - 1) + 0.5 * x_df["f0"].to_numpy()
    )
    fraud_prob = 1.0 / (1.0 + np.exp(-fraud_logit))
    y = (rng.uniform(0, 1, size=_N_ROWS) < _FRAUD_RATE * fraud_prob / fraud_prob.mean()).astype(
        np.int64
    )
    if y.sum() == 0:
        y[0] = 1
    if y.sum() == _N_ROWS:
        y[0] = 0
    return x_df, y


def main() -> None:
    """Run the 5-trial smoke and write configs/model_best_params.yaml."""
    x_df, y = _make_synthetic_xy()
    idx = np.random.default_rng(_SEED).permutation(_N_ROWS)
    cut = int(_N_ROWS * 0.7)
    train_idx, val_idx = idx[:cut], idx[cut:]
    x_train = x_df.iloc[train_idx].reset_index(drop=True)
    x_val = x_df.iloc[val_idx].reset_index(drop=True)
    y_train, y_val = y[train_idx], y[val_idx]

    result = run_tuning(
        x_train,
        y_train,
        x_val,
        y_val,
        n_trials=5,
        study_name="lightgbm_fraud_tuning_smoke",
        output_path=_OUTPUT_PATH,
        random_state=_SEED,
        num_boost_round=30,
        early_stopping_rounds=5,
    )
    print(f"best_value     = {result['best_value']:.6f}")
    print(f"best_params    = {result['best_params']}")
    print(f"output_path    = {result['output_path']}")


if __name__ == "__main__":
    main()
