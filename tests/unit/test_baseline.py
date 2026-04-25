"""Unit tests for `fraud_engine.models.baseline`.

Every test runs on a small synthetic DataFrame so the suite stays
data-free and fast. Integration coverage (real 10k sample, AUC
bounds, leakage shuffle test) lives in
`tests/integration/test_sprint1_baseline.py`.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from fraud_engine.config.settings import Settings, get_settings
from fraud_engine.models.baseline import BaselineResult, train_baseline


def _make_synthetic_merged(
    *,
    n_rows: int = 3000,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a synthetic merged frame with modelling signal.

    Fraud correlates with a high-amount tail plus a minority product
    code, so LightGBM can fit non-trivial AUC. TransactionDT spans
    roughly 180 days of seconds so temporal_split has room to carve
    out three non-empty windows on default Settings.
    """
    rng = np.random.default_rng(seed)
    dt = np.sort(rng.integers(0, 86400 * 180, size=n_rows))
    amt = np.exp(rng.normal(0, 1, size=n_rows)) * 50.0
    product = rng.choice(
        ["W", "C", "H", "R", "S"],
        size=n_rows,
        p=[0.5, 0.2, 0.15, 0.1, 0.05],
    )
    card_brand = rng.choice(
        ["visa", "mastercard", "amex", "discover"],
        size=n_rows,
    )
    fraud_logit = 0.6 * (amt / amt.mean() - 1) + 1.2 * (product == "C") - 3.0
    fraud_prob = 1.0 / (1.0 + np.exp(-fraud_logit))
    is_fraud = (rng.uniform(0, 1, size=n_rows) < fraud_prob).astype(np.int64)

    return pd.DataFrame(
        {
            "TransactionID": np.arange(n_rows, dtype=np.int64),
            "TransactionDT": dt.astype(np.int64),
            "TransactionAmt": amt.astype(np.float32),
            "ProductCD": pd.Categorical(product),
            "card4": pd.Categorical(card_brand),
            "C1": rng.integers(0, 100, size=n_rows).astype(np.int32),
            "C2": rng.integers(0, 50, size=n_rows).astype(np.int32),
            "V1": rng.normal(0, 1, size=n_rows).astype(np.float32),
            "V2": rng.normal(0, 1, size=n_rows).astype(np.float32),
            "isFraud": is_fraud,
        }
    )


@pytest.fixture
def baseline_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Settings]:
    """Settings with *fully* isolated paths — including MLflow.

    The shared `mock_settings` fixture isolates DATA_DIR / MODELS_DIR
    / LOGS_DIR but leaves `mlflow_tracking_uri` on its default of
    `./mlruns`. `train_baseline` calls `configure_mlflow()` which in
    turn reads `get_settings().mlflow_tracking_uri`, so we have to
    redirect that one too or the test run lands MLflow artefacts in
    the repo root.
    """
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    logs_dir = tmp_path / "logs"
    mlruns = tmp_path / "mlruns"
    for sub in ("raw", "interim", "processed"):
        (data_dir / sub).mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("MODELS_DIR", str(models_dir))
    monkeypatch.setenv("LOGS_DIR", str(logs_dir))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", str(mlruns))
    monkeypatch.setenv("SEED", "42")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    # Use a short val window so temporal_split fits inside the
    # synthetic 180-day span. 60 days train + 30 days val + 90 days
    # test keeps all three slices non-empty on 3000 rows.
    monkeypatch.setenv("TRAIN_END_DT", str(86400 * 60))
    monkeypatch.setenv("VAL_END_DT", str(86400 * 90))

    get_settings.cache_clear()
    settings = Settings()
    settings.ensure_directories()
    yield settings
    get_settings.cache_clear()
    # Clear any MLflow experiment cache so neighbouring tests start fresh.
    mlflow.set_tracking_uri("./mlruns")


class TestTrainBaselineContract:
    """Shape / contract checks for `train_baseline`."""

    def test_returns_result_on_random_variant(self, baseline_settings: Settings) -> None:
        df = _make_synthetic_merged()
        result = train_baseline(df, variant="random", settings=baseline_settings)
        assert isinstance(result, BaselineResult)
        assert result.variant == "random"
        assert result.model_path.is_file()
        assert 0.0 <= result.auc <= 1.0
        assert result.feature_importances
        assert re.fullmatch(r"[0-9a-f]{64}", result.content_hash)

    def test_returns_result_on_temporal_variant(self, baseline_settings: Settings) -> None:
        df = _make_synthetic_merged()
        result = train_baseline(df, variant="temporal", settings=baseline_settings)
        assert result.variant == "temporal"
        assert result.model_path.is_file()
        assert 0.0 <= result.auc <= 1.0
        assert result.feature_importances

    def test_invalid_variant_raises(self, baseline_settings: Settings) -> None:
        df = _make_synthetic_merged()
        with pytest.raises(ValueError, match="variant='bogus'"):
            train_baseline(df, variant="bogus", settings=baseline_settings)  # type: ignore[arg-type]

    def test_missing_column_raises(self, baseline_settings: Settings) -> None:
        df = _make_synthetic_merged().drop(columns=["TransactionID"])
        with pytest.raises(KeyError, match="TransactionID"):
            train_baseline(df, variant="random", settings=baseline_settings)


class TestMLflowLogging:
    """Verify the baseline writes a run to the configured tracking URI."""

    def test_opens_mlflow_run_with_variant_tag(self, baseline_settings: Settings) -> None:
        df = _make_synthetic_merged()
        train_baseline(df, variant="random", settings=baseline_settings)
        runs = mlflow.search_runs(
            experiment_names=[baseline_settings.mlflow_experiment_name],
        )
        assert not runs.empty
        assert (runs["tags.variant"] == "random").any()
        assert (runs["tags.stage"] == "sprint1_baseline").any()

    def test_logs_auc_metric(self, baseline_settings: Settings) -> None:
        df = _make_synthetic_merged()
        result = train_baseline(df, variant="temporal", settings=baseline_settings)
        runs = mlflow.search_runs(
            experiment_names=[baseline_settings.mlflow_experiment_name],
        )
        # The first (most recent) row should be our temporal run.
        auc_logged = float(runs.iloc[0]["metrics.auc"])
        assert auc_logged == pytest.approx(result.auc)


class TestModelArtefact:
    """Model-file-on-disk invariants."""

    def test_file_name_carries_content_hash_prefix(self, baseline_settings: Settings) -> None:
        df = _make_synthetic_merged()
        result = train_baseline(df, variant="random", settings=baseline_settings)
        expected_suffix = f"{result.content_hash[:12]}.joblib"
        assert result.model_path.name.endswith(expected_suffix)
        assert result.model_path.name.startswith("baseline_random_")

    def test_feature_importances_are_top_20_at_most(self, baseline_settings: Settings) -> None:
        df = _make_synthetic_merged()
        result = train_baseline(df, variant="random", settings=baseline_settings)
        assert 1 <= len(result.feature_importances) <= 20
        # Descending by value — critical because the sprint report
        # publishes the top-k verbatim.
        values = list(result.feature_importances.values())
        assert values == sorted(values, reverse=True)
