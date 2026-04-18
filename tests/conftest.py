"""Shared pytest fixtures.

Every fixture here is available to any test in this repo. The goal is
to give tests access to deterministic small DataFrames and isolated
filesystem / settings state so no test ever writes to the real
`data/` directory.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_engine.config.settings import Settings, get_settings

# Stable constants for fixture data. Extracted so tests can import them
# if they need to assert on fixture shape without reinstantiating it.
_N_ROWS_SMALL: int = 20
_FIXTURE_SEED: int = 42


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Return an isolated data directory with raw/interim/processed subdirs.

    Each test gets its own `tmp_path`, so fixture writes do not leak
    between tests. Shape mirrors `Settings.ensure_directories()` so
    tests that accept a Settings object can point at this tree.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        Path to the root data directory.
    """
    data = tmp_path / "data"
    for sub in ("raw", "interim", "processed"):
        (data / sub).mkdir(parents=True, exist_ok=True)
    return data


@pytest.fixture
def mock_settings(
    tmp_data_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Settings]:
    """Yield a Settings instance pointing at tmp directories.

    Env vars are set via monkeypatch so the Settings class re-reads a
    fresh environment. The `get_settings` lru_cache is cleared before
    and after so neighbouring tests see their own values, not this
    test's.

    Yields:
        A configured Settings instance whose writes never touch the
        real repo.
    """
    monkeypatch.setenv("DATA_DIR", str(tmp_data_dir))
    monkeypatch.setenv("SEED", "42")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    get_settings.cache_clear()
    settings = Settings(
        data_dir=tmp_data_dir,
        models_dir=tmp_path / "models",
        logs_dir=tmp_path / "logs",
    )
    settings.ensure_directories()
    yield settings
    get_settings.cache_clear()


@pytest.fixture
def small_transactions_df() -> pd.DataFrame:
    """Return 20 rows of synthetic transaction data.

    Deterministic via a fixed seed so assertions on specific values
    remain stable across runs. Shape intentionally matches the expected
    Sprint 1 transaction schema so feature tests can slot in.

    Returns:
        DataFrame with columns: transaction_id, user_id, amount,
        currency, merchant_id, timestamp, is_fraud.
    """
    rng = np.random.default_rng(_FIXTURE_SEED)
    base_time = datetime(2026, 1, 1, tzinfo=UTC)
    return pd.DataFrame(
        {
            "transaction_id": [f"tx_{i:05d}" for i in range(_N_ROWS_SMALL)],
            "user_id": [f"u_{rng.integers(0, 5):03d}" for _ in range(_N_ROWS_SMALL)],
            "amount": rng.uniform(1.0, 500.0, size=_N_ROWS_SMALL).round(2),
            "currency": rng.choice(["USD", "EUR", "GBP"], size=_N_ROWS_SMALL),
            "merchant_id": [f"m_{rng.integers(0, 8):03d}" for _ in range(_N_ROWS_SMALL)],
            "timestamp": [
                base_time + timedelta(minutes=int(m)) for m in rng.integers(0, 10000, _N_ROWS_SMALL)
            ],
            # ~15% fraud rate — above real base rates but keeps tests stable
            # at small N. Real prevalence is ~0.5-2%; Sprint 2 tests must use
            # realistic rates.
            "is_fraud": rng.choice([0, 1], size=_N_ROWS_SMALL, p=[0.85, 0.15]).astype(int),
        }
    )


@pytest.fixture
def small_identity_df() -> pd.DataFrame:
    """Return 20 rows of synthetic identity data keyed on user_id.

    Keys align with `small_transactions_df.user_id` (u_000..u_004) so
    downstream join tests have non-empty matches.

    Returns:
        DataFrame with columns: user_id, device_id, ip_address,
        email_domain, account_age_days.
    """
    rng = np.random.default_rng(_FIXTURE_SEED + 1)
    user_ids = [f"u_{i:03d}" for i in range(_N_ROWS_SMALL)]
    return pd.DataFrame(
        {
            "user_id": user_ids,
            "device_id": [f"d_{rng.integers(0, 50):04d}" for _ in range(_N_ROWS_SMALL)],
            "ip_address": [
                f"10.{rng.integers(0, 256)}.{rng.integers(0, 256)}.{rng.integers(0, 256)}"
                for _ in range(_N_ROWS_SMALL)
            ],
            "email_domain": rng.choice(
                ["gmail.com", "yahoo.com", "outlook.com", "proton.me"], size=_N_ROWS_SMALL
            ),
            "account_age_days": rng.integers(1, 3650, size=_N_ROWS_SMALL),
        }
    )
