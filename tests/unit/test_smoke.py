"""Sprint 0 smoke tests.

These tests verify the bootstrap skeleton is intact:
    - Package and every submodule import cleanly
    - Settings loads with defaults
    - `ensure_directories()` is idempotent and actually creates dirs
    - `get_logger(...)` can emit a record without raising
    - `set_all_seeds(seed)` produces reproducible numpy output

If any of these fail, Sprint 0 acceptance fails.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

import numpy as np
import pytest

import fraud_engine
from fraud_engine.config.settings import Settings


def _iter_submodules() -> list[str]:
    """Return the fully-qualified names of every module under fraud_engine.

    Returns:
        A list of dotted module paths, e.g. `fraud_engine.config.settings`.
    """
    pkg = fraud_engine
    prefix = pkg.__name__ + "."
    return [name for _, name, _ in pkgutil.walk_packages(pkg.__path__, prefix=prefix)]


def test_package_version_exposed() -> None:
    """The top-level package exposes __version__."""
    assert fraud_engine.__version__ == "0.1.0"


@pytest.mark.parametrize("module_name", _iter_submodules())
def test_submodule_importable(module_name: str) -> None:
    """Every submodule imports without side-effect errors."""
    importlib.import_module(module_name)


def test_settings_loads_defaults(mock_settings: Settings) -> None:
    """Settings initialises with defaults from the tmp-backed env."""
    assert mock_settings.seed == 42
    assert 0.0 <= mock_settings.decision_threshold <= 1.0
    assert mock_settings.fraud_cost_usd > 0
    assert mock_settings.fp_cost_usd > 0
    # log_level is normalised to upper by the validator.
    assert mock_settings.log_level == "DEBUG"


def test_settings_ensure_directories(mock_settings: Settings) -> None:
    """ensure_directories creates raw/interim/processed and is idempotent."""
    for path in (mock_settings.raw_dir, mock_settings.interim_dir, mock_settings.processed_dir):
        assert path.exists(), f"{path} should exist after ensure_directories"
        assert path.is_dir()

    # Second call must not raise.
    mock_settings.ensure_directories()


def test_logger_emits(
    mock_settings: Settings,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Configuring the logger and emitting a record does not raise.

    Also asserts the configured file handler actually writes the
    record to disk, since Sprint 2+ pipelines rely on that mirror.
    """
    from fraud_engine.utils.logging import configure_logging, get_logger

    run_id = configure_logging(pipeline_name="smoke", log_dir=tmp_path / "logs")
    logger = get_logger(__name__)
    logger.info("smoke_test_event", foo="bar", count=3)

    expected_log = tmp_path / "logs" / "smoke" / f"{run_id}.log"
    # Flush all handlers so the assertion sees the write.
    import logging as _logging

    for handler in _logging.getLogger().handlers:
        handler.flush()

    assert expected_log.exists()
    contents = expected_log.read_text(encoding="utf-8")
    assert "smoke_test_event" in contents


def test_set_all_seeds_is_reproducible() -> None:
    """Resetting the seed between draws produces identical sequences."""
    from fraud_engine.utils.seeding import set_all_seeds

    set_all_seeds(123)
    first = np.random.rand(5)
    set_all_seeds(123)
    second = np.random.rand(5)

    np.testing.assert_allclose(first, second)


def test_set_all_seeds_returns_effective_seed() -> None:
    """The helper returns the seed it used, even when None is passed."""
    from fraud_engine.utils.seeding import set_all_seeds

    used = set_all_seeds(None)
    assert isinstance(used, int)
