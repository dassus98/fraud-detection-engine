"""Tests for `fraud_engine.utils.seeding.set_all_seeds`.

These tests are the reproducibility backstop for every model we train.
If Sprint 3 ever sees two runs diverge in AUC, the first question is
"did the seeding contract hold?" — and the answer lives here.
"""

from __future__ import annotations

import os
import random

import numpy as np
import pytest

from fraud_engine.utils.seeding import set_all_seeds


class TestReproducibility:
    """set_all_seeds seeds stdlib random + NumPy + PYTHONHASHSEED."""

    def test_numpy_legacy_global_is_reproducible(self) -> None:
        """Two numpy draws after identical seeds produce identical arrays."""
        set_all_seeds(7)
        first = np.random.rand(10)
        set_all_seeds(7)
        second = np.random.rand(10)
        np.testing.assert_array_equal(first, second)

    def test_python_random_is_reproducible(self) -> None:
        """Two stdlib-random draws after identical seeds match exactly."""
        set_all_seeds(11)
        first = [random.random() for _ in range(10)]
        set_all_seeds(11)
        second = [random.random() for _ in range(10)]
        assert first == second

    def test_different_seeds_diverge(self) -> None:
        """Two different seeds produce different sequences.

        Guards against a bug where a seed argument is accidentally
        ignored and a hard-coded default leaks in.
        """
        set_all_seeds(1)
        a = np.random.rand(10)
        set_all_seeds(2)
        b = np.random.rand(10)
        assert not np.allclose(a, b)

    def test_sets_python_hash_seed_env(self) -> None:
        """PYTHONHASHSEED is written to os.environ for subprocess inheritance."""
        set_all_seeds(99)
        assert os.environ["PYTHONHASHSEED"] == "99"

    def test_returns_explicit_seed_when_provided(self) -> None:
        """Passing an explicit int returns it back verbatim."""
        assert set_all_seeds(12345) == 12345

    def test_returns_settings_seed_when_none(self) -> None:
        """Passing None falls back to settings.seed and returns that value."""
        from fraud_engine.config.settings import get_settings

        effective = set_all_seeds(None)
        assert effective == get_settings().seed


class TestTorchReproducibility:
    """Torch-specific seeding: tensors match and cuDNN is deterministic."""

    def test_torch_tensors_identical_across_calls(self) -> None:
        """Two torch.randn draws after identical seeds produce equal tensors.

        Uses CPU only — CUDA availability is machine-dependent and not
        required for the contract this test enforces.
        """
        torch = pytest.importorskip("torch")
        set_all_seeds(42)
        first = torch.randn(16)
        set_all_seeds(42)
        second = torch.randn(16)
        assert torch.equal(first, second)

    def test_cudnn_deterministic_flag_is_set(self) -> None:
        """After seeding, cuDNN runs in deterministic mode with benchmark off.

        These two flags are the ones that matter for Sprint 3 model
        training reproducibility on GPU. Leaving `benchmark=True` can
        silently pick different kernels between runs.
        """
        torch = pytest.importorskip("torch")
        set_all_seeds(42)
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
