"""Deterministic-seeding helpers.

`set_all_seeds(seed)` seeds every RNG Python, NumPy, and PyTorch use so
that model training, data splits, and sampling are byte-reproducible
across runs. LightGBM is not seeded globally (no such API) — callers
pass `random_state=seed` to `LGBMClassifier` / `LGBMRegressor` or
`seed=seed` in training params. This is documented in the docstring so
downstream modules don't silently forget.

Business rationale:
    Reproducibility is non-negotiable for model audits and regression
    diagnosis. When a Sprint 3 evaluation shifts by 0.5 AUC between two
    runs with the same code + data, we need to know it's real drift and
    not RNG jitter. Seeding every source of randomness makes that
    possible.

Trade-offs considered:
    - Setting `torch.backends.cudnn.deterministic = True` costs ~10-20%
      throughput on GPU training vs. the non-deterministic kernels.
      Accepted: Sprint 3 training runs are infrequent, but any
      difference between two runs must be explainable.
    - We seed the legacy `np.random.seed()` for backward compatibility
      with libraries that use the global singleton. New code should
      use `np.random.default_rng(seed)` for a dedicated Generator.
"""

from __future__ import annotations

import os
import random

import numpy as np

from fraud_engine.config.settings import get_settings


def set_all_seeds(seed: int | None = None) -> int:
    """Seed every RNG used in the project and return the effective seed.

    Seeds:
        - Python's built-in `random`
        - NumPy (legacy global + default_rng state is unaffected by
          design; callers that need a dedicated Generator must pass
          the returned seed to `np.random.default_rng(seed)` explicitly)
        - `PYTHONHASHSEED` environment variable — takes effect on next
          interpreter boot, not the current one; set here so subprocess
          calls inherit determinism
        - PyTorch CPU + CUDA (guarded by `torch.cuda.is_available()`)
        - cuDNN: sets `deterministic=True`, `benchmark=False`

    **LightGBM is not seeded globally.** Pass `random_state=seed` or
    `params={"seed": seed}` to the estimator explicitly. This is a
    library-side constraint, not a choice.

    Args:
        seed: Seed value. If None, uses `settings.seed` (default 42).

    Returns:
        The effective seed that was used.
    """
    effective_seed = seed if seed is not None else get_settings().seed

    # Python stdlib and env var
    random.seed(effective_seed)
    os.environ["PYTHONHASHSEED"] = str(effective_seed)

    # NumPy — legacy global RNG
    np.random.seed(effective_seed)

    # PyTorch — CPU and (if available) CUDA
    _seed_torch(effective_seed)

    return effective_seed


def _seed_torch(seed: int) -> None:
    """Seed PyTorch CPU and CUDA state.

    Split out so tests can monkey-patch just this helper without
    shadowing the entire `torch` module. Imports are lazy because
    torch import is heavy (~1s cold start).

    Args:
        seed: Seed to apply.
    """
    import torch  # noqa: PLC0415 — lazy import is intentional

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
