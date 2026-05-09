"""Inference service: model + calibrator + threshold-based decision.

Sprint 5 prompt 5.1.d: the prediction half of the request path. Loads
the production LightGBM model + isotonic calibrator at startup,
exposes `predict(features) → InferenceResult`, supports atomic
mid-session model reload via single-attribute swap.

This is the bridge between Sprint 5.1.c's `FeatureService.get_features(...)`
and Sprint 5.1.a's `PredictionResponse`. The decision rule is the
cost-curve-derived `Settings.decision_threshold` (post-Sprint-4.4:
0.080000); calibration is the load-bearing dependency per ADR 0003.

Business rationale:
    The feature pipeline (Sprint 5.1.c) produces a 743-column DataFrame
    ready for `LightGBMFraudModel.predict_proba`. The model returns
    raw probabilities calibrated against `scale_pos_weight` (the
    class-imbalance compensation applied during training). To map
    those raw probabilities to a real-population probability suitable
    for the cost-curve threshold, the isotonic calibrator (Sprint
    3.3.c) must be applied. Without it, the threshold is comparing
    apples to oranges — the cost-curve was derived against calibrated
    probabilities (Sprint 4.4), so a raw-probability threshold would
    produce systematically wrong decisions.

    The reload contract matters because production deployments
    eventually re-train the model. Restarting the API process every
    time is operationally expensive (loses connection pools, request
    queues, observability state). Atomic mid-session reload — replace
    the artefact bundle in one assignment, keep serving requests with
    zero downtime — is the right operational primitive.

Trade-offs considered:
    - **Atomic single-attribute swap (GIL-safe), not lock-protected.**
      Python's GIL guarantees a single-attribute rebind is atomic;
      `predict` binds a local alias at the top of the method so a
      concurrent reload only affects the next call. Three options
      were considered: (A) atomic swap, (B) `asyncio.Lock` around
      both paths, (C) generation counter + retry. (A) wins on
      simplicity + zero lock overhead on the read path. The
      `_Artefacts` frozen dataclass holds (model, calibrator,
      content_hash) — one swap covers all three.
    - **Threshold bound at construction, not read per-request.**
      `get_settings()` is `lru_cached` so the per-request cost is
      negligible, but binding once at construction makes the
      decision rule reproducible across the process lifetime (no
      surprise threshold change mid-session if `.env` is hot-edited).
      Mirrors `EconomicCostModel`'s snapshot-semantics precedent
      (Sprint 4.1). `set_threshold(value)` provides explicit override
      for tests + future ops.
    - **Output is a frozen dataclass, not a tuple.** Tuples lose
      attribute names; consumers would have to remember positional
      order. Pydantic model would add I/O surface; this is internal
      so a frozen dataclass (with `slots=True`) is the right weight.
      Mirrors `FeatureVector` (Sprint 5.1.c).
    - **`predict` is synchronous, not async.** LightGBM's
      `predict_proba` is CPU-bound (no I/O). Sprint 5.1.e's FastAPI
      route will call it via `run_in_executor` if needed. The
      synchronous interface keeps the API surface clean and tests
      trivial; an async wrapper that just `await`s a sync call adds
      noise without latency benefit.
    - **Single-row contract on `predict(features)`.** The DataFrame
      has shape `(1, 743)` — one transaction per call. Multi-row
      callers must call once per row. Matches the request-time API
      pattern; batch prediction is Sprint 5.x territory.
    - **`model_version` is the manifest's `content_hash`** (SHA-256
      hex of the joblib bytes). Immutable per artefact: any joblib
      re-save changes the hash even if the model parameters are
      identical (bit-level re-serialisation). The calibrator carries
      no version field; if 5.x re-fits the calibrator without
      re-fitting the model, the model's hash won't change — Sprint
      5.x can add calibrator versioning if that use case emerges.

Module surface (re-exported from `fraud_engine.api`):
    - InferenceService
    - InferenceResult

Cross-references:
    - `src/fraud_engine/models/lightgbm_model.py` — `LightGBMFraudModel.load()`
      + `predict_proba()` contract.
    - `src/fraud_engine/evaluation/calibration.py` — `Calibrator` type
      + `transform()` signature.
    - `src/fraud_engine/api/schemas.py` (Sprint 5.1.a) — `DecisionLiteral`
      reused.
    - `src/fraud_engine/api/feature_service.py` (Sprint 5.1.c) —
      frozen-dataclass + `@log_call` patterns.
    - `scripts/run_economic_evaluation.py:258-284` — canonical
      load-model-and-calibrator pattern this service mirrors.
    - `docs/ADR/0003-economic-threshold.md` — calibration-as-load-bearing
      dependency rationale.
    - `CLAUDE.md` §3 (latency budget), §5.5 (logging discipline), §8
      (cost defaults).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Final, cast

import joblib
import numpy as np
import pandas as pd

from fraud_engine.api.schemas import DecisionLiteral
from fraud_engine.config.settings import Settings, get_settings
from fraud_engine.evaluation.calibration import Calibrator
from fraud_engine.models.lightgbm_model import LightGBMFraudModel
from fraud_engine.utils.logging import get_logger, log_call

# ---------------------------------------------------------------------
# Module constants.
# ---------------------------------------------------------------------

_DEFAULT_MODEL_DIR: Final[Path] = Path("models/sprint3")
_MODEL_FILENAME: Final[str] = "lightgbm_model.joblib"
_CALIBRATOR_FILENAME: Final[str] = "calibrator.joblib"
_MANIFEST_FILENAME: Final[str] = "lightgbm_model_manifest.json"

# Probability bounds for the score guard.
_PROB_LOWER: Final[float] = 0.0
_PROB_UPPER: Final[float] = 1.0

_logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Frozen dataclasses.
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class _Artefacts:
    """Bundle of (model, calibrator, content_hash) for atomic swap.

    Held in `InferenceService._artefacts`; replaced wholesale on
    `reload()`. Frozen + slots gives us:
      - Immutability: a held reference to an old bundle continues to
        work even if `_artefacts` is replaced.
      - Memory: single tuple-like object instead of a dict.
      - Speed: attribute access via slot is faster than dict lookup.
    """

    model: LightGBMFraudModel
    calibrator: Calibrator
    content_hash: str


@dataclasses.dataclass(frozen=True, slots=True)
class InferenceResult:
    """Output of `InferenceService.predict(...)`.

    Attributes:
        probability: Calibrated fraud probability in `[0, 1]`. NOT the
            raw `predict_proba` output — the isotonic calibrator has
            been applied per ADR 0003.
        decision: `"block"` iff `probability >= threshold`, else
            `"allow"`. Mirrors `PredictionResponse.decision`.
        model_version: The model artefact's `content_hash` (SHA-256
            hex of the joblib bytes). Suitable for
            `PredictionResponse.model_version`.
    """

    probability: float
    decision: DecisionLiteral
    model_version: str


# ---------------------------------------------------------------------
# Module-private helpers.
# ---------------------------------------------------------------------


def _load_artefacts(model_dir: Path) -> _Artefacts:
    """Read joblib model + calibrator + manifest content_hash.

    Mirrors the pattern in
    `scripts/run_economic_evaluation.py:258-284`. Each artefact's
    absence raises a distinct `FileNotFoundError` with the resolved
    path in the message — operationally helpful for "which file did I
    forget to deploy" triage.

    Args:
        model_dir: Directory containing `lightgbm_model.joblib`,
            `calibrator.joblib`, and `lightgbm_model_manifest.json`.

    Returns:
        Frozen `_Artefacts` bundle.

    Raises:
        FileNotFoundError: If any of the three artefact files is
            missing.
        ValueError: If the manifest is missing the `content_hash` key
            or it is empty.
    """
    model_path = model_dir / _MODEL_FILENAME
    calibrator_path = model_dir / _CALIBRATOR_FILENAME
    manifest_path = model_dir / _MANIFEST_FILENAME

    if not model_path.is_file():
        raise FileNotFoundError(
            f"InferenceService: model joblib not found at {model_path} — "
            f"run `uv run python scripts/train_lightgbm.py` first."
        )
    if not calibrator_path.is_file():
        raise FileNotFoundError(
            f"InferenceService: calibrator joblib not found at "
            f"{calibrator_path} — run `train_lightgbm.py`'s "
            f"`select_calibration_method` step first."
        )
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"InferenceService: model manifest not found at "
            f"{manifest_path} — manifest is written alongside the "
            f"joblib by `LightGBMFraudModel.save()`."
        )

    model = LightGBMFraudModel.load(model_dir)
    # The calibrator is a union type (`PlattScaler | IsotonicCalibrator
    # | _IdentityCalibrator`); cast for mypy.
    calibrator = cast(Calibrator, joblib.load(calibrator_path))

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    content_hash = manifest.get("content_hash")
    if not isinstance(content_hash, str) or not content_hash:
        raise ValueError(
            f"InferenceService: manifest at {manifest_path} missing "
            f"or malformed `content_hash` field; got {content_hash!r}"
        )

    return _Artefacts(model=model, calibrator=calibrator, content_hash=content_hash)


# ---------------------------------------------------------------------
# Public class.
# ---------------------------------------------------------------------


class InferenceService:
    """Production inference service for the fraud-scoring API.

    Composes:
        - LightGBM model (`predict_proba`).
        - Isotonic calibrator (`transform`) — applied to the raw
          probabilities to produce calibrated scores per ADR 0003.
        - Decision threshold from `Settings.decision_threshold` —
          `"block"` iff `score >= threshold`.

    Lifecycle:
        - `__init__` is cheap; no I/O. `load()` opens the artefacts.
        - `predict(features)` is the hot path; CPU-bound, synchronous.
        - `reload()` re-reads the artefacts from disk and atomically
          replaces the in-memory bundle. Concurrent `predict()` calls
          binding the artefacts at the top of the method see either
          the old or the new bundle, never a torn read (GIL-atomic
          single-attribute rebind).
        - `set_threshold(value)` updates the decision threshold
          mid-session — primarily for tests + future ops.

    Business rationale + trade-offs considered: see module docstring.
    """

    def __init__(
        self,
        model_dir: Path | None = None,
        threshold: float | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Configure the service; does NOT load the model artefacts.

        Args:
            model_dir: Override the model-artefacts directory. None →
                `models/sprint3/`.
            threshold: Override `Settings.decision_threshold`. None →
                resolve from Settings at construction time.
            settings: Inject a Settings instance for tests. None →
                `get_settings()`.

        Raises:
            ValueError: If `threshold` is outside [0, 1].
        """
        self._settings: Settings = settings if settings is not None else get_settings()
        self._model_dir: Path = model_dir if model_dir is not None else _DEFAULT_MODEL_DIR
        resolved_threshold = (
            threshold if threshold is not None else self._settings.decision_threshold
        )
        _validate_threshold(resolved_threshold)
        self._threshold: float = float(resolved_threshold)
        # Artefacts populated by `load()` / `reload()`. None → not loaded.
        self._artefacts: _Artefacts | None = None

    # ---------- lifecycle ----------------------------------------------

    @log_call
    def load(self) -> None:
        """Read joblib model + calibrator + manifest from disk.

        Idempotent: a second call replaces the in-memory bundle (same
        atomic-swap semantic as `reload()`).

        Raises:
            FileNotFoundError: If any artefact is missing.
            ValueError: If the manifest's `content_hash` is missing
                or empty.
        """
        new_artefacts = _load_artefacts(self._model_dir)
        # Single-attribute rebind — GIL-atomic. A concurrent `predict`
        # binding `local = self._artefacts` at method top sees either
        # the old bundle (if its bind happens before this assignment)
        # or the new one (after); never a partially-replaced view.
        self._artefacts = new_artefacts

    @log_call
    def reload(self) -> None:
        """Re-read all artefacts from disk and atomically swap.

        Functionally identical to `load()`; the separate name signals
        intent ("this is a deliberate mid-session model swap, not a
        startup load").
        """
        self.load()

    # ---------- the hot path -------------------------------------------

    @log_call
    def predict(self, features: pd.DataFrame) -> InferenceResult:
        """Score a single transaction; return calibrated probability + decision.

        Args:
            features: Single-row DataFrame with columns matching
                `model.feature_names_in_` (typically produced by
                `FeatureService.get_features(...)`'s `df` field).

        Returns:
            `InferenceResult(probability, decision, model_version)`.

        Raises:
            RuntimeError: If `load()` has not been called.
            KeyError: Propagated from
                `LightGBMFraudModel.predict_proba` when feature
                columns are missing.
            ValueError: If the calibrated probability falls outside
                `[0, 1]` (defensive — the calibrator's output should
                always be in range; a violation indicates a bug).
        """
        # Bind a local alias FIRST. A concurrent `reload()` only
        # affects the next call; this one is consistent.
        local = self._artefacts
        if local is None:
            raise RuntimeError(
                "InferenceService: call load() before predict(); the "
                "model artefacts have not been read from disk yet."
            )

        # Score: predict_proba returns (n, 2); column 1 is fraud-prob.
        proba_raw: np.ndarray[Any, Any] = local.model.predict_proba(features)[:, 1]
        # Apply calibration per ADR 0003.
        proba_cal: np.ndarray[Any, Any] = local.calibrator.transform(proba_raw)
        score = float(proba_cal[0])
        # Defensive guard: calibrators clip to [0, 1] but a future
        # implementation might not.
        if score < _PROB_LOWER or score > _PROB_UPPER:
            raise ValueError(
                f"InferenceService: calibrated probability {score} outside "
                f"[0, 1] — calibrator contract violated."
            )

        decision: DecisionLiteral = "block" if score >= self._threshold else "allow"
        return InferenceResult(
            probability=score,
            decision=decision,
            model_version=local.content_hash,
        )

    # ---------- threshold + version accessors ---------------------------

    def set_threshold(self, value: float) -> None:
        """Update the decision threshold mid-session.

        Primarily for tests + future ops (e.g. an emergency rollback
        to a more conservative threshold). The threshold is bound at
        construction by default; this is the explicit override path.

        Args:
            value: New threshold; must be in `[0, 1]`.

        Raises:
            ValueError: If `value` is outside `[0, 1]`.
        """
        _validate_threshold(value)
        self._threshold = float(value)

    @property
    def threshold(self) -> float:
        """Current decision threshold."""
        return self._threshold

    @property
    def model_version(self) -> str:
        """The currently-loaded model's `content_hash`.

        Raises:
            RuntimeError: If `load()` has not been called.
        """
        if self._artefacts is None:
            raise RuntimeError("InferenceService: call load() before accessing model_version.")
        return self._artefacts.content_hash


# ---------------------------------------------------------------------
# Module-private validators.
# ---------------------------------------------------------------------


def _validate_threshold(value: float) -> None:
    """Enforce `0 <= threshold <= 1` at construction + on `set_threshold`."""
    if not _PROB_LOWER <= value <= _PROB_UPPER:
        raise ValueError(f"InferenceService: threshold must be in [0, 1], got {value}")


__all__ = ["InferenceResult", "InferenceService"]
