"""Unit tests for `fraud_engine.api.inference.InferenceService`.

Sprint 5 prompt 5.1.d verification surface.

Business rationale:
    The InferenceService is the load-bearing prediction primitive at
    request time. A regression here — a calibrator bypass, a wrong
    threshold comparison, a torn read during reload — leaks into the
    production decision path and either (a) makes wrong block/allow
    calls, (b) returns inconsistent model_version values, or (c)
    crashes mid-request. The one place those contracts are pinned
    is here.

Trade-offs considered:
    - **Real model + calibrator artefacts (skip-if-missing).** The
      fitted model joblib + calibrator joblib are too small to mock
      meaningfully and too important to NOT exercise. Tests skip if
      `models/sprint3/{lightgbm_model.joblib, calibrator.joblib,
      lightgbm_model_manifest.json}` are missing — matches the
      Sprint 5.1.c precedent.
    - **Mock the model only for calibrator-bypass tests.** The
      `TestCalibration` class needs to verify the calibrator is
      called; mocking `predict_proba` to return known values lets us
      assert the calibrator's effect on the output. All other tests
      use the real model.
    - **`threading.Thread` for concurrent-reload race test.** The
      reload contract is "GIL-atomic single-attribute swap"; the only
      way to verify under concurrent reads is to actually run them
      concurrently. Python's `threading` module gives us that without
      requiring asyncio plumbing.
"""

from __future__ import annotations

import dataclasses
import json
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from fraud_engine.api.inference import (
    InferenceResult,
    InferenceService,
    _Artefacts,
    _load_artefacts,
)
from fraud_engine.models.lightgbm_model import LightGBMFraudModel

# ---------------------------------------------------------------------
# Required artefacts. Skip if any is absent.
# ---------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODEL_DIR = _REPO_ROOT / "models" / "sprint3"
_MODEL_FILE = _MODEL_DIR / "lightgbm_model.joblib"
_CALIBRATOR_FILE = _MODEL_DIR / "calibrator.joblib"
_MANIFEST_FILE = _MODEL_DIR / "lightgbm_model_manifest.json"

# Pipeline + manifest used by FeatureService to build a feature frame.
_PIPELINE_DIR = _REPO_ROOT / "models" / "pipelines"
_PIPELINE_FILE = _PIPELINE_DIR / "tier1_pipeline.joblib"


def _require_artefacts() -> None:
    missing = [p for p in (_MODEL_FILE, _CALIBRATOR_FILE, _MANIFEST_FILE) if not p.exists()]
    if missing:
        pytest.skip(f"InferenceService artefacts missing: {missing}")


def _build_features() -> pd.DataFrame:
    """Build a single-row DataFrame matching the model's `feature_names_in_`.

    Reads the manifest to get the canonical column list and fills with
    zeros — LightGBM tolerates NaN/zero natively, so this is a valid
    input for `predict_proba` even if the values are not realistic.
    """
    manifest = json.loads(_MANIFEST_FILE.read_text(encoding="utf-8"))
    columns = manifest["feature_names"]
    return pd.DataFrame([[0.0] * len(columns)], columns=columns)


# ---------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------


@pytest.fixture
def loaded_service() -> InferenceService:
    """An InferenceService with artefacts loaded from disk."""
    _require_artefacts()
    s = InferenceService(model_dir=_MODEL_DIR, threshold=0.5)
    s.load()
    return s


@pytest.fixture
def features() -> pd.DataFrame:
    """A single-row DataFrame matching the model's column list."""
    _require_artefacts()
    return _build_features()


# ---------------------------------------------------------------------
# TestLoad — artefact loading + missing-file handling.
# ---------------------------------------------------------------------


class TestLoad:
    """`load()` succeeds with real artefacts; raises on missing files."""

    def test_load_succeeds(self) -> None:
        _require_artefacts()
        s = InferenceService(model_dir=_MODEL_DIR)
        s.load()
        assert s._artefacts is not None
        assert isinstance(s._artefacts, _Artefacts)
        assert isinstance(s._artefacts.content_hash, str)
        assert len(s._artefacts.content_hash) == 64  # SHA-256 hex

    def test_missing_model_raises(self, tmp_path: Path) -> None:
        """An empty model_dir → FileNotFoundError on the model joblib."""
        s = InferenceService(model_dir=tmp_path)
        with pytest.raises(FileNotFoundError, match="model joblib"):
            s.load()

    def test_missing_calibrator_raises(
        self,
        tmp_path: Path,
    ) -> None:
        """If model is present but calibrator is missing, raise on calibrator."""
        # Copy only the model joblib + manifest, not the calibrator.
        _require_artefacts()
        (tmp_path / "lightgbm_model.joblib").write_bytes(_MODEL_FILE.read_bytes())
        (tmp_path / "lightgbm_model_manifest.json").write_bytes(_MANIFEST_FILE.read_bytes())
        s = InferenceService(model_dir=tmp_path)
        with pytest.raises(FileNotFoundError, match="calibrator"):
            s.load()

    def test_missing_manifest_raises(self, tmp_path: Path) -> None:
        """If model + calibrator are present but manifest is missing, raise."""
        _require_artefacts()
        (tmp_path / "lightgbm_model.joblib").write_bytes(_MODEL_FILE.read_bytes())
        (tmp_path / "calibrator.joblib").write_bytes(_CALIBRATOR_FILE.read_bytes())
        s = InferenceService(model_dir=tmp_path)
        with pytest.raises(FileNotFoundError, match="manifest"):
            s.load()

    def test_malformed_manifest_raises(self, tmp_path: Path) -> None:
        """Manifest missing `content_hash` → ValueError."""
        _require_artefacts()
        (tmp_path / "lightgbm_model.joblib").write_bytes(_MODEL_FILE.read_bytes())
        (tmp_path / "calibrator.joblib").write_bytes(_CALIBRATOR_FILE.read_bytes())
        # Write a manifest without `content_hash`.
        (tmp_path / "lightgbm_model_manifest.json").write_text(
            json.dumps({"feature_names": ["foo"]}),
            encoding="utf-8",
        )
        s = InferenceService(model_dir=tmp_path)
        with pytest.raises(ValueError, match="content_hash"):
            s.load()


# ---------------------------------------------------------------------
# TestPredict — basic shape + happy path.
# ---------------------------------------------------------------------


class TestPredict:
    """`predict()` returns the right shape + types."""

    def test_returns_inference_result(
        self,
        loaded_service: InferenceService,
        features: pd.DataFrame,
    ) -> None:
        result = loaded_service.predict(features)
        assert isinstance(result, InferenceResult)

    def test_probability_in_unit_interval(
        self,
        loaded_service: InferenceService,
        features: pd.DataFrame,
    ) -> None:
        result = loaded_service.predict(features)
        assert 0.0 <= result.probability <= 1.0

    def test_decision_is_literal(
        self,
        loaded_service: InferenceService,
        features: pd.DataFrame,
    ) -> None:
        result = loaded_service.predict(features)
        assert result.decision in ("block", "allow")

    def test_model_version_is_content_hash(
        self,
        loaded_service: InferenceService,
        features: pd.DataFrame,
    ) -> None:
        """`model_version` matches the manifest's `content_hash`."""
        manifest = json.loads(_MANIFEST_FILE.read_text(encoding="utf-8"))
        expected = manifest["content_hash"]
        result = loaded_service.predict(features)
        assert result.model_version == expected
        assert result.model_version == loaded_service.model_version

    def test_predict_before_load_raises(self, features: pd.DataFrame) -> None:
        """Calling `predict` before `load()` raises `RuntimeError`."""
        _require_artefacts()
        s = InferenceService(model_dir=_MODEL_DIR)
        with pytest.raises(RuntimeError, match="load"):
            s.predict(features)

    def test_missing_columns_propagate_keyerror(
        self,
        loaded_service: InferenceService,
    ) -> None:
        """A frame with missing columns surfaces `predict_proba`'s KeyError."""
        bad_df = pd.DataFrame([{"only_one_col": 1.0}])
        with pytest.raises(KeyError):
            loaded_service.predict(bad_df)


# ---------------------------------------------------------------------
# TestDecisionThreshold — block/allow boundary.
# ---------------------------------------------------------------------


class _MockModel:
    """A minimal model substitute returning a fixed proba value."""

    def __init__(self, fraud_prob: float) -> None:
        self._fraud_prob = fraud_prob

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray[Any, Any]:  # noqa: N803 — `X` mirrors sklearn convention also used in `LightGBMFraudModel.predict_proba`
        n = len(X)
        return np.tile(np.array([1.0 - self._fraud_prob, self._fraud_prob]), (n, 1))


class _IdentityCalibrator:
    """A calibrator that passes probabilities through unchanged."""

    def transform(self, p: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return np.asarray(p, dtype=np.float64)


def _make_service_with_score(score: float, threshold: float) -> InferenceService:
    """Build a service with a mocked artefacts bundle yielding `score`."""
    _require_artefacts()
    s = InferenceService(model_dir=_MODEL_DIR, threshold=threshold)
    artefacts = _Artefacts(
        model=_MockModel(score),  # type: ignore[arg-type]  # _MockModel duck-types LightGBMFraudModel
        calibrator=_IdentityCalibrator(),  # type: ignore[arg-type]
        content_hash="0" * 64,
    )
    s._artefacts = artefacts
    return s


class TestDecisionThreshold:
    """Threshold comparison: `>=` is the inclusive boundary."""

    @pytest.mark.parametrize(
        ("score", "threshold", "expected_decision"),
        [
            (0.0, 0.08, "allow"),
            (0.07, 0.08, "allow"),
            (0.0799, 0.08, "allow"),  # just below
            (0.08, 0.08, "block"),  # boundary inclusive
            (0.5, 0.08, "block"),
            (1.0, 0.08, "block"),
        ],
    )
    def test_boundary_inclusive(
        self,
        score: float,
        threshold: float,
        expected_decision: str,
    ) -> None:
        s = _make_service_with_score(score, threshold)
        df = pd.DataFrame([{"x": 1.0}])
        result = s.predict(df)
        assert result.decision == expected_decision

    def test_set_threshold_updates_decision(
        self,
        loaded_service: InferenceService,
        features: pd.DataFrame,
    ) -> None:
        """`set_threshold(0.0)` should make every prediction block."""
        loaded_service.set_threshold(0.0)
        result = loaded_service.predict(features)
        assert result.decision == "block"
        # Reset for fixture cleanliness.
        loaded_service.set_threshold(1.0)
        result2 = loaded_service.predict(features)
        # At threshold=1.0, only score==1.0 blocks. Realistic features
        # produce score < 1.0 → allow.
        assert result2.decision == "allow"

    @pytest.mark.parametrize("bad_value", [-0.01, 1.01, -1.0, 2.0])
    def test_threshold_validation_raises(
        self,
        loaded_service: InferenceService,
        bad_value: float,
    ) -> None:
        """`set_threshold` rejects out-of-[0, 1]."""
        with pytest.raises(ValueError, match="threshold"):
            loaded_service.set_threshold(bad_value)

    def test_constructor_rejects_bad_threshold(self) -> None:
        """`InferenceService(threshold=...)` validates at construction."""
        _require_artefacts()
        with pytest.raises(ValueError, match="threshold"):
            InferenceService(model_dir=_MODEL_DIR, threshold=1.5)


# ---------------------------------------------------------------------
# TestCalibration — calibrator IS applied.
# ---------------------------------------------------------------------


class TestCalibration:
    """The calibrator must be invoked (raw probs ≠ output probs)."""

    def test_calibrator_is_applied(self) -> None:
        """Mock the calibrator's `transform`; verify it was called."""
        _require_artefacts()
        s = InferenceService(model_dir=_MODEL_DIR, threshold=0.5)
        # Build a mocked calibrator that transforms input → 0.42.
        mock_calibrator = MagicMock()
        mock_calibrator.transform = MagicMock(return_value=np.array([0.42]))
        artefacts = _Artefacts(
            model=_MockModel(0.99),  # type: ignore[arg-type]  # raw prob 0.99
            calibrator=mock_calibrator,  # type: ignore[arg-type]
            content_hash="0" * 64,
        )
        s._artefacts = artefacts
        result = s.predict(pd.DataFrame([{"x": 1.0}]))
        # The mocked calibrator's transform was called.
        mock_calibrator.transform.assert_called_once()
        # The output probability is the mocked-calibrator return, not raw 0.99.
        assert result.probability == 0.42
        # At threshold 0.5, 0.42 → allow (block iff >= 0.5).
        assert result.decision == "allow"

    def test_out_of_range_calibrated_prob_raises(self) -> None:
        """Defensive guard: calibrator returning >1 raises ValueError."""
        _require_artefacts()
        s = InferenceService(model_dir=_MODEL_DIR, threshold=0.5)
        mock_calibrator = MagicMock()
        mock_calibrator.transform = MagicMock(return_value=np.array([1.5]))
        artefacts = _Artefacts(
            model=_MockModel(0.5),  # type: ignore[arg-type]
            calibrator=mock_calibrator,  # type: ignore[arg-type]
            content_hash="0" * 64,
        )
        s._artefacts = artefacts
        with pytest.raises(ValueError, match="outside"):
            s.predict(pd.DataFrame([{"x": 1.0}]))

    def test_real_calibrator_runs(
        self,
        loaded_service: InferenceService,
        features: pd.DataFrame,
    ) -> None:
        """Real calibrator returns probability in [0, 1] (sanity)."""
        result = loaded_service.predict(features)
        assert 0.0 <= result.probability <= 1.0


# ---------------------------------------------------------------------
# TestReload — atomic mid-session swap.
# ---------------------------------------------------------------------


class TestReload:
    """Reload re-reads artefacts; concurrent predict doesn't tear."""

    def test_reload_completes(
        self,
        loaded_service: InferenceService,
    ) -> None:
        """Calling `reload()` after `load()` re-populates artefacts."""
        old_hash = loaded_service.model_version
        loaded_service.reload()
        # Same artefact on disk → same content_hash.
        assert loaded_service.model_version == old_hash

    def test_reload_swaps_artefacts(
        self,
        loaded_service: InferenceService,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Monkeypatch `_load_artefacts` to return a different hash; verify swap."""
        old_hash = loaded_service.model_version
        new_hash = "f" * 64
        # Build a fake artefact bundle with the new hash.
        new_artefacts = _Artefacts(
            model=loaded_service._artefacts.model,  # type: ignore[union-attr]
            calibrator=loaded_service._artefacts.calibrator,  # type: ignore[union-attr]
            content_hash=new_hash,
        )
        monkeypatch.setattr(
            "fraud_engine.api.inference._load_artefacts",
            lambda _: new_artefacts,
        )
        loaded_service.reload()
        assert loaded_service.model_version == new_hash
        assert loaded_service.model_version != old_hash

    def test_concurrent_predict_during_reload_doesnt_tear(
        self,
        loaded_service: InferenceService,
        features: pd.DataFrame,
    ) -> None:
        """N predict-threads + reload-thread for ~1s; zero crashes."""
        n_predict_threads = 4
        n_reloads = 50
        crashes: list[BaseException] = []
        results: list[InferenceResult] = []
        stop = threading.Event()

        def _predict_loop() -> None:
            try:
                while not stop.is_set():
                    results.append(loaded_service.predict(features))
            except BaseException as exc:  # noqa: BLE001 — race-test catch
                crashes.append(exc)

        def _reload_loop() -> None:
            try:
                for _ in range(n_reloads):
                    loaded_service.reload()
            except BaseException as exc:  # noqa: BLE001
                crashes.append(exc)

        predict_threads = [threading.Thread(target=_predict_loop) for _ in range(n_predict_threads)]
        reload_thread = threading.Thread(target=_reload_loop)

        for t in predict_threads:
            t.start()
        reload_thread.start()
        reload_thread.join()
        stop.set()
        for t in predict_threads:
            t.join(timeout=5.0)

        assert not crashes, f"thread crashes: {crashes}"
        assert len(results) > 0
        # Every result has a valid model_version (no torn read).
        for r in results:
            assert isinstance(r, InferenceResult)
            assert isinstance(r.model_version, str)
            assert len(r.model_version) == 64


# ---------------------------------------------------------------------
# TestModelVersion — accessor + immutability.
# ---------------------------------------------------------------------


class TestModelVersion:
    """`model_version` is the manifest's `content_hash`."""

    def test_matches_content_hash(
        self,
        loaded_service: InferenceService,
    ) -> None:
        manifest = json.loads(_MANIFEST_FILE.read_text(encoding="utf-8"))
        assert loaded_service.model_version == manifest["content_hash"]

    def test_raises_before_load(self) -> None:
        """Accessing `model_version` before `load()` raises `RuntimeError`."""
        _require_artefacts()
        s = InferenceService(model_dir=_MODEL_DIR)
        with pytest.raises(RuntimeError, match="load"):
            _ = s.model_version


# ---------------------------------------------------------------------
# TestInferenceResult — frozen dataclass.
# ---------------------------------------------------------------------


class TestInferenceResult:
    """`InferenceResult` is frozen + slots."""

    def test_frozen(self) -> None:
        result = InferenceResult(probability=0.5, decision="block", model_version="x" * 64)
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.probability = 0.6  # type: ignore[misc]


# ---------------------------------------------------------------------
# TestLoadArtefacts — module-private helper.
# ---------------------------------------------------------------------


class TestLoadArtefacts:
    """`_load_artefacts` returns a frozen `_Artefacts` bundle."""

    def test_returns_frozen_bundle(self) -> None:
        _require_artefacts()
        bundle = _load_artefacts(_MODEL_DIR)
        assert isinstance(bundle, _Artefacts)
        assert isinstance(bundle.model, LightGBMFraudModel)
        # Calibrator is a Union; verify it has the transform method.
        assert hasattr(bundle.calibrator, "transform")
        assert isinstance(bundle.content_hash, str)
        assert len(bundle.content_hash) == 64
