"""Unit tests for `fraud_engine.features.base` and `fraud_engine.features.pipeline`.

Three contract surfaces:

- `BaseFeatureGenerator`: ABC enforcement (cannot instantiate
  abstract base; subclass missing a method also raises).
- `FeaturePipeline.fit_transform`: chains generators; output carries
  every input column plus every generator's added columns;
  `last_output_dtypes` populated.
- `FeaturePipeline.save / load`: writes joblib + manifest sidecar;
  reload reproduces transform output bit-for-bit.

Two minimal stub generators (`_MeanCenter`, `_LogPlusOne`) are
defined inline so the tests exercise the contract without depending
on Sprint 2's real generators (which don't exist yet).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Self

import joblib
import numpy as np
import pandas as pd
import pytest

from fraud_engine.features import BaseFeatureGenerator, FeaturePipeline


class _MeanCenter(BaseFeatureGenerator):
    """Stub generator: subtract the column mean. Stateful (learns mean)."""

    def __init__(self, col: str) -> None:
        self.col = col
        self._mean: float | None = None

    def fit(self, df: pd.DataFrame) -> Self:
        self._mean = float(df[self.col].mean())
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._mean is None:
            raise AttributeError("_MeanCenter must be fit before transform")
        out = df.copy()
        out[f"feat_{self.col}_centred"] = df[self.col] - self._mean
        return out

    def get_feature_names(self) -> list[str]:
        return [f"feat_{self.col}_centred"]

    def get_business_rationale(self) -> str:
        return f"Mean-centred {self.col}; stub generator for unit tests."


class _LogPlusOne(BaseFeatureGenerator):
    """Stub generator: log1p of a column. Stateless (no fit state)."""

    def __init__(self, col: str) -> None:
        self.col = col

    def fit(self, df: pd.DataFrame) -> Self:
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[f"feat_{self.col}_log"] = np.log1p(df[self.col].to_numpy())
        return out

    def get_feature_names(self) -> list[str]:
        return [f"feat_{self.col}_log"]

    def get_business_rationale(self) -> str:
        return f"log1p({self.col}); stateless stub for unit tests."


# ---------------- Base contract ---------------- #


class TestBaseFeatureGeneratorContract:
    """ABC enforcement on the abstract base."""

    def test_cannot_instantiate_abstract_base(self) -> None:
        """`BaseFeatureGenerator()` raises TypeError per ABC contract."""
        with pytest.raises(TypeError, match="abstract"):
            BaseFeatureGenerator()  # type: ignore[abstract]

    def test_subclass_missing_method_cannot_instantiate(self) -> None:
        """A subclass that forgets one of the abstract methods raises."""

        class _Incomplete(BaseFeatureGenerator):
            def fit(self, df: pd.DataFrame) -> Self:
                return self

            def transform(self, df: pd.DataFrame) -> pd.DataFrame:
                return df

            # forgets get_feature_names + get_business_rationale

        with pytest.raises(TypeError, match="abstract"):
            _Incomplete()  # type: ignore[abstract]

    def test_fit_transform_default_composition(self) -> None:
        """A complete subclass's `fit_transform` returns `fit(df).transform(df)`."""
        gen = _MeanCenter("x")
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        out = gen.fit_transform(df)
        assert "feat_x_centred" in out.columns
        # Mean of [1, 2, 3] is 2; centred values are [-1, 0, 1].
        np.testing.assert_allclose(
            out["feat_x_centred"].to_numpy(),
            np.array([-1.0, 0.0, 1.0]),
        )


# ---------------- Pipeline fit_transform ---------------- #


class TestFeaturePipelineFitTransform:
    """Pipeline's sequential composition + dtype tracking."""

    def test_pipeline_fit_transform_chains_generators(self) -> None:
        """Two generators in sequence both add their columns; originals survive."""
        pipe = FeaturePipeline(generators=[_MeanCenter("x"), _LogPlusOne("y")])
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
        out = pipe.fit_transform(df)
        assert "feat_x_centred" in out.columns
        assert "feat_y_log" in out.columns
        # Original columns preserved through the chain.
        assert "x" in out.columns
        assert "y" in out.columns

    def test_pipeline_records_last_output_dtypes(self) -> None:
        """`last_output_dtypes` populated after `fit_transform`."""
        pipe = FeaturePipeline(generators=[_MeanCenter("x")])
        assert pipe.last_output_dtypes is None
        pipe.fit_transform(pd.DataFrame({"x": [1.0, 2.0]}))
        assert pipe.last_output_dtypes is not None
        assert "feat_x_centred" in pipe.last_output_dtypes


# ---------------- Pipeline save / load ---------------- #


class TestFeaturePipelineSaveLoad:
    """Pipeline persistence: joblib + manifest sidecar."""

    def test_save_writes_pipeline_and_manifest(self, tmp_path: Path) -> None:
        """`save()` writes both files; manifest has the expected shape."""
        pipe = FeaturePipeline(generators=[_MeanCenter("x")])
        pipe.fit_transform(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))
        pipeline_path, manifest_path = pipe.save(tmp_path)
        assert pipeline_path.is_file()
        assert manifest_path.is_file()

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["schema_version"] == 1
        assert manifest["n_generators"] == 1
        assert manifest["n_features"] == 1
        assert manifest["features"][0]["name"] == "feat_x_centred"
        assert manifest["features"][0]["generator"] == "_MeanCenter"
        assert manifest["features"][0]["dtype"] == "float64"
        assert "Mean-centred" in manifest["features"][0]["rationale"]

    def test_load_round_trip_produces_identical_output(self, tmp_path: Path) -> None:
        """save → load → transform reproduces the original output bit-for-bit."""
        pipe = FeaturePipeline(generators=[_MeanCenter("x"), _LogPlusOne("x")])
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
        original_out = pipe.fit_transform(df)
        pipe.save(tmp_path)
        reloaded = FeaturePipeline.load(tmp_path)
        reloaded_out = reloaded.transform(df)
        pd.testing.assert_frame_equal(reloaded_out, original_out)

    def test_load_rejects_wrong_object_type(self, tmp_path: Path) -> None:
        """`load` raises `TypeError` if the joblib payload isn't a FeaturePipeline."""
        joblib.dump({"not": "a pipeline"}, tmp_path / "pipeline.joblib")
        with pytest.raises(TypeError, match="expected FeaturePipeline"):
            FeaturePipeline.load(tmp_path)
