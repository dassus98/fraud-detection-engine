"""Sequential composition of `BaseFeatureGenerator` instances.

`FeaturePipeline` runs each generator in order, threading the running
DataFrame from one stage's output into the next stage's input. The
fitted pipeline persists as a single joblib artefact alongside a
`feature_manifest.json` sidecar that lists every feature × generator
× rationale × output dtype.

Business rationale:
    A composed pipeline is **one** fitted artefact, **one** manifest,
    and **one** save / load surface. The alternative — saving each
    generator individually and reconstructing the chain at load time
    — ladders into 5–10 separate joblib files and a hand-rolled
    composition step that drifts every time a generator is added.
    Sprint 5's serving layer reads `feature_manifest.json` to render
    SHAP explanations alongside the per-feature rationale; the
    manifest is the single source of truth for "which generator
    produced this column and why".

Trade-offs considered:
    - **Sequential composition** rather than DAG-style. IEEE-CIS
      Sprint 2 features are linear (basic → aggregations →
      behavioural → EWM → graph); a DAG would let two T2
      aggregations run in parallel but the upstream → downstream
      flow is strictly ordered. DAG bookkeeping (topological sort,
      level-by-level execution) adds complexity that isn't yet
      justified.
    - **One lineage record per pipeline call**, not per generator.
      CLAUDE.md §7.2 mandates `@lineage_step` on every
      transformation; with 5+ generators a per-generator decoration
      would emit 5+ lineage records per fit. Logging at the
      pipeline level keeps the JSONL trail readable; per-generator
      visibility is recoverable from `@log_call`'s
      `feature_pipeline.start` / `.done` events. If Sprint 5 needs
      per-generator lineage for compliance, lift `@lineage_step`
      onto each subclass's `transform` at that point.
    - **Save writes a directory, not a single file.** `joblib.dump`
      could in principle inline the manifest into the same blob,
      but a separate `feature_manifest.json` is `cat`-able and
      `jq`-queryable without a Python interpreter — useful for
      ops review and Sprint 6 monitoring dashboards.
    - **`load` is a `@classmethod`** rather than a free function,
      to mirror `LineageLog.read` (also a classmethod after the
      Sprint 1 audit). Stateless reader; no instance needed.

Version history:
    1 — initial. `FeaturePipeline` dataclass; `pipeline.joblib` +
        `feature_manifest.json` sidecar layout; `_FEATURE_MANIFEST_SCHEMA_VERSION = 1`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import joblib
import pandas as pd

from fraud_engine.data.lineage import lineage_step
from fraud_engine.features.base import BaseFeatureGenerator
from fraud_engine.utils.logging import log_call

# Filename for the pickled pipeline. Pinned constant so a future
# rename surfaces in exactly one place.
_PIPELINE_FILENAME: Final[str] = "pipeline.joblib"

# Filename for the human-readable feature manifest sidecar. Sprint 5's
# serving layer and Sprint 6's monitoring read this name verbatim.
_MANIFEST_FILENAME: Final[str] = "feature_manifest.json"

# Version of the manifest's JSON shape. Bump when the schema changes
# in a non-backward-compatible way; `verify_lineage.py`-style audit
# scripts may pin this version literal as a regression detector.
_FEATURE_MANIFEST_SCHEMA_VERSION: Final[int] = 1

# Sentinel dtype string used in the manifest when `save()` is called
# before any `fit_transform` / `transform` populates `last_output_dtypes`.
_UNKNOWN_DTYPE: Final[str] = "unknown"


@dataclass
class FeaturePipeline:
    """Sequential composition of `BaseFeatureGenerator` instances.

    Business rationale:
        See module docstring. Short version: one pipeline ≡ one
        fitted artefact ≡ one manifest, so Sprint 5's serving layer
        and Sprint 6's monitoring both read a single source of truth.

    Trade-offs considered:
        - Pipeline-level `@lineage_step` only (one record per call,
          not per generator). See module docstring for the
          recoverability path via `@log_call` events.
        - `last_output_dtypes` is a mutable instance attribute,
          updated by every `fit_transform` / `transform` call. The
          alternative — recompute on `save()` by re-running the
          pipeline — would be slower and could re-introduce
          dtype-drift bugs the cleaner audit fixed.
        - `save()` returns a tuple (pipeline_path, manifest_path)
          for chaining and logging; callers that don't care can
          ignore the return.

    Attributes:
        generators: Ordered list of `BaseFeatureGenerator` instances.
            Each receives the running DataFrame from the previous
            stage and returns its own augmented copy.
        last_output_dtypes: Map of column name → pandas dtype string
            recorded after the most recent `fit_transform` /
            `transform`. `None` before the first call. `save()` reads
            this to populate manifest dtype info.
    """

    generators: list[BaseFeatureGenerator]
    last_output_dtypes: dict[str, str] | None = field(default=None)

    @log_call
    @lineage_step("feature_pipeline")
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit each generator on the running frame, transform, pass on.

        For each generator in order: call `fit(current).transform(current)`
        and feed the output into the next stage. The first
        generator sees the raw input; subsequent generators see
        progressively augmented frames.

        Args:
            df: Input frame; must satisfy whatever the first
                generator's `fit` requires.

        Returns:
            The final transformed frame, with every original column
            preserved plus every generator's added columns.
        """
        current = df
        for gen in self.generators:
            current = gen.fit(current).transform(current)
        self.last_output_dtypes = {col: str(current[col].dtype) for col in current.columns}
        return current

    @log_call
    @lineage_step("feature_pipeline_transform")
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply every fitted generator in order; return augmented frame.

        Caller is responsible for having called `fit_transform` (or
        fitted the generators externally) first. We do not enforce a
        state-machine "must fit before transform" — sklearn doesn't
        either, and the natural failure mode (missing fitted
        attribute) raises `AttributeError` from inside the generator
        with a clear traceback.

        Args:
            df: Frame to transform.

        Returns:
            The final transformed frame.
        """
        current = df
        for gen in self.generators:
            current = gen.transform(current)
        self.last_output_dtypes = {col: str(current[col].dtype) for col in current.columns}
        return current

    def save(
        self,
        path: Path,
        pipeline_filename: str = _PIPELINE_FILENAME,
        manifest_filename: str = _MANIFEST_FILENAME,
    ) -> tuple[Path, Path]:
        """Persist the pipeline + manifest under `path`.

        Args:
            path: Destination directory. Created if missing.
                Existing files at the resolved names are overwritten
                silently.
            pipeline_filename: Filename for the joblib payload.
                Default `"pipeline.joblib"`. Sprint 2's tier builds
                pass tier-specific names (e.g. `"tier1_pipeline.joblib"`)
                so the same `models/pipelines/` directory can host
                multiple tiers without collision.
            manifest_filename: Filename for the manifest sidecar.
                Default `"feature_manifest.json"`.

        Returns:
            ``(pipeline_path, manifest_path)`` for caller logging.
        """
        path.mkdir(parents=True, exist_ok=True)
        pipeline_path = path / pipeline_filename
        manifest_path = path / manifest_filename
        joblib.dump(self, pipeline_path)
        manifest = self._build_manifest()
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return pipeline_path, manifest_path

    @classmethod
    def load(cls, path: Path, pipeline_filename: str = _PIPELINE_FILENAME) -> FeaturePipeline:
        """Inverse of `save`. Reads `path / pipeline_filename`.

        Args:
            path: Directory containing the saved pipeline. The
                manifest sidecar is not read here — it's an audit
                artefact, not part of the runtime contract.
            pipeline_filename: Filename to load from. Must match
                whatever was passed to `save`. Default
                `"pipeline.joblib"`.

        Returns:
            The reconstructed `FeaturePipeline`.

        Raises:
            TypeError: If the joblib payload is not a
                `FeaturePipeline` instance. Defends against a
                caller pointing `load` at the wrong file.
        """
        pipeline_path = path / pipeline_filename
        loaded = joblib.load(pipeline_path)
        if not isinstance(loaded, cls):
            raise TypeError(
                f"Loaded object at {pipeline_path} is "
                f"{type(loaded).__name__}, expected FeaturePipeline"
            )
        return loaded

    def _build_manifest(self) -> dict[str, Any]:
        """Render a feature × generator × rationale × dtype manifest.

        For each generator, iterate `get_feature_names()` and pair
        each feature with its generator's class name, business
        rationale, and (if available) the dtype recorded by the
        most recent `fit_transform` / `transform` call. Features
        for which no dtype is known render as ``"unknown"`` rather
        than failing — `save()` is a best-effort artefact, not a
        validation gate.

        Returns:
            A JSON-safe dict ready for `json.dumps`.
        """
        features: list[dict[str, Any]] = []
        observed_dtypes = self.last_output_dtypes or {}
        for gen in self.generators:
            generator_name = type(gen).__name__
            rationale = gen.get_business_rationale()
            for feature_name in gen.get_feature_names():
                features.append(
                    {
                        "name": feature_name,
                        "generator": generator_name,
                        "rationale": rationale,
                        "dtype": observed_dtypes.get(feature_name, _UNKNOWN_DTYPE),
                    }
                )
        return {
            "schema_version": _FEATURE_MANIFEST_SCHEMA_VERSION,
            "generators": [type(g).__name__ for g in self.generators],
            "n_generators": len(self.generators),
            "n_features": len(features),
            "features": features,
        }


__all__ = ["FeaturePipeline"]
