"""Abstract base for every feature generator from Sprint 2 onwards.

`BaseFeatureGenerator` is the contract every feature generator (T1
basic, T2 aggregations, T3 behavioural, T4 EWM, T5 graph) inherits.
It is deliberately minimal — `fit` / `transform` / `fit_transform`
plus two introspection methods (`get_feature_names` /
`get_business_rationale`) — so a Sprint 5 serving layer can read a
fitted `FeaturePipeline` and render a per-feature audit trail without
plumbing extra metadata through every subclass.

Business rationale:
    Sprint 5's online API needs to answer "which generator
    contributed this column, and what business rule does it encode?"
    for any feature in the model's input vector. Sprint 4's
    economic-cost evaluation reads the same audit trail to attribute
    the cost-curve lift to the generators that drove it. Without a
    uniform contract every consumer would invent its own
    introspection layer; we'd carry the technical debt for the rest
    of the project.

Trade-offs considered:
    - `ABC` over `typing.Protocol`. Protocols give static
      duck-typing — useful, but the failure mode is silent: a future
      generator that forgets `get_business_rationale` would type-check
      but raise at first call site. Runtime ABC enforcement raises at
      *instantiation*, before any pipeline-level damage.
    - `Self` (PEP 673, Python 3.11+) as `fit`'s return type rather
      than `BaseFeatureGenerator`. `Self` preserves the concrete
      subclass type so chained calls (`MyGen().fit(df).transform(df2)`)
      mypy-narrow correctly to `MyGen.transform`, not the abstract
      method. The project pins Python ≥ 3.11 so no backport needed.
    - `BaseEstimator` (from sklearn) was rejected. Feature pipeline
      state isn't always sklearn-API-shaped (entity-keyed dicts,
      EWM warm starts, graph adjacency tables); inheriting
      `BaseEstimator` would force a `get_params` / `set_params`
      surface that adds boilerplate to every subclass for zero gain.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

import pandas as pd


class BaseFeatureGenerator(ABC):
    """Abstract base for every feature generator in Sprint 2 onwards.

    Contract:
        - `fit(df)` learns any stateful params (encoder vocabs,
          per-entity statistics, EWM warm starts) and returns
          `self` so `MyGen().fit(df).transform(df)` chains cleanly.
        - `transform(df)` is a pure function of fitted state — no
          side effects; idempotent under repeated calls. Subclasses
          that depend on fitted state must raise loudly when
          called pre-fit (typically `AttributeError` on the
          missing attribute).
        - `fit_transform(df)` is the composition; default
          implementation returns ``self.fit(df).transform(df)``.
          Subclasses override only when fit and transform can share
          intermediate state for performance (e.g., sklearn's
          `OneHotEncoder.fit_transform` avoids a re-pass).
        - Output of `transform` MUST contain every input column
          plus the new feature columns. Downstream pipelines depend
          on the input columns surviving each stage; a generator
          that drops columns silently breaks composition.
        - Subclasses MUST declare `get_feature_names()` and
          `get_business_rationale()` so the pipeline's
          `feature_manifest.json` renders a human-readable audit
          trail.

    Attributes:
        (None at this layer — concrete subclasses define their own
        fitted state.)
    """

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> Self:
        """Learn stateful parameters from `df` and return self.

        Args:
            df: Training-fold frame. Subclasses define the required
                columns; the base contract only requires it to be a
                `pd.DataFrame`.

        Returns:
            ``self``, fitted in place.
        """

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the transformation; return a new frame with extra columns.

        Args:
            df: Frame to transform. Must satisfy whatever schema
                the concrete subclass declares.

        Returns:
            A new `pd.DataFrame` containing every input column plus
            the columns named in `get_feature_names()`.
        """

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Default composition: ``self.fit(df).transform(df)``.

        Subclasses override only for performance wins where
        intermediate state can be shared between fit and transform.

        Args:
            df: Frame to fit and transform in one pass.

        Returns:
            The transformed frame — same as ``self.fit(df).transform(df)``.
        """
        return self.fit(df).transform(df)

    @abstractmethod
    def get_feature_names(self) -> list[str]:
        """Return the names of the columns this generator adds.

        The list MUST be deterministic — calling twice returns the
        same names — so the pipeline manifest is reproducible.

        Returns:
            Ordered list of column names added by `transform`.
        """

    @abstractmethod
    def get_business_rationale(self) -> str:
        """Return a one-paragraph business rationale for these features.

        Rendered into `feature_manifest.json` and surfaced by Sprint
        5's serving layer when explaining a prediction. Keep it
        terse (1–3 sentences); the deeper trade-offs belong in the
        concrete subclass's module docstring.

        Returns:
            Human-readable rationale string.
        """


__all__ = ["BaseFeatureGenerator"]
