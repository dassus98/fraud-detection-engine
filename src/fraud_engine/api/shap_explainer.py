"""SHAP-based interpretability surface for the fraud-scoring API.

Sprint 5 prompt 5.1.e: precomputed `shap.TreeExplainer` wrapping the
production LightGBM booster + a hand-curated `configs/reason_codes.yaml`
mapping feature names to user-facing text. Together these populate
`PredictionResponse.top_reasons` for the route handler in Sprint 5.1.f.

Public surface:
    - `ShapExplainer.top_k_contributions(features, k=3)` returns the
      top-k features by `abs(shap_value)`, each as a `Contribution`
      tuple `(feature_name, shap_value, direction)`.
    - `ShapExplainer.map_to_reasons(contributions)` translates those
      contributions into human-readable strings via the YAML.
    - `ShapExplainer.reload(model)` atomically swaps the underlying
      explainer when the model is reloaded mid-session.

Business rationale:
    A production fraud system that blocks a customer's transaction
    must be able to explain why. "The model said 0.94" is not an
    explanation; "high amount + new device + unusual time of day" is.
    SHAP TreeExplainer gives us per-feature contributions to the
    model's log-odds, additive by construction so the contributions
    sum to the prediction. The YAML translates the technical feature
    names (`card1_v_ewm_lambda_0.05`) into business-meaningful text
    ("sustained unusual activity on this card over the last several
    days") so the API response is readable by a CS agent or end user
    without dropping into the engineering layer.

Trade-offs considered:
    - **`Contribution(NamedTuple)` over raw tuple, frozen dataclass,
      or Pydantic Reason.** The spec says "list of (feature_name,
      shap_value, direction)" — NamedTuple keeps that shape
      (`isinstance(c, tuple)` is True; positional unpacking works)
      while adding attribute access (`c.feature_name`) that makes the
      consumer code readable. Pydantic `Reason` (5.1.a) would couple
      the explainer to the API schema; the route handler (5.1.f)
      converts at the boundary instead.
    - **Sort by `abs(shap_value)` descending; drop zero entries.** A
      top-3 list should surface the strongest drivers regardless of
      direction — a +0.4 alongside a -0.3 is more informative than
      two +0.4s. Zero contributions are noise (the booster never
      split on that feature for this row); including them would
      generate "transaction blocked because [V137 contributed
      nothing]" which is meaningless.
    - **Silent drop on unmapped features in `map_to_reasons`.** The
      YAML covers ~24 of 743 features by design (Vesta-anonymised
      V/C/D/M and `is_null_*` indicators are deliberately excluded
      because there's no honest reason text we can publish). Raising
      on every unmapped feature would force every test fixture to
      populate the YAML with the test's specific feature set; silent
      drop matches the production semantic (we couldn't explain this
      feature; show the others).
    - **`shap.TreeExplainer` constructed at startup, not per-request.**
      Construction is ~50–200ms for a 743-feature LightGBM; per-call
      `shap_values(X)` is ~30–50ms. Caching the explainer cuts the
      per-request budget to just the per-call cost. `reload(model)`
      is the only path that re-runs the constructor — same atomic-
      swap semantic as Sprint 5.1.d's `InferenceService.reload`.
    - **SHAP runs against the raw booster, NOT the calibrator.**
      TreeExplainer is a tree-model technique; the isotonic
      calibrator (Sprint 3.3.c) is not a tree, so it can't be
      explained the same way. Contributions are in **log-odds
      space** (the model's raw output), which is the natural
      additive scale where the sum-check invariant holds. The
      route handler (5.1.f) knows the calibrated probability for
      the API surface; the per-feature contributions are about the
      raw model's reasoning. This is documented inline at the
      `top_k_contributions` call site too.
    - **`expected_value` defensively coerced to scalar float.** SHAP
      0.46.0 returns `expected_value` as a numpy `ndarray` of shape
      `(1,)` for LightGBM binary classifiers (verified empirically).
      `np.asarray(ev).reshape(-1)[0]` extracts the scalar regardless
      of the underlying shape; future SHAP versions changing the
      shape don't break the contract.

Module surface (re-exported from `fraud_engine.api`):
    - ShapExplainer
    - Contribution

Cross-references:
    - `src/fraud_engine/models/lightgbm_model.py` — `booster_` access
      + `predict(X, raw_score=True)` for the sum-check invariant.
    - `src/fraud_engine/api/inference.py` (Sprint 5.1.d) — atomic-
      reload pattern this class mirrors.
    - `src/fraud_engine/api/schemas.py` (Sprint 5.1.a) —
      `ReasonDirectionLiteral` + the `Reason` Pydantic model that
      Sprint 5.1.f's route handler will populate from `Contribution`.
    - `configs/reason_codes.yaml` — the runtime-consumed YAML.
    - `models/sprint3/lightgbm_model_manifest.json:feature_names` —
      canonical 743-column list; cross-referenced by the `_validate_*`
      helpers and tests.
    - `CLAUDE.md` §3 (latency budget), §5.5 (logging discipline).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final, Literal, NamedTuple, cast

import numpy as np
import pandas as pd
import shap
import yaml

from fraud_engine.api.schemas import ReasonDirectionLiteral
from fraud_engine.models.lightgbm_model import LightGBMFraudModel
from fraud_engine.utils.logging import get_logger, log_call

# ---------------------------------------------------------------------
# Module constants.
# ---------------------------------------------------------------------

_DEFAULT_MODEL_DIR: Final[Path] = Path("models/sprint3")
_DEFAULT_REASON_CODES_FILENAME: Final[str] = "reason_codes.yaml"

# Default top-k for `top_k_contributions`. Per Sprint 5.1.a's
# `PredictionResponse.top_reasons.max_length = 10`, the route handler
# can request up to 10; 3 is the sensible default for a quick block
# explanation.
_DEFAULT_K: Final[int] = 3

# Direction literals, mirrored from `schemas.ReasonDirectionLiteral`.
_INCREASES_RISK: Final[ReasonDirectionLiteral] = "increases_risk"
_DECREASES_RISK: Final[ReasonDirectionLiteral] = "decreases_risk"

# YAML schema: each value must carry both `high` and `low` keys (one
# or both may be `None`). The validator enforces this invariant so
# `entry.get("high")` cannot silently miss a typo'd key.
_REQUIRED_DIRECTION_KEYS: Final[tuple[Literal["high"], Literal["low"]]] = ("high", "low")

# Number of classes in a binary classifier — used in older-SHAP-API
# defensive coercion. Pinned as a constant to keep ruff PLR2004 quiet
# and to document the assumption ("binary classifier; positive class
# is index 1").
_BINARY_CLASSES: Final[int] = 2

# Expected ndim of a `(n_rows, n_features)` shap_values array.
_NDIM_2D: Final[int] = 2

_logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Contribution NamedTuple.
# ---------------------------------------------------------------------


class Contribution(NamedTuple):
    """One feature's SHAP contribution to a single prediction.

    Attributes:
        feature_name: The model's column name (e.g.
            `card1_fraud_v_ewm_lambda_0.05`). Matches an entry in
            the model's `feature_names_` list.
        shap_value: The SHAP contribution in **log-odds** space. The
            sign indicates direction (positive → increases the
            predicted fraud probability); the magnitude indicates
            relative importance for ranking.
        direction: Plain-English sign of `shap_value`. Mirrors
            Sprint 5.1.a's `ReasonDirectionLiteral`.
    """

    feature_name: str
    shap_value: float
    direction: ReasonDirectionLiteral


# ---------------------------------------------------------------------
# Module-private helpers.
# ---------------------------------------------------------------------


def _load_reason_codes(path: Path) -> dict[str, dict[str, str | None]]:
    """Read and validate `configs/reason_codes.yaml`.

    Args:
        path: Absolute path to the YAML file.

    Returns:
        Mapping from feature name to `{"high": str | None, "low":
        str | None}`. Both keys always present; values may be None.

    Raises:
        FileNotFoundError: If `path` does not exist.
        TypeError: If the YAML root is not a mapping.
        ValueError: If any entry is malformed (not a mapping, missing
            required keys, non-string non-None text value).
    """
    if not path.exists():
        raise FileNotFoundError(f"ShapExplainer: reason_codes YAML not found at {path}")
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    _validate_reason_codes_shape(loaded, path)
    # `cast` for mypy — `_validate_reason_codes_shape` enforces the shape.
    return cast(dict[str, dict[str, str | None]], loaded)


def _validate_reason_codes_shape(loaded: object, path: Path) -> None:
    """Enforce the YAML's structural invariants.

    Required: top-level mapping; every value is a mapping with both
    `high` and `low` keys; each text value is `None` or a non-empty
    string.

    Raises:
        TypeError: If the YAML root is not a mapping.
        ValueError: If any entry is malformed.
    """
    if not isinstance(loaded, dict):
        raise TypeError(
            f"reason_codes YAML at {path}: top-level must be a mapping, "
            f"got {type(loaded).__name__}"
        )
    for feature_name, entry in loaded.items():
        if not isinstance(entry, dict):
            raise ValueError(
                f"reason_codes YAML at {path}: entry for {feature_name!r} "
                f"must be a mapping, got {type(entry).__name__}"
            )
        for direction_key in _REQUIRED_DIRECTION_KEYS:
            if direction_key not in entry:
                raise ValueError(
                    f"reason_codes YAML at {path}: entry for "
                    f"{feature_name!r} missing required key {direction_key!r}"
                )
            value = entry[direction_key]
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(
                    f"reason_codes YAML at {path}: entry for "
                    f"{feature_name!r}.{direction_key} must be a non-empty "
                    f"string or null, got {value!r}"
                )


def _resolve_default_reason_codes_path() -> Path:
    """Resolve `configs/reason_codes.yaml` relative to the repo root.

    Mirrors the pattern in
    `features.tier4_decay._resolve_config_path` and
    `api.redis_store._resolve_config_path` — repo root is four
    `parents[]` up from this file.
    """
    project_root = Path(__file__).resolve().parents[3]
    return project_root / "configs" / _DEFAULT_REASON_CODES_FILENAME


def _expected_value_to_scalar(expected_value: object) -> float:
    """Coerce `shap.TreeExplainer.expected_value` to a scalar float.

    SHAP 0.46.0 returns this as a numpy ndarray of shape `(1,)` for
    LightGBM binary classifiers (verified empirically); future
    versions may return a scalar or list. `np.asarray(...).reshape(-1)[0]`
    handles all three shapes uniformly.
    """
    return float(np.asarray(expected_value).reshape(-1)[0])


def _shap_values_to_row_array(
    shap_values: object,
    n_features: int,
) -> np.ndarray[Any, Any]:
    """Coerce `shap.TreeExplainer.shap_values(X)` output for a single row.

    SHAP versions vary:
    - 0.46.0 binary classifier: returns `np.ndarray` of shape
      `(n_rows, n_features)` (single positive-class array).
    - Older versions: returned a 2-element list `[neg_class_arr,
      pos_class_arr]`; we want the positive-class array (index 1).

    This helper handles both shapes and returns the 1-D `(n_features,)`
    array for the first row.

    Raises:
        ValueError: If the shape is unrecognised.
    """
    arr: np.ndarray[Any, Any]
    if isinstance(shap_values, list):
        # Older SHAP: list per class; positive class is index 1 for binary.
        if len(shap_values) != _BINARY_CLASSES:
            raise ValueError(
                f"ShapExplainer: shap_values list has {len(shap_values)} "
                f"elements; expected {_BINARY_CLASSES} (binary classifier)"
            )
        arr = np.asarray(shap_values[1])
    else:
        arr = np.asarray(shap_values)

    if arr.ndim == _NDIM_2D:
        if arr.shape[1] != n_features:
            raise ValueError(
                f"ShapExplainer: shap_values 2nd dim {arr.shape[1]} != " f"n_features {n_features}"
            )
        return cast(np.ndarray[Any, Any], arr[0])
    if arr.ndim == 1 and arr.shape[0] == n_features:
        return arr
    raise ValueError(
        f"ShapExplainer: shap_values shape {arr.shape} is not " f"(n_features,) or (1, n_features)"
    )


# ---------------------------------------------------------------------
# Public class.
# ---------------------------------------------------------------------


class ShapExplainer:
    """Precomputed SHAP TreeExplainer + reason-code mapping.

    Public API:
        - `top_k_contributions(features, k=3) -> list[Contribution]`:
          ranked by `abs(shap_value)` descending; zero contributions
          dropped.
        - `map_to_reasons(contributions) -> list[str]`: translates
          contributions to user-facing strings via `reason_codes.yaml`;
          unmapped features and `null` directions silently dropped.
        - `reload(model)`: atomic explainer-swap when the model is
          re-loaded (Sprint 5.1.d `InferenceService.reload`-equivalent
          for the explainer surface).

    Lifecycle:
        - `__init__` constructs the TreeExplainer and loads the YAML.
          Both are cached for the process lifetime; re-construction
          only happens on `reload()`.
        - All public methods are synchronous + thread-safe under the
          GIL-atomic-swap pattern (mirrors `InferenceService`).

    Business rationale + trade-offs considered: see module docstring.
    """

    def __init__(
        self,
        model: LightGBMFraudModel | None = None,
        reason_codes_path: Path | None = None,
    ) -> None:
        """Construct the explainer and load reason codes.

        Args:
            model: Inject a pre-loaded `LightGBMFraudModel`. None →
                `LightGBMFraudModel.load(models/sprint3)`.
            reason_codes_path: Override the YAML path. None →
                `configs/reason_codes.yaml`.

        Raises:
            FileNotFoundError: If model artefacts or reason_codes
                YAML are missing.
            ValueError / TypeError: If the YAML is malformed.
            RuntimeError: If the loaded model has no fitted booster.
        """
        if model is None:
            model = LightGBMFraudModel.load(_DEFAULT_MODEL_DIR)
        if model.booster_ is None:
            raise RuntimeError(
                "ShapExplainer: model has no fitted booster_; "
                "call `LightGBMFraudModel.fit(...)` first."
            )
        if model.feature_names_ is None:
            raise RuntimeError(
                "ShapExplainer: model has no fitted feature_names_; "
                "the model artefact appears unfitted."
            )

        path = (
            reason_codes_path
            if reason_codes_path is not None
            else _resolve_default_reason_codes_path()
        )

        self._model: LightGBMFraudModel = model
        self._explainer = shap.TreeExplainer(model.booster_)
        self._feature_names: list[str] = list(model.feature_names_)
        self._reason_codes: dict[str, dict[str, str | None]] = _load_reason_codes(path)

    # ---------- public API ---------------------------------------------

    @log_call
    def top_k_contributions(
        self,
        features: pd.DataFrame,
        k: int = _DEFAULT_K,
    ) -> list[Contribution]:
        """Return the top-k SHAP contributions for a single transaction.

        Args:
            features: Single-row DataFrame matching the model's
                `feature_names_in_` (typically the `df` field of a
                `FeatureVector` from Sprint 5.1.c).
            k: Maximum number of contributions to return. Default 3.

        Returns:
            List of `Contribution(feature_name, shap_value, direction)`
            tuples, sorted by `abs(shap_value)` descending. Length is
            `min(k, n_nonzero_shap_values)`; zero contributions are
            dropped.

        Raises:
            ValueError: If `k < 0`.
        """
        if k < 0:
            raise ValueError(f"ShapExplainer: k must be >= 0, got {k}")

        # Bind locals once. A concurrent reload only affects the next call.
        local_explainer = self._explainer
        local_feature_names = self._feature_names

        # SHAP returns log-odds-space contributions for the positive class.
        # See SHAP's TreeExplainer docs: `model_output="raw"` (default) is
        # what we want for LightGBM binary classifiers.
        raw_shap = local_explainer.shap_values(features)
        row = _shap_values_to_row_array(raw_shap, n_features=len(local_feature_names))

        # Build (name, value) pairs, dropping zero entries.
        pairs: list[tuple[str, float]] = [
            (name, float(value))
            for name, value in zip(local_feature_names, row, strict=True)
            if value != 0.0
        ]
        # Sort by |shap_value| descending — strongest drivers first.
        pairs.sort(key=lambda p: abs(p[1]), reverse=True)
        top = pairs[:k]

        return [
            Contribution(
                feature_name=name,
                shap_value=value,
                direction=_INCREASES_RISK if value > 0 else _DECREASES_RISK,
            )
            for name, value in top
        ]

    def map_to_reasons(self, contributions: list[Contribution]) -> list[str]:
        """Translate SHAP contributions to human-readable reason strings.

        Each `Contribution` is looked up in the YAML; the matching
        direction's text is appended to the output. Contributions
        whose feature name is missing from the YAML, or whose
        direction has `null` text, are silently dropped — the YAML
        covers ~24 of 743 features by design.

        Args:
            contributions: Output of `top_k_contributions`.

        Returns:
            List of user-facing strings in the same order as the
            input contributions. Length is ≤ `len(contributions)`.
        """
        # Bind once for thread-safety under reload.
        local_codes = self._reason_codes
        out: list[str] = []
        for c in contributions:
            entry = local_codes.get(c.feature_name)
            if entry is None:
                continue  # feature not in YAML
            text = entry.get("high" if c.direction == _INCREASES_RISK else "low")
            if text is None:
                continue  # YAML covers feature but no text for this direction
            out.append(text)
        return out

    @log_call
    def reload(self, model: LightGBMFraudModel) -> None:
        """Atomically replace the explainer + feature names with a new model.

        Mirrors `InferenceService.reload` (Sprint 5.1.d) — single-
        attribute rebind is GIL-atomic, so concurrent `top_k_contributions`
        calls binding `local_explainer = self._explainer` at the top of
        the method see either the old or new explainer, never a torn
        view.

        Args:
            model: A fitted `LightGBMFraudModel` with a populated
                `booster_` and `feature_names_`.

        Raises:
            RuntimeError: If `model` lacks fitted state.
        """
        if model.booster_ is None or model.feature_names_ is None:
            raise RuntimeError(
                "ShapExplainer.reload: model lacks fitted booster_ or "
                "feature_names_; cannot reload from an unfitted model."
            )
        new_explainer = shap.TreeExplainer(model.booster_)
        new_feature_names = list(model.feature_names_)
        # GIL-atomic swap: each rebind is one assignment.
        self._explainer = new_explainer
        self._feature_names = new_feature_names
        self._model = model

    # ---------- read-only accessors -------------------------------------

    @property
    def expected_value(self) -> float:
        """The base log-odds for the explainer's expected prediction.

        Sum-check invariant:
            `expected_value + sum(shap_values_for_one_row) ≈
             booster.predict(X, raw_score=True)[0]` to within 1e-5.
        """
        return _expected_value_to_scalar(self._explainer.expected_value)

    @property
    def feature_names(self) -> list[str]:
        """The model's feature-name list (read-only copy)."""
        return list(self._feature_names)

    @property
    def reason_codes(self) -> dict[str, dict[str, str | None]]:
        """The loaded YAML mapping (read-only copy)."""
        return {k: dict(v) for k, v in self._reason_codes.items()}


__all__ = ["Contribution", "ShapExplainer"]
