"""Unit tests for `fraud_engine.api.shap_explainer.ShapExplainer`.

Sprint 5 prompt 5.1.e verification surface.

Business rationale:
    The ShapExplainer is the load-bearing interpretability primitive
    for the API's `top_reasons` field. A regression here — a
    direction sign flip, a YAML-key drift, a sum-check invariant
    violation — leaks into the production audit trail. The one place
    those contracts are pinned is here.

Trade-offs considered:
    - **Real model + real YAML (skip-if-missing).** Sprint 5.1.d's
      precedent. The Tier-1 pipeline + LightGBM model artefacts live
      in `models/` (gitignored); the YAML is in `configs/` (committed).
      A test fixture that mocks the booster would lose the sum-check
      invariant — that's the load-bearing test, so we use the real
      booster.
    - **Mocked explainer for direction-sign + zero-drop tests.** Need
      to verify mapping from shap-value sign to direction string;
      easiest with hand-constructed shap arrays via mock.
    - **Hand-curated fixtures for `map_to_reasons`.** A handful of
      `Contribution` literals exercise the high/low/null/unmapped
      branches without coupling tests to the real YAML's specific
      content.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import yaml

from fraud_engine.api.shap_explainer import (
    Contribution,
    ShapExplainer,
    _expected_value_to_scalar,
    _load_reason_codes,
    _shap_values_to_row_array,
    _validate_reason_codes_shape,
)
from fraud_engine.models.lightgbm_model import LightGBMFraudModel

# ---------------------------------------------------------------------
# Required artefacts — skip if absent.
# ---------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODEL_DIR = _REPO_ROOT / "models" / "sprint3"
_MODEL_FILE = _MODEL_DIR / "lightgbm_model.joblib"
_MANIFEST_FILE = _MODEL_DIR / "lightgbm_model_manifest.json"
_REASON_CODES_FILE = _REPO_ROOT / "configs" / "reason_codes.yaml"


def _require_artefacts() -> None:
    missing = [p for p in (_MODEL_FILE, _MANIFEST_FILE, _REASON_CODES_FILE) if not p.exists()]
    if missing:
        pytest.skip(f"ShapExplainer artefacts missing: {missing}")


def _build_features() -> pd.DataFrame:
    """Single-row zeros DataFrame matching the model's `feature_names_`."""
    manifest = json.loads(_MANIFEST_FILE.read_text(encoding="utf-8"))
    columns = manifest["feature_names"]
    return pd.DataFrame([[0.0] * len(columns)], columns=columns)


# ---------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def loaded_model() -> Iterator[LightGBMFraudModel]:
    """Load the model once per module — TreeExplainer construction is the cost driver."""
    _require_artefacts()
    yield LightGBMFraudModel.load(_MODEL_DIR)


@pytest.fixture
def explainer(loaded_model: LightGBMFraudModel) -> ShapExplainer:
    """A fully-constructed `ShapExplainer` for the production model + real YAML."""
    return ShapExplainer(model=loaded_model, reason_codes_path=_REASON_CODES_FILE)


@pytest.fixture
def features() -> pd.DataFrame:
    """Single-row zero-filled DataFrame for the canonical model column set."""
    _require_artefacts()
    return _build_features()


# ---------------------------------------------------------------------
# TestExplainerInit — construction + YAML loading.
# ---------------------------------------------------------------------


class TestExplainerInit:
    """Constructor builds TreeExplainer + loads YAML; raises on missing inputs."""

    def test_default_construction(self, loaded_model: LightGBMFraudModel) -> None:
        e = ShapExplainer(model=loaded_model, reason_codes_path=_REASON_CODES_FILE)
        # The explainer is built; expected_value is a finite scalar.
        assert isinstance(e.expected_value, float)
        assert np.isfinite(e.expected_value)
        # Feature names match the model.
        assert e.feature_names == list(loaded_model.feature_names_ or [])
        # YAML loaded a non-empty mapping.
        assert len(e.reason_codes) > 0

    def test_missing_reason_codes_raises(
        self,
        loaded_model: LightGBMFraudModel,
        tmp_path: Path,
    ) -> None:
        bogus = tmp_path / "missing.yaml"
        with pytest.raises(FileNotFoundError, match="reason_codes"):
            ShapExplainer(model=loaded_model, reason_codes_path=bogus)

    def test_malformed_yaml_root_raises(
        self,
        loaded_model: LightGBMFraudModel,
        tmp_path: Path,
    ) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("- list\n- root\n", encoding="utf-8")
        with pytest.raises(TypeError, match="mapping"):
            ShapExplainer(model=loaded_model, reason_codes_path=bad)

    def test_malformed_entry_raises(
        self,
        loaded_model: LightGBMFraudModel,
        tmp_path: Path,
    ) -> None:
        """An entry missing the `low` key must raise."""
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            yaml.safe_dump({"feat_x": {"high": "x"}}),  # missing `low`
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="low"):
            ShapExplainer(model=loaded_model, reason_codes_path=bad)


# ---------------------------------------------------------------------
# TestTopKContributions — ranking, k, direction, zero-drop.
# ---------------------------------------------------------------------


class TestTopKContributions:
    """`top_k_contributions` ranks by abs(shap_value); drops zeros."""

    def test_default_k_is_3(
        self,
        explainer: ShapExplainer,
        features: pd.DataFrame,
    ) -> None:
        out = explainer.top_k_contributions(features)
        assert len(out) <= 3
        assert all(isinstance(c, Contribution) for c in out)

    def test_respects_custom_k(
        self,
        explainer: ShapExplainer,
        features: pd.DataFrame,
    ) -> None:
        out = explainer.top_k_contributions(features, k=5)
        assert len(out) <= 5

    def test_negative_k_raises(
        self,
        explainer: ShapExplainer,
        features: pd.DataFrame,
    ) -> None:
        with pytest.raises(ValueError, match="k"):
            explainer.top_k_contributions(features, k=-1)

    def test_sorted_by_abs_shap_descending(
        self,
        explainer: ShapExplainer,
        features: pd.DataFrame,
    ) -> None:
        """Ranking is by |shap_value| descending."""
        out = explainer.top_k_contributions(features, k=10)
        abs_values = [abs(c.shap_value) for c in out]
        assert abs_values == sorted(abs_values, reverse=True)

    def test_direction_matches_sign(
        self,
        explainer: ShapExplainer,
        features: pd.DataFrame,
    ) -> None:
        """Positive shap → 'increases_risk'; negative → 'decreases_risk'."""
        out = explainer.top_k_contributions(features, k=10)
        for c in out:
            if c.shap_value > 0:
                assert c.direction == "increases_risk"
            elif c.shap_value < 0:
                assert c.direction == "decreases_risk"
            else:
                pytest.fail("Zero-shap entries should have been dropped")

    def test_zero_shap_dropped(self, loaded_model: LightGBMFraudModel) -> None:
        """Mock the explainer to return a row with one zero; verify it's dropped."""
        e = ShapExplainer(model=loaded_model, reason_codes_path=_REASON_CODES_FILE)
        n = len(e.feature_names)
        # Build a synthetic shap array: first entry zero, rest non-zero.
        synthetic = np.zeros((1, n))
        synthetic[0, 1] = 0.5
        synthetic[0, 2] = -0.3
        synthetic[0, 3] = 0.1
        e._explainer = MagicMock()
        e._explainer.shap_values = MagicMock(return_value=synthetic)
        out = e.top_k_contributions(pd.DataFrame([[0.0] * n], columns=e.feature_names), k=10)
        # First feature (index 0) had shap=0 — must be absent.
        feature_names_in_output = {c.feature_name for c in out}
        assert e.feature_names[0] not in feature_names_in_output
        # The three non-zero entries must be present.
        assert e.feature_names[1] in feature_names_in_output
        assert e.feature_names[2] in feature_names_in_output
        assert e.feature_names[3] in feature_names_in_output


# ---------------------------------------------------------------------
# TestMapToReasons — YAML lookup + null/unmapped handling.
# ---------------------------------------------------------------------


class TestMapToReasons:
    """`map_to_reasons` translates contributions; drops null/unmapped silently."""

    def test_high_text_used_for_increases_risk(
        self,
        loaded_model: LightGBMFraudModel,
        tmp_path: Path,
    ) -> None:
        yaml_path = tmp_path / "reasons.yaml"
        yaml_path.write_text(
            yaml.safe_dump({"feat_a": {"high": "HIGH_TEXT", "low": "LOW_TEXT"}}),
            encoding="utf-8",
        )
        e = ShapExplainer(model=loaded_model, reason_codes_path=yaml_path)
        contribs = [Contribution("feat_a", 0.5, "increases_risk")]
        assert e.map_to_reasons(contribs) == ["HIGH_TEXT"]

    def test_low_text_used_for_decreases_risk(
        self,
        loaded_model: LightGBMFraudModel,
        tmp_path: Path,
    ) -> None:
        yaml_path = tmp_path / "reasons.yaml"
        yaml_path.write_text(
            yaml.safe_dump({"feat_a": {"high": "HIGH_TEXT", "low": "LOW_TEXT"}}),
            encoding="utf-8",
        )
        e = ShapExplainer(model=loaded_model, reason_codes_path=yaml_path)
        contribs = [Contribution("feat_a", -0.5, "decreases_risk")]
        assert e.map_to_reasons(contribs) == ["LOW_TEXT"]

    def test_null_text_drops_contribution(
        self,
        loaded_model: LightGBMFraudModel,
        tmp_path: Path,
    ) -> None:
        """`low: null` means drop the contribution when direction='decreases_risk'."""
        yaml_path = tmp_path / "reasons.yaml"
        yaml_path.write_text(
            yaml.safe_dump({"feat_a": {"high": "HIGH_TEXT", "low": None}}),
            encoding="utf-8",
        )
        e = ShapExplainer(model=loaded_model, reason_codes_path=yaml_path)
        contribs = [Contribution("feat_a", -0.5, "decreases_risk")]
        assert e.map_to_reasons(contribs) == []

    def test_unmapped_feature_drops_silently(
        self,
        explainer: ShapExplainer,
    ) -> None:
        contribs = [Contribution("feature_not_in_yaml", 0.5, "increases_risk")]
        assert explainer.map_to_reasons(contribs) == []

    def test_mixed_mapped_unmapped_preserves_order(
        self,
        loaded_model: LightGBMFraudModel,
        tmp_path: Path,
    ) -> None:
        """Mapped contributions appear in input order; unmapped are dropped."""
        yaml_path = tmp_path / "reasons.yaml"
        yaml_path.write_text(
            yaml.safe_dump(
                {
                    "feat_a": {"high": "A_HIGH", "low": None},
                    "feat_b": {"high": "B_HIGH", "low": "B_LOW"},
                }
            ),
            encoding="utf-8",
        )
        e = ShapExplainer(model=loaded_model, reason_codes_path=yaml_path)
        contribs = [
            Contribution("feat_a", 0.9, "increases_risk"),
            Contribution("not_mapped", 0.5, "increases_risk"),
            Contribution("feat_b", -0.4, "decreases_risk"),
        ]
        assert e.map_to_reasons(contribs) == ["A_HIGH", "B_LOW"]

    def test_empty_contributions_returns_empty_list(
        self,
        explainer: ShapExplainer,
    ) -> None:
        assert explainer.map_to_reasons([]) == []


# ---------------------------------------------------------------------
# TestSumCheckInvariant — the load-bearing correctness gate.
# ---------------------------------------------------------------------


class TestSumCheckInvariant:
    """`expected_value + sum(shap[0]) ≈ booster.predict(X, raw_score=True)[0]`."""

    def test_zeros_input_sum_check(
        self,
        loaded_model: LightGBMFraudModel,
        explainer: ShapExplainer,
        features: pd.DataFrame,
    ) -> None:
        """Sum-check holds for the zeros input row."""
        # Compute via the explainer's underlying SHAP.
        raw_shap = explainer._explainer.shap_values(features)
        row = _shap_values_to_row_array(raw_shap, n_features=len(explainer.feature_names))
        base = explainer.expected_value
        # Compute via the booster.
        booster = loaded_model.booster_
        assert booster is not None
        raw_logit = float(
            booster.predict(
                features,
                num_iteration=loaded_model.best_iteration_,
                raw_score=True,
            )[0]
        )
        # The invariant: base + sum(shap) ≈ raw_logit.
        assert np.isclose(base + float(row.sum()), raw_logit, atol=1e-5)

    def test_random_input_sum_check(
        self,
        loaded_model: LightGBMFraudModel,
        explainer: ShapExplainer,
    ) -> None:
        """Sum-check holds for a non-trivial input row."""
        rng = np.random.default_rng(42)
        n = len(explainer.feature_names)
        row_values = rng.normal(0.0, 0.1, size=n).astype(np.float64)
        df = pd.DataFrame([row_values], columns=explainer.feature_names)
        raw_shap = explainer._explainer.shap_values(df)
        row = _shap_values_to_row_array(raw_shap, n_features=n)
        base = explainer.expected_value
        booster = loaded_model.booster_
        assert booster is not None
        raw_logit = float(
            booster.predict(
                df,
                num_iteration=loaded_model.best_iteration_,
                raw_score=True,
            )[0]
        )
        assert np.isclose(base + float(row.sum()), raw_logit, atol=1e-5)


# ---------------------------------------------------------------------
# TestReasonCodesYaml — the actual file's contents.
# ---------------------------------------------------------------------


class TestReasonCodesYaml:
    """The shipped `configs/reason_codes.yaml` validates against the manifest."""

    def test_yaml_loads_cleanly(self) -> None:
        _require_artefacts()
        codes = _load_reason_codes(_REASON_CODES_FILE)
        assert isinstance(codes, dict)
        assert len(codes) >= 20  # spec says 20-30 entries

    def test_every_entry_has_some_text(self) -> None:
        """Every entry must have AT LEAST one of `high`/`low` non-null."""
        _require_artefacts()
        codes = _load_reason_codes(_REASON_CODES_FILE)
        for feature_name, entry in codes.items():
            assert (
                entry["high"] is not None or entry["low"] is not None
            ), f"{feature_name} has both `high` and `low` as null — useless entry"

    def test_all_keys_in_manifest(self) -> None:
        """Every YAML key must be a valid feature name in the model manifest."""
        _require_artefacts()
        codes = _load_reason_codes(_REASON_CODES_FILE)
        manifest = json.loads(_MANIFEST_FILE.read_text(encoding="utf-8"))
        valid = set(manifest["feature_names"])
        for feature_name in codes:
            assert (
                feature_name in valid
            ), f"YAML entry {feature_name!r} not in model's feature_names"

    def test_no_shell_injection_or_html(self) -> None:
        """Reason text contains no `<`, `>`, `${`, or backticks (defensive)."""
        _require_artefacts()
        codes = _load_reason_codes(_REASON_CODES_FILE)
        bad_chars = {"<", ">", "${", "`"}
        for feature_name, entry in codes.items():
            for direction, text in entry.items():
                if text is None:
                    continue
                for ch in bad_chars:
                    assert ch not in text, (
                        f"reason_codes.yaml: {feature_name}.{direction} "
                        f"contains forbidden character {ch!r}: {text!r}"
                    )


# ---------------------------------------------------------------------
# TestReload — atomic explainer swap.
# ---------------------------------------------------------------------


class TestReload:
    """`reload(model)` atomically replaces the explainer + feature names."""

    def test_reload_with_same_model_idempotent(
        self,
        explainer: ShapExplainer,
        loaded_model: LightGBMFraudModel,
        features: pd.DataFrame,
    ) -> None:
        before = explainer.top_k_contributions(features, k=3)
        explainer.reload(loaded_model)
        after = explainer.top_k_contributions(features, k=3)
        # SHAP is deterministic for a given (model, X); same shap values.
        for b, a in zip(before, after, strict=True):
            assert b.feature_name == a.feature_name
            assert np.isclose(b.shap_value, a.shap_value, atol=1e-9)
            assert b.direction == a.direction

    def test_reload_with_unfitted_model_raises(
        self,
        explainer: ShapExplainer,
    ) -> None:
        """An unfitted model raises `RuntimeError`."""
        unfitted = LightGBMFraudModel()
        with pytest.raises(RuntimeError, match="fitted"):
            explainer.reload(unfitted)


# ---------------------------------------------------------------------
# TestEndToEnd — features → top_k → map_to_reasons.
# ---------------------------------------------------------------------


class TestEndToEnd:
    """The full chain: features → contributions → reason strings."""

    def test_full_chain_smoke(
        self,
        explainer: ShapExplainer,
        features: pd.DataFrame,
    ) -> None:
        contribs = explainer.top_k_contributions(features, k=10)
        reasons = explainer.map_to_reasons(contribs)
        # The smoke test: the chain runs without error and returns a list.
        assert isinstance(reasons, list)
        # Every reason is a non-empty string.
        for r in reasons:
            assert isinstance(r, str)
            assert len(r) > 0
        # Length is bounded by the input contributions.
        assert len(reasons) <= len(contribs)


# ---------------------------------------------------------------------
# TestPrivateHelpers — the small helpers (defensive coercion logic).
# ---------------------------------------------------------------------


class TestPrivateHelpers:
    """Defensive helpers that handle SHAP-version variation."""

    def test_expected_value_to_scalar_handles_ndarray(self) -> None:
        assert _expected_value_to_scalar(np.array([0.5])) == 0.5
        assert _expected_value_to_scalar(np.array([1.0, 2.0])[0:1]) == 1.0

    def test_expected_value_to_scalar_handles_python_float(self) -> None:
        assert _expected_value_to_scalar(0.5) == 0.5

    def test_shap_values_to_row_array_handles_2d_ndarray(self) -> None:
        arr = np.array([[1.0, 2.0, 3.0]])
        result = _shap_values_to_row_array(arr, n_features=3)
        assert np.array_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_shap_values_to_row_array_handles_legacy_list(self) -> None:
        """Older SHAP API: list of [neg_class, pos_class] arrays."""
        neg = np.array([[1.0, 2.0, 3.0]])
        pos = np.array([[4.0, 5.0, 6.0]])
        result = _shap_values_to_row_array([neg, pos], n_features=3)
        assert np.array_equal(result, np.array([4.0, 5.0, 6.0]))

    def test_shap_values_bad_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="n_features"):
            _shap_values_to_row_array(np.array([[1.0, 2.0]]), n_features=3)

    def test_validate_reason_codes_shape_rejects_non_dict_root(
        self,
        tmp_path: Path,
    ) -> None:
        with pytest.raises(TypeError):
            _validate_reason_codes_shape(["a", "b"], tmp_path)

    def test_validate_reason_codes_shape_rejects_empty_string(
        self,
        tmp_path: Path,
    ) -> None:
        bad: dict[str, Any] = {"feat_a": {"high": "", "low": None}}
        with pytest.raises(ValueError, match="non-empty"):
            _validate_reason_codes_shape(bad, tmp_path)
