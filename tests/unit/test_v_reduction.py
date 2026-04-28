"""Unit tests for `fraud_engine.features.v_reduction.NanGroupReducer`.

Five contract surfaces:

- `TestNanGroupIdentification`: columns with identical isna() vectors
  group together; columns with different patterns split.
- `TestCorrelationMode`: keeps the most-target-correlated column;
  drops siblings above the correlation threshold; keeps siblings
  below it.
- `TestPCAMode`: replaces a group with its PCA components; output
  columns are named `v_group_{i}_pc_{j}`.
- `TestManifest`: `get_manifest` lists every dropped column with a
  reason and per-group statistics.
- `TestConfigLoad`: default constructor reads `tier3_config.yaml`;
  unsupported method raises.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_engine.features.v_reduction import NanGroupReducer

# ---------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------


def _build_v_frame(n: int = 60, seed: int = 0) -> pd.DataFrame:
    """Synthetic 6-column V frame for the reduction tests.

    - V1, V2, V3 share NaN pattern A (drawn from a single signal);
      V1 is most correlated with isFraud, V2 highly correlated with
      V1, V3 nearly independent of V1.
    - V4, V5 share NaN pattern B (a different mask); V4 highly
      correlated with V5.
    - V6 has its own unique NaN pattern (sole member of its group).
    """
    rng = np.random.default_rng(seed)

    # Target is a binary signal we'll correlate the V columns with.
    target = rng.integers(0, 2, size=n).astype(int)

    # Group A: shared NaN mask (rows 0–9 are NaN for all three).
    mask_a = np.zeros(n, dtype=bool)
    mask_a[:10] = True
    base_a = target.astype(float) + rng.normal(0, 0.1, size=n)
    v1 = base_a.copy()
    v2 = base_a + rng.normal(0, 0.05, size=n)  # ρ(V1, V2) ≈ 0.99
    v3 = rng.normal(0, 1.0, size=n)  # ρ(V1, V3) ≈ 0
    for arr in (v1, v2, v3):
        arr[mask_a] = np.nan

    # Group B: a different shared NaN mask.
    mask_b = np.zeros(n, dtype=bool)
    mask_b[20:25] = True
    base_b = rng.normal(0, 1.0, size=n)
    v4 = base_b.copy()
    v5 = base_b + rng.normal(0, 0.02, size=n)  # ρ(V4, V5) ≈ 1
    for arr in (v4, v5):
        arr[mask_b] = np.nan

    # Group C: singleton with its own NaN mask.
    mask_c = np.zeros(n, dtype=bool)
    mask_c[30:32] = True
    v6 = rng.normal(0, 1.0, size=n)
    v6[mask_c] = np.nan

    return pd.DataFrame(
        {
            "TransactionDT": np.arange(n, dtype=np.int64),
            "isFraud": target,
            "V1": v1,
            "V2": v2,
            "V3": v3,
            "V4": v4,
            "V5": v5,
            "V6": v6,
        }
    )


# ---------------------------------------------------------------------
# NaN-group identification.
# ---------------------------------------------------------------------


class TestNanGroupIdentification:
    """Columns with identical isna() vectors group together."""

    def test_columns_with_same_nan_pattern_grouped(self) -> None:
        df = _build_v_frame()
        reducer = NanGroupReducer(method="correlation", correlation_threshold=0.95)
        reducer.fit(df)
        assert reducer.groups_ is not None
        # V1, V2, V3 share NaN pattern → one group of size 3.
        sizes = sorted(g["size"] for g in reducer.groups_)
        # Three groups: size 3 (V1/V2/V3), size 2 (V4/V5), size 1 (V6).
        assert sizes == [1, 2, 3]

    def test_columns_with_different_nan_patterns_separate(self) -> None:
        df = pd.DataFrame(
            {
                "isFraud": [0, 1, 0, 1, 0, 1],
                "V1": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
                "V2": [1.0, 2.0, 3.0, 4.0, np.nan, 6.0],  # different mask
            }
        )
        reducer = NanGroupReducer(method="correlation", correlation_threshold=0.95)
        reducer.fit(df)
        assert reducer.groups_ is not None
        assert len(reducer.groups_) == 2  # two singleton groups
        for g in reducer.groups_:
            assert g["size"] == 1


# ---------------------------------------------------------------------
# Correlation mode.
# ---------------------------------------------------------------------


class TestCorrelationMode:
    """Greedy keep-by-target-correlation; drop siblings above threshold."""

    def test_keeps_most_target_correlated(self) -> None:
        """V1 is the most-target-correlated of {V1, V2, V3}; it must be kept."""
        df = _build_v_frame()
        reducer = NanGroupReducer(method="correlation", correlation_threshold=0.95)
        reducer.fit(df)
        assert reducer.kept_columns_ is not None
        assert "V1" in reducer.kept_columns_

    def test_drops_correlated_siblings(self) -> None:
        """V2 is highly correlated with V1 → drop. V3 is nearly orthogonal → keep."""
        df = _build_v_frame()
        reducer = NanGroupReducer(method="correlation", correlation_threshold=0.95)
        reducer.fit(df)
        assert reducer.dropped_columns_ is not None
        # V2 should be dropped (|ρ(V1, V2)| > 0.95).
        assert "V2" in reducer.dropped_columns_
        # V3 should be kept (|ρ(V1, V3)| < 0.95).
        assert "V3" in (reducer.kept_columns_ or [])
        # V5 should be dropped (|ρ(V4, V5)| > 0.95).
        assert "V5" in reducer.dropped_columns_

    def test_keeps_singleton_groups(self) -> None:
        """A NaN-group with one column → that column is kept by definition."""
        df = _build_v_frame()
        reducer = NanGroupReducer(method="correlation", correlation_threshold=0.95)
        reducer.fit(df)
        assert "V6" in (reducer.kept_columns_ or [])

    def test_transform_drops_only_dropped_columns(self) -> None:
        """`transform` drops the learned columns and preserves everything else."""
        df = _build_v_frame()
        reducer = NanGroupReducer(method="correlation", correlation_threshold=0.95)
        reducer.fit(df)
        out = reducer.transform(df)
        for col in reducer.dropped_columns_ or []:
            assert col not in out.columns
        for col in reducer.kept_columns_ or []:
            assert col in out.columns
        # Non-V columns are preserved.
        assert "isFraud" in out.columns
        assert "TransactionDT" in out.columns


# ---------------------------------------------------------------------
# PCA mode.
# ---------------------------------------------------------------------


class TestPCAMode:
    """PCA replaces each multi-column group with named components."""

    def test_pca_replaces_group_columns(self) -> None:
        df = _build_v_frame()
        reducer = NanGroupReducer(method="pca", pca_variance_threshold=0.95)
        reducer.fit(df)
        out = reducer.transform(df)

        # Group A (V1, V2, V3) and Group B (V4, V5) replaced; V6 singleton kept.
        assert "V1" not in out.columns
        assert "V2" not in out.columns
        assert "V3" not in out.columns
        assert "V4" not in out.columns
        assert "V5" not in out.columns
        assert "V6" in out.columns

    def test_pca_creates_named_components(self) -> None:
        df = _build_v_frame()
        reducer = NanGroupReducer(method="pca", pca_variance_threshold=0.95)
        reducer.fit(df)
        out = reducer.transform(df)
        # At least one PC column must be present (the largest group has ≥ 1 PC).
        pc_cols = [c for c in out.columns if c.startswith("v_group_") and "_pc_" in c]
        assert len(pc_cols) > 0
        for col in pc_cols:
            # `v_group_{i}_pc_{j}` shape.
            assert col.startswith("v_group_")
            assert "_pc_" in col

    def test_pca_components_have_finite_values(self) -> None:
        df = _build_v_frame()
        reducer = NanGroupReducer(method="pca", pca_variance_threshold=0.95)
        reducer.fit(df)
        out = reducer.transform(df)
        pc_cols = [c for c in out.columns if c.startswith("v_group_") and "_pc_" in c]
        for col in pc_cols:
            assert np.isfinite(out[col]).all()


# ---------------------------------------------------------------------
# Manifest.
# ---------------------------------------------------------------------


class TestManifest:
    """`get_manifest` exposes every drop with a reason."""

    def test_manifest_records_dropped_with_reason(self) -> None:
        df = _build_v_frame()
        reducer = NanGroupReducer(method="correlation", correlation_threshold=0.95)
        reducer.fit(df)
        manifest = reducer.get_manifest()
        # Top-level summary.
        assert manifest["method"] == "correlation"
        assert manifest["n_columns_input"] == 6  # V1..V6
        assert manifest["n_columns_output"] >= 1
        assert manifest["n_columns_dropped"] >= 1
        # Each dropped entry has reason + abs_rho_to_kept.
        all_dropped: list[dict] = []
        for g in manifest["groups"]:
            all_dropped.extend(g["dropped"])
        assert len(all_dropped) == manifest["n_columns_dropped"]
        for d in all_dropped:
            assert d["reason"].startswith("correlated_with_")
            assert 0.0 <= d["abs_rho_to_kept"] <= 1.0

    def test_manifest_pca_records_explained_variance(self) -> None:
        df = _build_v_frame()
        reducer = NanGroupReducer(method="pca", pca_variance_threshold=0.95)
        reducer.fit(df)
        manifest = reducer.get_manifest()
        assert manifest["method"] == "pca"
        # At least one group has the PCA fields populated.
        pca_groups = [g for g in manifest["groups"] if "pca_components" in g]
        assert len(pca_groups) >= 1
        for g in pca_groups:
            ratios = g["pca_explained_variance_ratio"]
            assert isinstance(ratios, list)
            assert all(0.0 <= r <= 1.0 for r in ratios)


# ---------------------------------------------------------------------
# Config + error handling.
# ---------------------------------------------------------------------


class TestConfigLoad:
    """Default constructor reads `tier3_config.yaml`."""

    def test_default_config_loads(self) -> None:
        reducer = NanGroupReducer()
        assert reducer.method == "correlation"
        assert reducer.correlation_threshold == pytest.approx(0.95)
        assert reducer.pca_variance_threshold == pytest.approx(0.95)
        assert reducer.target_col == "isFraud"

    def test_invalid_method_raises(self) -> None:
        with pytest.raises(ValueError, match="method"):
            NanGroupReducer(method="not_a_method")

    def test_constructor_overrides_config(self) -> None:
        reducer = NanGroupReducer(
            method="pca",
            correlation_threshold=0.5,
            pca_variance_threshold=0.8,
            target_col="some_other_target",
        )
        assert reducer.method == "pca"
        assert reducer.correlation_threshold == pytest.approx(0.5)
        assert reducer.pca_variance_threshold == pytest.approx(0.8)
        assert reducer.target_col == "some_other_target"


# ---------------------------------------------------------------------
# Edge cases.
# ---------------------------------------------------------------------


class TestEdgeCases:
    """Boundary conditions: missing target, transform-before-fit, etc."""

    def test_fit_without_target_raises(self) -> None:
        df = pd.DataFrame({"V1": [1.0, 2.0]})
        with pytest.raises(KeyError, match="target column"):
            NanGroupReducer(method="correlation").fit(df)

    def test_transform_before_fit_raises(self) -> None:
        df = _build_v_frame()
        reducer = NanGroupReducer(method="correlation")
        with pytest.raises(AttributeError, match="must be fit"):
            reducer.transform(df)

    def test_get_feature_names_before_fit_raises(self) -> None:
        reducer = NanGroupReducer(method="correlation")
        with pytest.raises(AttributeError, match="must be fit"):
            reducer.get_feature_names()

    def test_manifest_before_fit_raises(self) -> None:
        reducer = NanGroupReducer(method="correlation")
        with pytest.raises(AttributeError, match="must be fit"):
            reducer.get_manifest()

    def test_no_v_columns_yields_empty_groups(self) -> None:
        """A frame with no V-prefix columns produces zero groups."""
        df = pd.DataFrame({"isFraud": [0, 1], "TransactionDT": [0, 1]})
        reducer = NanGroupReducer(method="correlation")
        reducer.fit(df)
        assert reducer.groups_ == []
        assert reducer.kept_columns_ == []
        assert reducer.dropped_columns_ == []
