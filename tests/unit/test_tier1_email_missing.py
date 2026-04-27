"""Unit tests for `EmailDomainExtractor` and `MissingIndicatorGenerator`.

Four test surfaces:

- `TestEmailDomainExtractorSpec`: known-input → known-output for
  free / disposable / unknown / null cases; explicit lists override
  YAML; both email columns produce 4 derived features each.
- `TestMissingIndicatorGeneratorSpec`: fit learns columns above
  threshold; transform emits the same set on val (even when val has
  no nulls in those columns); config-driven threshold changes the
  set; pre-fit transform raises.
- `TestContractCompliance`: feature names and rationale on both
  generators.
- `TestPipelineIntegration`: both generators slot into a
  `FeaturePipeline` cleanly and produce the expected columns.

Plus one default-config test (`test_email_loads_default_config`) that
exercises the on-disk YAML; everything else passes lists / threshold
explicitly so the test suite is decoupled from YAML edits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_engine.features import FeaturePipeline
from fraud_engine.features.tier1_basic import (
    EmailDomainExtractor,
    MissingIndicatorGenerator,
)

# Test-only literal sets. Decoupled from the YAML so a future YAML
# extension doesn't ripple test expectations.
_TEST_FREE: frozenset[str] = frozenset({"gmail.com", "yahoo.com"})
_TEST_DISPOSABLE: frozenset[str] = frozenset({"guerrillamail.com", "tempmail.com"})


def _make_extractor(
    email_columns: tuple[str, ...] = ("P_emaildomain",),
) -> EmailDomainExtractor:
    """Construct an extractor with the test literal sets."""
    return EmailDomainExtractor(
        email_columns=email_columns,
        free_providers=_TEST_FREE,
        disposable_providers=_TEST_DISPOSABLE,
    )


# ---------------- EmailDomainExtractor spec ---------------- #


class TestEmailDomainExtractorSpec:
    """Known-input → known-output cases."""

    def test_known_free_provider(self) -> None:
        """gmail.com → provider=gmail, tld=com, is_free=1, is_disposable=0."""
        gen = _make_extractor()
        df = pd.DataFrame({"P_emaildomain": ["gmail.com"]})
        out = gen.fit_transform(df)
        assert out["P_emaildomain_provider"].iloc[0] == "gmail"
        assert out["P_emaildomain_tld"].iloc[0] == "com"
        assert out["P_emaildomain_is_free"].iloc[0] == 1
        assert out["P_emaildomain_is_disposable"].iloc[0] == 0

    def test_known_disposable_provider(self) -> None:
        """guerrillamail.com → is_disposable=1, is_free=0."""
        gen = _make_extractor()
        df = pd.DataFrame({"P_emaildomain": ["guerrillamail.com"]})
        out = gen.fit_transform(df)
        assert out["P_emaildomain_is_disposable"].iloc[0] == 1
        assert out["P_emaildomain_is_free"].iloc[0] == 0

    def test_unknown_provider(self) -> None:
        """weirdvalue.io → split correctly; both flags 0."""
        gen = _make_extractor()
        df = pd.DataFrame({"P_emaildomain": ["weirdvalue.io"]})
        out = gen.fit_transform(df)
        assert out["P_emaildomain_provider"].iloc[0] == "weirdvalue"
        assert out["P_emaildomain_tld"].iloc[0] == "io"
        assert out["P_emaildomain_is_free"].iloc[0] == 0
        assert out["P_emaildomain_is_disposable"].iloc[0] == 0

    def test_null_passes_through_as_na(self) -> None:
        """Null input row → all four derived fields are <NA>."""
        gen = _make_extractor()
        df = pd.DataFrame({"P_emaildomain": [pd.NA]}, dtype="string")
        out = gen.fit_transform(df)
        assert pd.isna(out["P_emaildomain_provider"].iloc[0])
        assert pd.isna(out["P_emaildomain_tld"].iloc[0])
        assert pd.isna(out["P_emaildomain_is_free"].iloc[0])
        assert pd.isna(out["P_emaildomain_is_disposable"].iloc[0])

    def test_explicit_lists_override_yaml(self) -> None:
        """Custom `free_providers` flags a domain not in the on-disk YAML."""
        gen = EmailDomainExtractor(
            email_columns=("P_emaildomain",),
            free_providers=frozenset({"unusual-free.com"}),
            disposable_providers=frozenset({"unusual-disp.com"}),
        )
        df = pd.DataFrame({"P_emaildomain": ["unusual-free.com", "unusual-disp.com"]})
        out = gen.fit_transform(df)
        assert out["P_emaildomain_is_free"].iloc[0] == 1
        assert out["P_emaildomain_is_disposable"].iloc[1] == 1

    def test_handles_both_email_columns(self) -> None:
        """Both P_emaildomain and R_emaildomain produce the 4 derived columns."""
        gen = _make_extractor(email_columns=("P_emaildomain", "R_emaildomain"))
        df = pd.DataFrame(
            {
                "P_emaildomain": ["gmail.com"],
                "R_emaildomain": ["yahoo.com"],
            }
        )
        out = gen.fit_transform(df)
        for col in ("P_emaildomain", "R_emaildomain"):
            assert f"{col}_provider" in out.columns
            assert f"{col}_tld" in out.columns
            assert f"{col}_is_free" in out.columns
            assert f"{col}_is_disposable" in out.columns
            # All four free-domain rows.
            assert out[f"{col}_is_free"].iloc[0] == 1

    def test_email_loads_default_config(self) -> None:
        """Constructing without explicit lists pulls from the on-disk YAML.

        This is the **only** test that exercises the YAML path.
        Verifies a stable known entry (`gmail.com`) is in the loaded
        free-provider set.
        """
        gen = EmailDomainExtractor(email_columns=("P_emaildomain",))
        df = pd.DataFrame({"P_emaildomain": ["gmail.com"]})
        out = gen.fit_transform(df)
        assert out["P_emaildomain_is_free"].iloc[0] == 1


# ---------------- MissingIndicatorGenerator spec ---------------- #


def _missingness_frame(
    n_rows: int = 100,
    col_a_missing: float = 0.10,
    col_b_missing: float = 0.04,
    col_c_missing: float = 0.0,
) -> pd.DataFrame:
    """Build a synthetic frame with controlled per-column missingness."""
    rng = np.random.default_rng(42)
    out = pd.DataFrame(
        {
            "col_a": rng.normal(size=n_rows),
            "col_b": rng.normal(size=n_rows),
            "col_c": rng.normal(size=n_rows),
        }
    )
    out.loc[: int(n_rows * col_a_missing) - 1, "col_a"] = np.nan
    out.loc[: int(n_rows * col_b_missing) - 1, "col_b"] = np.nan
    if col_c_missing > 0:
        out.loc[: int(n_rows * col_c_missing) - 1, "col_c"] = np.nan
    return out


class TestMissingIndicatorGeneratorSpec:
    """Spec-mandated assertions for `MissingIndicatorGenerator`."""

    def test_learns_columns_above_threshold(self) -> None:
        """threshold=0.05, col_a 10% missing, col_b 4% → only col_a learned."""
        gen = MissingIndicatorGenerator(threshold=0.05)
        gen.fit(_missingness_frame())
        assert gen.target_columns == ["col_a"]

    def test_transforms_same_set_for_val(self) -> None:
        """Fit on train; transform on val produces is_null_col_a even if val has 0% nulls."""
        gen = MissingIndicatorGenerator(threshold=0.05).fit(_missingness_frame())
        val = _missingness_frame(col_a_missing=0.0)
        out = gen.transform(val)
        assert "is_null_col_a" in out.columns
        # Val has zero nulls in col_a → indicator is all zeros.
        assert (out["is_null_col_a"] == 0).all()

    def test_config_driven_threshold_changes_behavior(self) -> None:
        """Lowering the threshold expands the learned set."""
        gen_loose = MissingIndicatorGenerator(threshold=0.01).fit(_missingness_frame())
        gen_strict = MissingIndicatorGenerator(threshold=0.05).fit(_missingness_frame())
        assert set(gen_loose.target_columns or []) == {"col_a", "col_b"}
        assert set(gen_strict.target_columns or []) == {"col_a"}

    def test_transform_before_fit_raises(self) -> None:
        """Pre-fit transform raises AttributeError."""
        gen = MissingIndicatorGenerator(threshold=0.05)
        with pytest.raises(AttributeError, match="must be fit"):
            gen.transform(_missingness_frame())

    def test_target_columns_sorted_alphabetically(self) -> None:
        """`target_columns` ordering is deterministic across runs."""
        gen = MissingIndicatorGenerator(threshold=0.01).fit(_missingness_frame())
        assert gen.target_columns == sorted(gen.target_columns or [])

    def test_schema_drift_emits_all_ones(self) -> None:
        """Target column absent at transform → all-1s indicator."""
        gen = MissingIndicatorGenerator(threshold=0.05).fit(_missingness_frame())
        # Drop col_a from the val frame entirely.
        val = _missingness_frame().drop(columns=["col_a"])
        out = gen.transform(val)
        assert "is_null_col_a" in out.columns
        assert (out["is_null_col_a"] == 1).all()


# ---------------- Contract compliance ---------------- #


class TestContractCompliance:
    """Both new generators satisfy `BaseFeatureGenerator` introspection."""

    def test_email_feature_names(self) -> None:
        gen = _make_extractor(email_columns=("P_emaildomain", "R_emaildomain"))
        names = gen.get_feature_names()
        # 4 features × 2 columns = 8 total.
        assert len(names) == 8
        for col in ("P_emaildomain", "R_emaildomain"):
            for suffix in ("provider", "tld", "is_free", "is_disposable"):
                assert f"{col}_{suffix}" in names

    def test_email_rationale_non_empty(self) -> None:
        assert len(_make_extractor().get_business_rationale()) > 50

    def test_missing_feature_names_pre_and_post_fit(self) -> None:
        """Pre-fit returns []; post-fit lists the learned `is_null_*` columns."""
        gen = MissingIndicatorGenerator(threshold=0.05)
        assert gen.get_feature_names() == []  # pre-fit
        gen.fit(_missingness_frame())
        assert gen.get_feature_names() == ["is_null_col_a"]

    def test_missing_rationale_non_empty(self) -> None:
        assert len(MissingIndicatorGenerator(threshold=0.05).get_business_rationale()) > 50


# ---------------- Pipeline integration ---------------- #


class TestPipelineIntegration:
    """Both generators slot into `FeaturePipeline.fit_transform` cleanly."""

    def test_email_and_missing_in_pipeline(self) -> None:
        """A pipeline with both generators produces all expected columns."""
        df = pd.DataFrame(
            {
                "P_emaildomain": ["gmail.com", "guerrillamail.com", None] * 10,
                "col_a": [None] * 10 + list(range(20)),
                "col_b": list(range(30)),
            }
        )
        pipe = FeaturePipeline(
            generators=[
                _make_extractor(),
                MissingIndicatorGenerator(threshold=0.05),
            ]
        )
        out = pipe.fit_transform(df)
        # Email features.
        for suffix in ("provider", "tld", "is_free", "is_disposable"):
            assert f"P_emaildomain_{suffix}" in out.columns
        # Missing indicator on col_a (10/30 ≈ 33% null > 5%).
        assert "is_null_col_a" in out.columns
