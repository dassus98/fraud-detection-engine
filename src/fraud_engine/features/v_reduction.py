"""V-feature dimensionality reduction by NaN-group exploitation.

IEEE-CIS publishes 339 anonymised V-features (`V1` … `V339`). Many
share *identical NaN patterns* — when one is missing, others are too —
because they were measured by the same upstream sensor / pipeline.
Within a NaN-group the surviving columns tend to be highly correlated;
keeping all of them inflates the feature space without proportional
signal gain.

`NanGroupReducer` exploits this structure with two interchangeable
strategies (selected via `tier3_config.yaml:v_reduction_method`):

1. **correlation mode** — within each NaN-group, keep the V column
   with the highest absolute Pearson correlation against `isFraud`;
   drop siblings whose absolute correlation with the kept column
   exceeds `nan_group_correlation_threshold` (default 0.95). The
   kept column carries the same target signal as its dropped
   siblings.

2. **pca mode** — within each NaN-group, fit a `StandardScaler` +
   `PCA` and replace the group's columns with PCA components
   keeping `pca_variance_threshold` of the within-group variance
   (default 0.95). Output columns are named
   `v_group_{group_id}_pc_{i}`.

Both modes write a `v_reduction_manifest.json` listing every dropped
column with the reason — important provenance for the Sprint-3 tuning
sweep, which needs to know what was discarded and why.

Business rationale:
    Wide V-feature spaces hurt LightGBM's per-tree split selection.
    Pre-reducing the space lets the model concentrate on the columns
    that actually carry signal. The 2.1.d / 2.2.e Tier-1/2 builds saw
    val AUC plateau (0.9143–0.9165) at default hyperparameters precisely
    because the model couldn't discriminate signal from redundancy in
    the wide V space. Reducing V columns is expected to either match
    that AUC at lower variance OR free LightGBM to discover new
    interactions Sprint-3 tuning will exploit.

Trade-offs considered:
    - **NaN-group equivalence.** Two columns are in the same group
      iff their `isna()` boolean vectors are bit-identical. We hash
      each column's NaN bytes and bucket. Cheap; deterministic.
    - **Greedy keep-by-target-correlation.** Within a group, sort
      columns by `|ρ(col, isFraud)|` descending; keep top, drop
      siblings highly correlated with kept. Greedy is optimal-enough
      and reproducible; a global integer-programming solution would
      be neater but the runtime cost is not justified at this scale.
    - **Pearson correlation, NaN-aware.** Each pairwise correlation
      uses `pd.Series.corr` (drops NaN-NaN pairs by default). Within a
      NaN-group every column shares the same NaN mask, so all pairwise
      correlations use the same row subset.
    - **Constant columns.** `np.corrcoef` returns NaN when one
      variable is constant. Treated as `0` (no signal); never kept
      as the "most-correlated-with-target" anchor.
    - **PCA at transform time** fills NaN with `0` after StandardScaler
      (so NaN imputes to the column mean). Within a NaN-group all
      members share the NaN mask, so this only fires for rows where
      the entire group is missing — and `0` (= post-scale mean) is
      the lossless default for a fully-missing group.
    - **Drops columns from the input frame** — exception to the
      `BaseFeatureGenerator` 'preserve all columns' rule. Documented
      in the class docstring; the canonical pipeline placement is
      AFTER all generators that ADD columns (Tier 1-3), so no
      downstream stage references the dropped V columns.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final, Self

import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from fraud_engine.features.base import BaseFeatureGenerator

# ---------------------------------------------------------------------
# Module constants.
# ---------------------------------------------------------------------

_DEFAULT_TARGET_COL: Final[str] = "isFraud"
_DEFAULT_METHOD: Final[str] = "correlation"
_DEFAULT_CORR_THRESHOLD: Final[float] = 0.95
_DEFAULT_PCA_VARIANCE: Final[float] = 0.95

_TIER3_CONFIG_FILENAME: Final[str] = "tier3_config.yaml"
_VALID_METHODS: Final[frozenset[str]] = frozenset({"correlation", "pca"})

_MANIFEST_SCHEMA_VERSION: Final[int] = 1

# Pearson correlation needs at least this many overlapping non-NaN
# pairs before the result is meaningful.
_MIN_PAIRS_FOR_CORR: Final[int] = 2


# ---------------------------------------------------------------------
# YAML helpers.
# ---------------------------------------------------------------------


def _resolve_config_path(filename: str) -> Path:
    """Resolve `configs/{filename}` against the repo root."""
    project_root = Path(__file__).resolve().parents[3]
    return project_root / "configs" / filename


def _load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file; raise if non-mapping."""
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected top-level mapping in {path}, got {type(loaded).__name__}")
    return loaded


def _detect_v_columns(df: pd.DataFrame) -> list[str]:
    """Auto-detect the IEEE-CIS V-feature columns (`V1`, `V2`, ..., `V339`)."""
    return sorted(
        (c for c in df.columns if isinstance(c, str) and c.startswith("V") and c[1:].isdigit()),
        key=lambda c: int(c[1:]),
    )


def _nan_signature(series: pd.Series[Any]) -> str:
    """Stable hex hash of a column's NaN mask.

    Two columns return the same signature iff their `isna()` boolean
    vectors are bit-identical.
    """
    mask = series.isna().to_numpy()
    return hashlib.sha256(mask.tobytes()).hexdigest()


def _abs_corr(a: pd.Series[Any], b: pd.Series[Any]) -> float:
    """Absolute Pearson correlation, NaN-aware. Returns 0 if undefined.

    `pd.Series.corr` uses pairwise complete observations; if either
    series is constant or fewer than 2 overlapping non-NaN pairs exist,
    the result is NaN, which we coerce to 0 (no signal).
    """
    if a.isna().all() or b.isna().all():
        return 0.0
    overlapping = a.notna() & b.notna()
    if int(overlapping.sum()) < _MIN_PAIRS_FOR_CORR:
        return 0.0
    rho = a.corr(b)
    if rho is None or pd.isna(rho):
        return 0.0
    return float(abs(rho))


# ---------------------------------------------------------------------
# `NanGroupReducer`.
# ---------------------------------------------------------------------


class NanGroupReducer(BaseFeatureGenerator):
    """V-feature reduction by NaN-group correlation OR PCA.

    See module docstring for business rationale and trade-offs. This
    class is the **one exception** to the `BaseFeatureGenerator`
    'preserve all columns' rule — it exists to remove columns. The
    canonical pipeline placement is AFTER all column-adding generators
    so no downstream stage references the dropped V columns.

    Attributes:
        target_col: name of the binary target column (default `isFraud`).
        v_columns: explicit V-column override; if `None`, auto-detect
            via `_detect_v_columns`.
        method: `"correlation"` or `"pca"`.
        correlation_threshold: drop a sibling if `|ρ(kept, sibling)| >`
            this. Used in correlation mode.
        pca_variance_threshold: keep PCA components up to this fraction
            of within-group variance. Used in PCA mode.
        groups_: list of per-group manifest dicts; populated by `fit`.
        kept_columns_: list of output V-column names after reduction;
            populated by `fit`.
        dropped_columns_: list of input columns dropped (or replaced
            in PCA mode); populated by `fit`.
        pca_models_: per-group dict holding the fitted scaler + PCA;
            populated by `fit` only when `method == "pca"`.
    """

    def __init__(  # noqa: PLR0913 — six explicit kwargs reflect the six YAML overrides; condensing into a config dict would be worse for callers.
        self,
        target_col: str | None = None,
        v_columns: Sequence[str] | None = None,
        method: str | None = None,
        correlation_threshold: float | None = None,
        pca_variance_threshold: float | None = None,
        config_path: Path | None = None,
    ) -> None:
        """Construct the V-reducer.

        Each kwarg falls back to the YAML default when `None` (except
        `v_columns`, which auto-detects from the input frame at fit
        time when left `None`).
        """
        if (
            target_col is None
            or method is None
            or correlation_threshold is None
            or pca_variance_threshold is None
        ):
            cfg = _load_yaml(config_path or _resolve_config_path(_TIER3_CONFIG_FILENAME))
        else:
            cfg = {}

        self.target_col: str = (
            target_col if target_col is not None else cfg.get("target_col", _DEFAULT_TARGET_COL)
        )
        self.method: str = (
            method if method is not None else cfg.get("v_reduction_method", _DEFAULT_METHOD)
        )
        if self.method not in _VALID_METHODS:
            raise ValueError(
                f"NanGroupReducer: method {self.method!r} not in {sorted(_VALID_METHODS)}"
            )
        self.correlation_threshold: float = float(
            correlation_threshold
            if correlation_threshold is not None
            else cfg.get("nan_group_correlation_threshold", _DEFAULT_CORR_THRESHOLD)
        )
        self.pca_variance_threshold: float = float(
            pca_variance_threshold
            if pca_variance_threshold is not None
            else cfg.get("pca_variance_threshold", _DEFAULT_PCA_VARIANCE)
        )
        self.v_columns: tuple[str, ...] | None = tuple(v_columns) if v_columns is not None else None

        # Fitted state.
        self.groups_: list[dict[str, Any]] | None = None
        self.kept_columns_: list[str] | None = None
        self.dropped_columns_: list[str] | None = None
        self.pca_models_: dict[str, dict[str, Any]] = {}

    # -----------------------------------------------------------------
    # `BaseFeatureGenerator` contract.
    # -----------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> Self:
        """Identify NaN-groups; learn drops (correlation) or PCA models.

        Args:
            df: training frame; must contain `target_col` and the V
                columns (auto-detected if `v_columns` is None).

        Returns:
            self.

        Raises:
            KeyError: if `target_col` is missing from `df`.
        """
        if self.target_col not in df.columns:
            raise KeyError(f"NanGroupReducer.fit: target column {self.target_col!r} not in df")

        v_cols = list(self.v_columns) if self.v_columns is not None else _detect_v_columns(df)

        # Group by NaN signature.
        signatures: dict[str, list[str]] = defaultdict(list)
        for col in v_cols:
            if col not in df.columns:
                continue
            signatures[_nan_signature(df[col])].append(col)

        # Deterministic per-group ordering: sort by (size desc, smallest col name)
        # so the manifest reads top-down by group size.
        ordered_groups = sorted(
            signatures.items(),
            key=lambda kv: (-len(kv[1]), sorted(kv[1])[0] if kv[1] else ""),
        )

        groups: list[dict[str, Any]] = []
        kept: list[str] = []
        dropped: list[str] = []
        pca_models: dict[str, dict[str, Any]] = {}

        target = df[self.target_col]

        for group_id, (sig, cols) in enumerate(ordered_groups):
            cols_sorted = sorted(cols, key=lambda c: int(c[1:]))
            group_info: dict[str, Any] = {
                "group_id": group_id,
                "nan_signature_hash": sig,
                "size": len(cols_sorted),
                "input_columns": cols_sorted,
                "kept": [],
                "dropped": [],
            }

            if len(cols_sorted) == 1:
                kept.extend(cols_sorted)
                group_info["kept"] = cols_sorted
            elif self.method == "correlation":
                self._reduce_by_correlation(df, target, cols_sorted, group_info, kept, dropped)
            else:  # method == "pca"
                self._reduce_by_pca(
                    df, cols_sorted, group_id, group_info, kept, dropped, pca_models
                )

            groups.append(group_info)

        self.groups_ = groups
        self.kept_columns_ = kept
        self.dropped_columns_ = dropped
        self.pca_models_ = pca_models
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted reduction.

        Correlation mode drops the learned columns. PCA mode replaces
        each group's columns with its fitted components.

        Raises:
            AttributeError: if `fit` hasn't run.
        """
        if self.groups_ is None or self.dropped_columns_ is None:
            raise AttributeError("NanGroupReducer must be fit before transform")

        out = df.copy()

        if self.method == "correlation":
            present = [c for c in self.dropped_columns_ if c in out.columns]
            if present:
                out = out.drop(columns=present)
            return out

        # PCA mode.
        for _group_key, model in self.pca_models_.items():
            input_cols: list[str] = model["input_cols"]
            output_cols: list[str] = model["output_cols"]
            scaler: StandardScaler = model["scaler"]
            pca: PCA = model["pca"]

            present_inputs = [c for c in input_cols if c in out.columns]
            if not present_inputs:
                continue
            # NaN within a group is by construction shared across
            # members; fillna(0) post-scaling means impute to the
            # column mean — see module trade-off note.
            scaled = scaler.transform(out[input_cols].fillna(0.0))
            transformed = pca.transform(scaled)
            for i, col_name in enumerate(output_cols):
                out[col_name] = transformed[:, i]
            out = out.drop(columns=present_inputs)

        return out

    def get_feature_names(self) -> list[str]:
        """Return the OUTPUT V-column names after reduction.

        For correlation mode: the surviving (kept) V columns.
        For PCA mode: the generated `v_group_{i}_pc_{j}` columns.

        Raises:
            AttributeError: if `fit` hasn't run.
        """
        if self.kept_columns_ is None:
            raise AttributeError("NanGroupReducer must be fit before get_feature_names")
        return list(self.kept_columns_)

    def get_business_rationale(self) -> str:
        """Return the manifest-rendered business rationale."""
        return (
            "V-feature dimensionality reduction by NaN-group "
            "exploitation. IEEE-CIS V columns share NaN patterns "
            "by upstream sensor; within a group columns are highly "
            "correlated. Reduction keeps the most target-correlated "
            "column (correlation mode) or runs PCA to a variance "
            "threshold (PCA mode). Pre-reducing wide V-spaces lets "
            "Sprint 3's LightGBM concentrate on signal-bearing "
            "columns rather than redundant siblings."
        )

    # -----------------------------------------------------------------
    # Reduction helpers.
    # -----------------------------------------------------------------

    def _reduce_by_correlation(  # noqa: PLR0913 — six positional args carry the per-group reduction state; collapsing into a struct would obscure the mutation pattern.
        self,
        df: pd.DataFrame,
        target: pd.Series[Any],
        cols: list[str],
        group_info: dict[str, Any],
        kept: list[str],
        dropped: list[str],
    ) -> None:
        """Greedy keep-by-target-correlation; drop siblings at threshold."""
        # 1. Sort cols by |corr(col, target)| descending (ties broken by name).
        target_corrs = {col: _abs_corr(df[col], target) for col in cols}
        cols_ranked = sorted(cols, key=lambda c: (-target_corrs[c], c))

        # 2. Greedy: keep the top, drop siblings with |ρ(kept, sibling)| > threshold.
        kept_in_group: list[str] = []
        for col in cols_ranked:
            anchor: str | None = None
            anchor_rho = 0.0
            for kept_col in kept_in_group:
                rho = _abs_corr(df[col], df[kept_col])
                if rho > self.correlation_threshold:
                    anchor = kept_col
                    anchor_rho = rho
                    break
            if anchor is not None:
                dropped.append(col)
                group_info["dropped"].append(
                    {
                        "column": col,
                        "reason": f"correlated_with_{anchor}",
                        "abs_rho_to_kept": anchor_rho,
                        "abs_rho_to_target": target_corrs[col],
                    }
                )
            else:
                kept_in_group.append(col)
                kept.append(col)
                group_info["kept"].append(col)

    def _reduce_by_pca(  # noqa: PLR0913 — seven positional args are the per-group reduction state plus the persistent PCA-models map; collapsing would obscure the mutation pattern.
        self,
        df: pd.DataFrame,
        cols: list[str],
        group_id: int,
        group_info: dict[str, Any],
        kept: list[str],
        dropped: list[str],
        pca_models: dict[str, dict[str, Any]],
    ) -> None:
        """Replace the group's columns with PCA components to variance threshold."""
        # Use rows where ALL group members are non-NaN. Within a NaN-group
        # they share the mask, so this is just the non-NaN row subset.
        group_data = df[cols].dropna()
        if len(group_data) < _MIN_PAIRS_FOR_CORR:
            # Not enough non-NaN rows; keep the group as-is.
            kept.extend(cols)
            group_info["kept"] = cols
            group_info["pca_skipped_reason"] = "insufficient_non_nan_rows"
            return

        scaler = StandardScaler()
        scaled = scaler.fit_transform(group_data)
        pca = PCA(n_components=self.pca_variance_threshold)
        pca.fit(scaled)
        n_components = int(pca.n_components_)

        group_key = f"v_group_{group_id}"
        output_cols = [f"{group_key}_pc_{i}" for i in range(n_components)]
        pca_models[group_key] = {
            "scaler": scaler,
            "pca": pca,
            "input_cols": cols,
            "output_cols": output_cols,
        }
        dropped.extend(cols)
        kept.extend(output_cols)
        group_info["kept"] = output_cols
        group_info["dropped"] = [
            {"column": col, "reason": "replaced_by_pca", "group_key": group_key} for col in cols
        ]
        group_info["pca_components"] = n_components
        group_info["pca_explained_variance_ratio"] = pca.explained_variance_ratio_.tolist()

    # -----------------------------------------------------------------
    # Manifest.
    # -----------------------------------------------------------------

    def get_manifest(self) -> dict[str, Any]:
        """Render the v_reduction_manifest dict.

        Includes top-level summary stats and per-group details (the
        `groups_` list as populated by `fit`). Suitable for
        `json.dumps` write to `v_reduction_manifest.json`.

        Raises:
            AttributeError: if `fit` hasn't run.
        """
        if self.groups_ is None or self.kept_columns_ is None or self.dropped_columns_ is None:
            raise AttributeError("NanGroupReducer must be fit before get_manifest")

        n_input = sum(int(g["size"]) for g in self.groups_)
        return {
            "schema_version": _MANIFEST_SCHEMA_VERSION,
            "method": self.method,
            "correlation_threshold": self.correlation_threshold,
            "pca_variance_threshold": self.pca_variance_threshold,
            "target_col": self.target_col,
            "n_groups": len(self.groups_),
            "n_columns_input": n_input,
            "n_columns_output": len(self.kept_columns_),
            "n_columns_dropped": len(self.dropped_columns_),
            "groups": self.groups_,
        }


__all__ = ["NanGroupReducer"]
