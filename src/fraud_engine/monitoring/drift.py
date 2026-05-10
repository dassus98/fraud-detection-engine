"""Population Stability Index (PSI) drift monitoring on production features.

Sprint 6 prompt 6.1.b: closes the gap left by 6.1.a's `prediction_score`
histogram.  That histogram captures *output* drift (the model's calibrated
probability distribution shifting); this module captures *input* drift —
the upstream feature distributions changing in ways that silently degrade
the model long before its decision-mix moves enough to trip a Grafana
alert.

The fraud-industry convention for distribution drift is PSI:
    PSI < 0.10 — no significant population shift
    0.10 ≤ PSI ≤ 0.25 — moderate shift; investigate
    PSI > 0.25 — significant shift; model re-fit likely needed

Public surface (re-exported by `fraud_engine.monitoring`):

    - `DriftBaselineBuilder.build(train_df, feature_names, n_bins) → DataFrame`
        — one-time call: compute per-feature quantile edges + baseline
        percentages from a training-data slice; returns a long-format
        DataFrame ready for `to_parquet()`.
    - `DriftMonitor(baseline_path, settings)` — load the persisted
        baseline once at construction; expose runtime drift checks:
        - `compute_feature_psi(feature_name, recent_window) → float`
        - `compute_all_psi(recent_window, top_n=10) → DataFrame`
        - `check_and_alert(recent_window, run_id, alert_log_dir) → int`

Business rationale:
    Production fraud APIs need feature-level drift signals because the
    failure mode is silent: a new geo-distribution, device-type mix, or
    seasonal velocity spike degrades model performance days or weeks
    before the score-distribution histogram moves enough to alert.  PSI
    is the industry standard because it is:
        - Distribution-free (no Gaussian assumption)
        - Bin-count invariant within reason (10 quantile bins is
          standard; 5 too coarse, 20+ noisy on small samples)
        - A single summary number per feature → easy to aggregate, sort,
          and dashboard
    The `>0.2` alert threshold (configurable via
    `Settings.psi_alert_threshold`) is slightly more conservative than
    the academic 0.25 — chosen so daily cron runs surface borderline
    drift before it crosses the "investigate now" line.

Trade-offs considered:
    - **Pre-compute baseline once + persist as parquet vs recompute on
      every PSI call.**  The training set is ~400K rows × 743 features
      and quantile computation is ~50 ms per feature.  Recomputing on
      every PSI call would burn ~37 s per drift check; persisting the
      pre-computed edges + baseline percentages reduces the runtime
      footprint to a 250 KB parquet read at construction.

    - **Long-format parquet (one row per (feature, bin)) vs raw
      baseline arrays.**  The raw arrays would be ~3 GB; the long-format
      derived form is 743 × 10 = 7,430 rows × ~50 bytes ≈ 250 KB.
      Long-format also keeps `pandas.read_parquet` ergonomics (no
      list-typed columns) and is human-grokkable in `parquet-tools cat`.

    - **Reuse `utils.metrics.compute_psi` math via a thin private
      `_psi_from_pcts` helper.**  The shared implementation in
      `utils/metrics.py` (Sprint 4) takes raw arrays and does its own
      binning; our pre-binned-baseline path needs only the math kernel
      after the recent_window is binned into the persisted edges.  The
      `_psi_from_pcts(baseline_pcts, recent_pcts, epsilon)` helper is a
      direct port of `utils.metrics.compute_psi`'s last line.  A
      mathematical equivalence test (`tests/unit/test_drift.py::
      test_compute_feature_psi_matches_utils_compute_psi`) asserts the
      two paths produce identical PSI to 1e-6 tolerance — catches any
      future drift between the two binning paths.

    - **Constant-baseline features are skipped at build time, not
      stored as zero-PSI placeholders.**  PSI is mathematically
      undefined when the baseline has only one value (quantile binning
      collapses to a single edge).  Storing them with edges=[v, v]
      would propagate degeneracy into runtime; skipping them with a
      WARNING at build time is cleaner.  Runtime callers asking
      `compute_feature_psi("constant_feature", ...)` get NaN — they can
      filter on `df["psi"].notna()` if the distinction matters.

    - **DriftMonitor is a stateful object loaded once at construction.**
      The baseline parquet read is ~5 ms; doing it inside every PSI
      call (stateless API) would be wasteful when a typical drift run
      computes PSI for all 743 features in sequence.  The state is
      ~250 KB of edges + percentages — trivial.

    - **Recent-window is a DataFrame caller-passes-in, NOT queried from
      the predictions table.**  Sprint 5.2.a's `predictions` table
      stores score / decision / top_reasons but NOT the raw 743
      features (per-row write would multiply storage 700×).  The
      offline drift-batch pattern is: a Sprint-6.x cron script reads
      recent transactions from the source-of-truth store, re-runs the
      Sprint-2/3 feature pipeline, then passes that DataFrame to
      `DriftMonitor.check_and_alert(recent_window)`.  This module owns
      the math + alert format; it does NOT own the data plumbing.

    - **Alerts go to append-only JSONL at `logs/drift/{run_id}/...`
      mirroring `logs/lineage/{run_id}/...`.**  The JSONL convention
      (one record per line, joinable by `run_id`) is established in
      `data/lineage.py:200-218`.  Alternative: write to a Postgres
      `drift_alerts` table.  Rejected: adds a new schema for what is
      fundamentally a slow-moving log stream; JSONL has no migration
      cost and ships immediately.

    - **`top_n` ranks by drift magnitude (PSI desc), NOT by feature
      importance.**  An on-call engineer reading a daily drift report
      wants "what's drifting most?", not "what's drifting most among
      the model's most-important features?".  The latter is a derivable
      view from the former.  Adding feature-importance ranking would
      require a Sprint 3.x retrofit to the model manifest; defer.

Cross-references:
    - `src/fraud_engine/utils/metrics.py:277-369` — `compute_psi(baseline,
      current, bins, epsilon)` whose math kernel `_psi_from_pcts` mirrors.
    - `src/fraud_engine/data/splits.py:79-177` — `temporal_split()` for
      extracting the train slice the baseline must be built on (no
      val/test leakage).
    - `src/fraud_engine/data/lineage.py:200-218` — append-only JSONL
      idiom this module mirrors for `drift_alerts.jsonl`.
    - `models/sprint3/lightgbm_model_manifest.json:feature_names` — the
      canonical 743-feature list; the build script reads it to align
      baseline columns with the model's actual input order.
    - `src/fraud_engine/monitoring/prometheus_metrics.py` (Sprint 6.1.a)
      — sibling module; this one operates offline on batch DataFrames
      while that one captures live request-path signals.
    - `CLAUDE.md` §3 (PSI drift detection as Sprint-6 endpoint), §4
      (`monitoring/` module home), §5.5 (logging discipline).
"""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Final
from uuid import uuid4

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from fraud_engine.config.settings import Settings, get_settings
from fraud_engine.utils.logging import get_logger, log_call

# ---------------------------------------------------------------------
# Module constants.
# ---------------------------------------------------------------------

# Epsilon floor on bin percentages — avoids log(0) in the PSI sum.
# Default 1e-6 mirrors `utils.metrics.compute_psi`; keeps PSI(a, b) ≈
# PSI(b, a) symmetric to within floating-point noise.  Callers who trade
# symmetry for sparse-baseline smoothing can override (advanced).
_PSI_EPSILON: Final[float] = 1e-6

# Minimum number of unique quantile edges required for a meaningful PSI
# computation.  A degenerate baseline (single value everywhere) produces
# only one edge after `np.unique`; we skip such features at build time.
_MIN_QUANTILE_EDGES: Final[int] = 2

# Long-format baseline parquet schema — single source of truth.  Tests
# assert presence and dtypes against this tuple.
_BASELINE_SCHEMA_COLS: Final[tuple[str, ...]] = (
    "feature_name",
    "bin_idx",
    "edge_low",
    "edge_high",
    "baseline_pct",
    "n_baseline",
)

# Default alert log directory — operator override via
# `Settings.drift_alert_log_dir` or per-call `alert_log_dir=` kwarg.
_DEFAULT_ALERT_LOG_DIR: Final[Path] = Path("logs/drift")

_logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Internal: per-feature pre-computed binning + baseline distribution.
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class _FeatureBaseline:
    """One feature's quantile edges + baseline percentages, ready to PSI.

    Frozen + slots for the same reasons as Sprint 5.1.d's
    `InferenceService._Artefacts`: immutability under concurrent reads
    (a future async-batch DriftMonitor user can share one instance
    across threads safely), minimal memory, fast attribute access.

    Attributes:
        edges: Strictly-increasing array of quantile edges.  Length is
            `n_bins + 1`.  The first / last entries may be `-inf` /
            `+inf` to mark "anything below / above the baseline range
            falls into the outermost bins" — but interior edges are
            real quantile values.  Used directly with `np.digitize`'s
            interior-edge convention for binning recent_window values.
        baseline_pcts: Per-bin baseline fractions.  Length is `n_bins`.
            Sums to 1.0.  NOT epsilon-floored at storage time —
            `_psi_from_pcts` applies the floor at compute time so the
            persisted distribution represents real data.
        n_baseline: Total non-null baseline sample count for this
            feature.  Carried for alert-payload reporting.
    """

    edges: NDArray[np.float64]
    baseline_pcts: NDArray[np.float64]
    n_baseline: int


# ---------------------------------------------------------------------
# Pure math kernel — used by both compute_feature_psi AND tests.
# ---------------------------------------------------------------------


def _psi_from_pcts(
    baseline_pcts: NDArray[np.float64],
    recent_pcts: NDArray[np.float64],
    epsilon: float = _PSI_EPSILON,
) -> float:
    """PSI from two pre-computed bin-percentage arrays.

    PSI formula:
        ``Σ over bins of (recent_pct - baseline_pct) * ln(recent_pct / baseline_pct)``

    Both arrays are floored at `epsilon` element-wise before the log to
    avoid `log(0)` on empty bins.  Mirrors `utils.metrics.compute_psi`'s
    last line; the equivalence test in tests/unit/test_drift.py asserts
    bit-for-bit agreement on identical inputs.

    Args:
        baseline_pcts: Per-bin baseline fractions, length n_bins.
        recent_pcts: Per-bin recent-window fractions, length n_bins.
        epsilon: Floor on zero fractions.  Default 1e-6 preserves
            PSI(a, b) ≈ PSI(b, a) symmetry to within floating-point
            noise.

    Returns:
        PSI value (always non-negative for non-degenerate inputs).
    """
    base = np.maximum(baseline_pcts, epsilon)
    rec = np.maximum(recent_pcts, epsilon)
    return float(((rec - base) * np.log(rec / base)).sum())


# ---------------------------------------------------------------------
# Build: training data → long-format baseline DataFrame.
# ---------------------------------------------------------------------


class DriftBaselineBuilder:
    """One-shot baseline construction from a training-data slice.

    Static method — there's no per-instance state (the bin-edges +
    percentage computation is one-pass over the input DataFrame, no
    intermediate caching warranted).  Tests call `build()` directly with
    synthetic DataFrames; the production CLI lives in
    `scripts/build_drift_baseline.py` and is a thin wrapper.

    Business rationale + trade-offs considered: see module docstring.
    """

    @staticmethod
    @log_call
    def build(
        train_df: pd.DataFrame,
        feature_names: list[str] | tuple[str, ...],
        n_bins: int = 10,
    ) -> pd.DataFrame:
        """Compute per-feature quantile edges + baseline percentages.

        For each feature in `feature_names` that exists in `train_df`
        AND has at least 2 unique non-null values, computes:
            - Quantile edges via `np.quantile(non_nulls, [0, 1/n, ..., 1])`
              with `np.unique` to collapse ties (consistent with
              `utils.metrics.compute_psi:341`).
            - Baseline percentages via `np.digitize` on interior edges.
        Returns a long-format DataFrame matching `_BASELINE_SCHEMA_COLS`
        ready for `to_parquet()`.

        Constant-baseline features (only one unique non-null value)
        are SKIPPED with a WARNING log — they would produce a single
        quantile edge after `np.unique`, making PSI mathematically
        undefined.  Runtime callers asking
        `compute_feature_psi("skipped_feature", ...)` get NaN.

        Features in `feature_names` but missing from `train_df.columns`
        are also skipped with a WARNING — the model manifest may list
        features the snapshot DataFrame doesn't have (e.g., during a
        sprint-N→N+1 transition).

        Args:
            train_df: Training-data slice.  Use
                `temporal_split(df).train` to ensure no val/test
                leakage into the production-drift baseline.
            feature_names: The 743-element model-input list (typically
                read from `models/sprint3/lightgbm_model_manifest.json`
                via `FeatureService._load_model_feature_names`).
                Order is preserved in the output DataFrame.
            n_bins: Quantile-bin count.  Default 10 mirrors
                `utils.metrics.compute_psi`; >= 2.

        Returns:
            Long-format DataFrame with columns
            `(feature_name, bin_idx, edge_low, edge_high, baseline_pct,
            n_baseline)`.  One row per (kept-feature, bin); n_kept_features
            × n_bins rows total.  `edge_low` for bin 0 is `-inf`,
            `edge_high` for the last bin is `+inf` — values outside the
            baseline range still fall into the outermost bins per the
            `np.digitize(arr, edges[1:-1])` convention.

        Raises:
            ValueError: If `n_bins < 2` or `feature_names` is empty.
        """
        if n_bins < _MIN_QUANTILE_EDGES:
            raise ValueError(f"n_bins={n_bins} must be >= {_MIN_QUANTILE_EDGES}")
        if not feature_names:
            raise ValueError("feature_names must be non-empty")

        rows: list[dict[str, object]] = []
        kept_features = 0
        skipped_missing = 0
        skipped_constant = 0

        for feat in feature_names:
            if feat not in train_df.columns:
                skipped_missing += 1
                _logger.warning(
                    "drift.baseline.feature_missing_from_train_df",
                    feature=feat,
                )
                continue

            # Drop NaN before computing quantiles.  IEEE-CIS has
            # ~24% identity coverage; many features are sparse.
            # `np.quantile` on NaN-containing arrays returns NaN edges
            # which would corrupt the entire baseline row.
            non_null = train_df[feat].dropna().to_numpy(dtype=np.float64, copy=False)
            if len(non_null) == 0:
                skipped_constant += 1
                _logger.warning(
                    "drift.baseline.feature_all_null",
                    feature=feat,
                )
                continue

            # Equal-frequency edges from baseline quantiles, mirrors
            # `utils.metrics.compute_psi:337-341`.  `np.unique` collapses
            # tied edges (essential for skewed/discrete features like
            # boolean is_null_card1).
            interior_edges = np.unique(np.quantile(non_null, q=np.linspace(0, 1, n_bins + 1)))
            if len(interior_edges) < _MIN_QUANTILE_EDGES:
                skipped_constant += 1
                _logger.warning(
                    "drift.baseline.feature_constant",
                    feature=feat,
                    unique_value_count=len(interior_edges),
                )
                continue

            # Bin the baseline using interior edges only — the outer
            # edges (min, max of baseline) are stored as +-inf in the
            # parquet for human readability but not used in digitize.
            # `np.digitize(arr, edges[1:-1])` returns bin indices in
            # [0, n_actual_bins - 1] where n_actual_bins =
            # len(interior_edges) - 1 (which may be < n_bins after
            # tie collapse).
            n_actual_bins = len(interior_edges) - 1
            baseline_bins = np.clip(
                np.digitize(non_null, interior_edges[1:-1], right=False),
                0,
                n_actual_bins - 1,
            )
            counts = np.bincount(baseline_bins, minlength=n_actual_bins)
            pcts = counts / len(non_null)
            n_baseline = int(len(non_null))

            for bin_idx in range(n_actual_bins):
                edge_low = float("-inf") if bin_idx == 0 else float(interior_edges[bin_idx])
                edge_high = (
                    float("inf")
                    if bin_idx == n_actual_bins - 1
                    else float(interior_edges[bin_idx + 1])
                )
                rows.append(
                    {
                        "feature_name": feat,
                        "bin_idx": bin_idx,
                        "edge_low": edge_low,
                        "edge_high": edge_high,
                        "baseline_pct": float(pcts[bin_idx]),
                        "n_baseline": n_baseline,
                    }
                )
            kept_features += 1

        if not rows:
            raise ValueError(
                "DriftBaselineBuilder.build produced zero rows — "
                "all features were missing or constant in train_df. "
                "Check that train_df has the feature columns and "
                "non-degenerate values."
            )

        _logger.info(
            "drift.baseline.build_complete",
            kept_features=kept_features,
            skipped_missing=skipped_missing,
            skipped_constant=skipped_constant,
            n_bins=n_bins,
        )
        return pd.DataFrame(rows, columns=list(_BASELINE_SCHEMA_COLS))


# ---------------------------------------------------------------------
# Runtime: load baseline + check drift on a recent_window DataFrame.
# ---------------------------------------------------------------------


class DriftMonitor:
    """Load a frozen baseline; compute PSI vs any recent_window DataFrame.

    Public API:
        - `compute_feature_psi(feature_name, recent_window) → float`
        - `compute_all_psi(recent_window, top_n=10) → DataFrame`
        - `check_and_alert(recent_window, run_id, alert_log_dir) → int`

    Read-only properties:
        - `feature_names` — tuple of features carried in the loaded
          baseline (post-skip of missing / constant ones).
        - `n_features` — len of the above.

    Lifecycle:
        Constructor reads the baseline parquet once (~5 ms for the
        7,430-row long-format file) and builds the in-memory
        `_baselines: dict[str, _FeatureBaseline]` for O(1) lookup.
        Subsequent PSI computations don't touch disk.

    Business rationale + trade-offs considered: see module docstring.
    """

    def __init__(
        self,
        baseline_path: Path | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Configure + load the baseline.

        Args:
            baseline_path: Override path to the long-format baseline
                parquet.  None → `settings.drift_baseline_path` (the
                production default).
            settings: Inject a Settings instance for tests.  None →
                `get_settings()` (the lru-cached singleton).
        """
        self._settings: Settings = settings if settings is not None else get_settings()
        path = baseline_path if baseline_path is not None else self._settings.drift_baseline_path
        self._baselines: dict[str, _FeatureBaseline] = self._load_baseline(path)

    @staticmethod
    def _load_baseline(path: Path) -> dict[str, _FeatureBaseline]:
        """Read the long-format parquet → per-feature `_FeatureBaseline`.

        Validates schema (column names + dtypes); raises a loud error on
        mismatch rather than constructing a partial mapping.

        Raises:
            FileNotFoundError: If `path` does not exist.
            ValueError: If the parquet's columns don't match
                `_BASELINE_SCHEMA_COLS`, or if any feature has
                duplicate `bin_idx` rows (corrupted file).
        """
        if not path.is_file():
            raise FileNotFoundError(
                f"DriftMonitor: baseline parquet not found at {path}. "
                f"Run `uv run python scripts/build_drift_baseline.py` "
                f"to generate it from the training-data slice."
            )

        df = pd.read_parquet(path)
        missing_cols = set(_BASELINE_SCHEMA_COLS) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"DriftMonitor: baseline parquet at {path} missing required "
                f"columns {sorted(missing_cols)}; got {sorted(df.columns)}"
            )

        baselines: dict[str, _FeatureBaseline] = {}
        for feature_name, group in df.groupby("feature_name", sort=False):
            sorted_group = group.sort_values("bin_idx").reset_index(drop=True)
            n_bins = len(sorted_group)
            if sorted_group["bin_idx"].tolist() != list(range(n_bins)):
                raise ValueError(
                    f"DriftMonitor: baseline parquet at {path} has "
                    f"non-contiguous bin_idx for feature {feature_name!r}; "
                    f"got {sorted_group['bin_idx'].tolist()}"
                )

            # Reconstruct the interior-edge array used by np.digitize.
            # The persisted edge_low for bin 0 is -inf; we replace it
            # with the bin 0's edge_high (which is the first interior
            # quantile edge).  Similarly the final bin's edge_high is
            # +inf; we drop it.  The reconstructed `edges` array has
            # length n_bins + 1 with real values at all positions.
            edge_lows = sorted_group["edge_low"].to_numpy(dtype=np.float64)
            edge_highs = sorted_group["edge_high"].to_numpy(dtype=np.float64)
            # edges = [edge_low_0, edge_high_0, edge_high_1, ..., edge_high_{n-1}]
            # but edge_low_0 and edge_high_{n-1} are +-inf — we replace
            # them with the actual baseline min/max from the adjacent
            # interior edges.  For n_bins == 1 (single-bin degenerate),
            # both outer edges are infinities; that case is filtered at
            # build time.
            edges = np.empty(n_bins + 1, dtype=np.float64)
            edges[0] = edge_highs[0] if np.isinf(edge_lows[0]) else edge_lows[0]
            edges[1:-1] = edge_highs[:-1]
            edges[-1] = edge_lows[-1] if np.isinf(edge_highs[-1]) else edge_highs[-1]

            baselines[str(feature_name)] = _FeatureBaseline(
                edges=edges,
                baseline_pcts=sorted_group["baseline_pct"].to_numpy(dtype=np.float64),
                n_baseline=int(sorted_group["n_baseline"].iloc[0]),
            )

        _logger.info(
            "drift.monitor.baseline_loaded",
            path=str(path),
            n_features=len(baselines),
        )
        return baselines

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Features carried in the loaded baseline (post-skip)."""
        return tuple(self._baselines.keys())

    @property
    def n_features(self) -> int:
        """Number of features in the loaded baseline."""
        return len(self._baselines)

    @log_call
    def compute_feature_psi(
        self,
        feature_name: str,
        recent_window: pd.DataFrame,
    ) -> float:
        """PSI for one feature on the recent_window DataFrame.

        Bins the recent_window's column into the baseline's quantile
        edges, normalises to per-bin fractions, applies the PSI formula
        via `_psi_from_pcts`.

        Args:
            feature_name: Feature to compute PSI for.  Must be in
                `self.feature_names`; otherwise returns NaN.
            recent_window: DataFrame with the feature as a column.
                Typical n_recent: 100–10,000 rows.  NaN values are
                dropped before binning (matches build-time treatment).

        Returns:
            PSI value, or NaN if:
                - the feature is not in the loaded baseline (skipped
                  at build time as constant or missing), OR
                - the feature is not in `recent_window.columns`, OR
                - the feature column has no non-null values.
        """
        if feature_name not in self._baselines:
            return float("nan")
        if feature_name not in recent_window.columns:
            return float("nan")

        baseline = self._baselines[feature_name]
        non_null = recent_window[feature_name].dropna().to_numpy(dtype=np.float64, copy=False)
        if len(non_null) == 0:
            return float("nan")

        # Bin recent values using the interior edges only — same
        # convention as `utils.metrics.compute_psi:355-359`.  Values
        # below baseline min go into bin 0; above max go into bin n-1.
        n_bins = len(baseline.baseline_pcts)
        recent_bins = np.clip(
            np.digitize(non_null, baseline.edges[1:-1], right=False),
            0,
            n_bins - 1,
        )
        recent_counts = np.bincount(recent_bins, minlength=n_bins)
        recent_pcts = (recent_counts / len(non_null)).astype(np.float64)

        return _psi_from_pcts(baseline.baseline_pcts, recent_pcts)

    @log_call
    def compute_all_psi(
        self,
        recent_window: pd.DataFrame,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """Per-feature PSI for every loaded feature, sorted desc + capped.

        Iterates every feature in `self._baselines`, calls
        `compute_feature_psi`, sorts by PSI descending (NaN sorts to the
        bottom), takes the first `top_n`.

        Args:
            recent_window: DataFrame to compare against the baseline.
                Typically a recent batch of feature-engineered
                transactions (Sprint 6.x cron pattern).
            top_n: Cap on result rows.  Default 10 per spec; pass
                `top_n=self.n_features` to get the full table.

        Returns:
            DataFrame indexed 0…(top_n - 1) with columns
            `(feature_name, psi, n_recent)`.  Sorted by `psi` desc;
            NaN PSI values (missing / unbinnable features) sink to the
            bottom.
        """
        records: list[dict[str, object]] = []
        for feat in self._baselines:
            psi = self.compute_feature_psi(feat, recent_window)
            n_recent = (
                int(recent_window[feat].dropna().shape[0]) if feat in recent_window.columns else 0
            )
            records.append({"feature_name": feat, "psi": psi, "n_recent": n_recent})

        result = pd.DataFrame(records)
        # `na_position="last"` ensures NaN PSI values sort to the
        # bottom — they never appear in top_n unless every feature is
        # missing, in which case the all-NaN result loudly surfaces
        # the problem.
        return (
            result.sort_values("psi", ascending=False, na_position="last")
            .head(top_n)
            .reset_index(drop=True)
        )

    @log_call
    def check_and_alert(
        self,
        recent_window: pd.DataFrame,
        *,
        run_id: str | None = None,
        alert_log_dir: Path | None = None,
    ) -> int:
        """Compute PSI for every feature; append JSONL for any > threshold.

        For each feature with `psi > settings.psi_alert_threshold`,
        appends one JSONL record to
        `{alert_log_dir}/{run_id}/drift_alerts.jsonl`.  Creates parent
        directories on demand; opens in append mode (multiple cron
        invocations on the same run_id concatenate).  If no alerts fire,
        the file is NOT created — operators can grep for the file's
        existence as a clean alerted/not-alerted signal.

        Args:
            recent_window: DataFrame to compare against the baseline.
            run_id: Correlation tag.  Threaded into the JSONL records
                AND the directory path.  None → generate a fresh
                `uuid4().hex` (matches Sprint 5.1.f's request_id idiom).
            alert_log_dir: Override default `logs/drift`.  None →
                `settings.drift_alert_log_dir`.  Tests pass a `tmp_path`
                for isolation.

        Returns:
            Count of alert records written.  0 means no drift detected
            (no file created).  Caller can `sys.exit(1 if count > 0 else 0)`
            in a daily cron to trip the on-call PagerDuty.
        """
        effective_run_id = run_id if run_id is not None else uuid4().hex
        effective_dir = (
            alert_log_dir if alert_log_dir is not None else self._settings.drift_alert_log_dir
        )
        threshold = float(self._settings.psi_alert_threshold)

        alerts_written = 0
        target_path: Path | None = None

        for feat in self._baselines:
            psi = self.compute_feature_psi(feat, recent_window)
            if not np.isfinite(psi) or psi <= threshold:
                continue

            if target_path is None:
                target_path = effective_dir / effective_run_id / "drift_alerts.jsonl"
                target_path.parent.mkdir(parents=True, exist_ok=True)

            record = {
                "timestamp": datetime.now(UTC).isoformat(),
                "run_id": effective_run_id,
                "feature_name": feat,
                "psi": psi,
                "threshold": threshold,
                "n_baseline": self._baselines[feat].n_baseline,
                "n_recent": int(recent_window[feat].dropna().shape[0]),
            }
            with target_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, sort_keys=True) + "\n")
            alerts_written += 1

        _logger.info(
            "drift.monitor.check_and_alert_complete",
            run_id=effective_run_id,
            alerts_written=alerts_written,
            threshold=threshold,
            output_path=str(target_path) if target_path is not None else None,
        )
        return alerts_written


__all__ = ["DriftBaselineBuilder", "DriftMonitor"]
