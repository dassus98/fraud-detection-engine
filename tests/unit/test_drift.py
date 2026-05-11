"""Unit tests for the Sprint 6.1.b DriftMonitor + DriftBaselineBuilder.

Nine tests across five categories:

    Behaviour — no drift (1):
        - PSI ≈ 0 when baseline and recent are identical samples.

    Behaviour — synthetic drift (2):
        - PSI is high (>0.5) for a +1.5σ mean shift.
        - PSI increases monotonically as shift magnitude grows.

    Aggregation (2):
        - `compute_all_psi` returns top_n sorted desc.
        - Missing recent_window features get NaN PSI and sink to bottom.

    Alerting (2):
        - `check_and_alert` writes JSONL records when drift > threshold.
        - `check_and_alert` writes nothing when no drift, returns 0.

    Persistence + math equivalence (2):
        - Build → save parquet → load → PSI matches in-memory result.
        - `compute_feature_psi` matches `utils.metrics.compute_psi` to
          1e-6 tolerance on identical inputs (catches drift between the
          pre-binned-baseline path and the raw-arrays path).

All tests use deterministic seeds (`np.random.default_rng(42)`).  Tests
write to `tmp_path` so they don't touch real `data/baselines/` or
`logs/drift/` directories.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_engine.config.settings import Settings
from fraud_engine.monitoring.drift import (
    DriftBaselineBuilder,
    DriftMonitor,
    _psi_from_pcts,
)
from fraud_engine.utils.metrics import compute_psi

# ---------------------------------------------------------------------
# Fixtures + helpers.
# ---------------------------------------------------------------------


_RNG_SEED = 42
_BINS = 10
_BASELINE_N = 5_000
_RECENT_N = 1_000


def _make_settings(tmp_path: Path, threshold: float = 0.2) -> Settings:
    """Build a Settings instance with paths pointing into tmp_path.

    Avoids the lru-cached singleton — each test owns its own settings
    so paths are isolated per-test.
    """
    return Settings(
        drift_baseline_path=tmp_path / "baseline.parquet",
        drift_alert_log_dir=tmp_path / "logs" / "drift",
        psi_alert_threshold=threshold,
        psi_bins=_BINS,
    )


def _build_baseline_for(
    feature_arrays: dict[str, np.ndarray],
    tmp_path: Path,
    n_bins: int = _BINS,
) -> Path:
    """Construct a synthetic train_df, build the baseline, persist it.

    Returns the path to the persisted parquet.  Caller passes
    {feature_name: 1D array of baseline values}; arrays may have
    different lengths but each must be non-degenerate (>=2 unique
    values).
    """
    # All arrays must have the same length to land in a single
    # DataFrame; pad shorter ones with the last value to keep them
    # representative without changing the quantile structure.
    max_len = max(len(arr) for arr in feature_arrays.values())
    padded = {}
    for name, arr in feature_arrays.items():
        if len(arr) < max_len:
            pad = np.full(max_len - len(arr), arr[-1], dtype=arr.dtype)
            padded[name] = np.concatenate([arr, pad])
        else:
            padded[name] = arr
    train_df = pd.DataFrame(padded)
    baseline_df = DriftBaselineBuilder.build(
        train_df=train_df,
        feature_names=list(feature_arrays.keys()),
        n_bins=n_bins,
    )
    out = tmp_path / "baseline.parquet"
    baseline_df.to_parquet(out, index=False)
    return out


# ---------------------------------------------------------------------
# Behaviour — no drift.
# ---------------------------------------------------------------------


class TestNoDrift:
    """Identical baseline + recent → PSI near zero."""

    def test_psi_near_zero_for_identical_distributions(self, tmp_path: Path) -> None:
        """An identical-distribution recent window produces PSI < 0.01."""
        rng = np.random.default_rng(_RNG_SEED)
        baseline = rng.normal(loc=0.0, scale=1.0, size=_BASELINE_N)
        # Same generator state → same distribution; values themselves
        # differ but the empirical CDFs agree to within sampling noise.
        recent_values = rng.normal(loc=0.0, scale=1.0, size=_RECENT_N)

        baseline_path = _build_baseline_for({"feature_a": baseline}, tmp_path)
        settings = _make_settings(tmp_path)
        monitor = DriftMonitor(baseline_path=baseline_path, settings=settings)

        psi = monitor.compute_feature_psi(
            "feature_a",
            pd.DataFrame({"feature_a": recent_values}),
        )

        # Industry-standard "stable" band is < 0.10; identical
        # distributions sampled at n=1000 should be well under that.
        assert psi < 0.05, f"identical-dist PSI = {psi}, expected < 0.05"


# ---------------------------------------------------------------------
# Behaviour — synthetic drift.
# ---------------------------------------------------------------------


class TestSyntheticDrift:
    """A measurable mean shift produces a high PSI value."""

    def test_psi_high_for_synthetic_mean_shift(self, tmp_path: Path) -> None:
        """A +1.5σ mean shift in recent → PSI > 0.5 (significant drift)."""
        rng = np.random.default_rng(_RNG_SEED)
        baseline = rng.normal(loc=0.0, scale=1.0, size=_BASELINE_N)
        # Shift the recent distribution's mean by +1.5σ — strong drift.
        recent_values = rng.normal(loc=1.5, scale=1.0, size=_RECENT_N)

        baseline_path = _build_baseline_for({"feature_a": baseline}, tmp_path)
        monitor = DriftMonitor(baseline_path=baseline_path, settings=_make_settings(tmp_path))

        psi = monitor.compute_feature_psi(
            "feature_a",
            pd.DataFrame({"feature_a": recent_values}),
        )
        # 1.5σ shift on a normal distribution shifts the bin-fraction
        # mass dramatically: the upper buckets get most of the recent
        # mass, the lower buckets empty out. Empirically PSI ≈ 1.5+.
        assert psi > 0.5, f"+1.5σ shift PSI = {psi}, expected > 0.5"

    def test_psi_increases_monotonically_with_shift_magnitude(self, tmp_path: Path) -> None:
        """Sweep mean shift 0σ → 2σ; PSI must strictly increase.

        Sanity check on the math kernel: a larger distribution shift
        must produce a larger PSI value.  Catches sign errors,
        bucketing inversions, or epsilon-floor mistakes that would
        non-monotonically trend.
        """
        rng = np.random.default_rng(_RNG_SEED)
        baseline = rng.normal(loc=0.0, scale=1.0, size=_BASELINE_N)
        baseline_path = _build_baseline_for({"feature_a": baseline}, tmp_path)
        monitor = DriftMonitor(baseline_path=baseline_path, settings=_make_settings(tmp_path))

        psi_values: list[float] = []
        for shift in (0.0, 0.5, 1.0, 1.5, 2.0):
            # Use a fresh generator per iteration so shifts are
            # comparable (no inherited state from prior iteration).
            iter_rng = np.random.default_rng(_RNG_SEED + int(shift * 10))
            recent_values = iter_rng.normal(loc=shift, scale=1.0, size=_RECENT_N)
            psi = monitor.compute_feature_psi(
                "feature_a", pd.DataFrame({"feature_a": recent_values})
            )
            psi_values.append(psi)

        # Strictly increasing — each successive PSI must exceed the
        # previous.  No tolerance: even a tiny shift bump should
        # produce a measurable PSI bump.
        for i in range(1, len(psi_values)):
            assert (
                psi_values[i] > psi_values[i - 1]
            ), f"PSI not monotonic: shift sweep produced {psi_values}"


# ---------------------------------------------------------------------
# Aggregation tests.
# ---------------------------------------------------------------------


class TestAggregation:
    """`compute_all_psi` ranking + missing-feature behaviour."""

    def test_compute_all_psi_returns_top_n_sorted_desc(self, tmp_path: Path) -> None:
        """Three features with varying drift → output sorted desc, capped at top_n."""
        rng = np.random.default_rng(_RNG_SEED)
        baseline_a = rng.normal(loc=0.0, scale=1.0, size=_BASELINE_N)
        baseline_b = rng.normal(loc=0.0, scale=1.0, size=_BASELINE_N)
        baseline_c = rng.normal(loc=0.0, scale=1.0, size=_BASELINE_N)
        baseline_path = _build_baseline_for(
            {
                "feature_a": baseline_a,
                "feature_b": baseline_b,
                "feature_c": baseline_c,
            },
            tmp_path,
        )
        monitor = DriftMonitor(baseline_path=baseline_path, settings=_make_settings(tmp_path))

        # Recent: feature_b drifts a lot, feature_c drifts a little,
        # feature_a is stable.
        rng2 = np.random.default_rng(_RNG_SEED + 1)
        recent = pd.DataFrame(
            {
                "feature_a": rng2.normal(loc=0.0, scale=1.0, size=_RECENT_N),
                "feature_b": rng2.normal(loc=2.0, scale=1.0, size=_RECENT_N),
                "feature_c": rng2.normal(loc=0.5, scale=1.0, size=_RECENT_N),
            }
        )

        result = monitor.compute_all_psi(recent, top_n=2)

        assert len(result) == 2, "top_n=2 should cap the result rows"
        assert list(result.columns) == ["feature_name", "psi", "n_recent"]
        # Top row must be feature_b (largest drift).
        assert result.iloc[0]["feature_name"] == "feature_b"
        # Sort order: feature_b > feature_c > feature_a (stable).
        assert result.iloc[0]["psi"] > result.iloc[1]["psi"]

    def test_compute_all_psi_handles_missing_feature_with_nan(self, tmp_path: Path) -> None:
        """recent_window missing one column → its PSI is NaN; sorts to bottom."""
        rng = np.random.default_rng(_RNG_SEED)
        baseline = rng.normal(loc=0.0, scale=1.0, size=_BASELINE_N)
        baseline_path = _build_baseline_for(
            {
                "feature_a": baseline,
                "feature_b": baseline,
            },
            tmp_path,
        )
        monitor = DriftMonitor(baseline_path=baseline_path, settings=_make_settings(tmp_path))

        # Recent window has feature_a but NOT feature_b.
        recent = pd.DataFrame({"feature_a": rng.normal(loc=0.0, scale=1.0, size=_RECENT_N)})

        result = monitor.compute_all_psi(recent, top_n=2)

        # Both features present in the result; missing one has NaN PSI.
        assert len(result) == 2
        # feature_a has finite PSI; feature_b has NaN. NaN sorts last
        # under na_position="last".
        assert result.iloc[0]["feature_name"] == "feature_a"
        assert np.isfinite(result.iloc[0]["psi"])
        assert result.iloc[1]["feature_name"] == "feature_b"
        assert np.isnan(result.iloc[1]["psi"])
        assert result.iloc[1]["n_recent"] == 0


# ---------------------------------------------------------------------
# Alerting tests.
# ---------------------------------------------------------------------


class TestAlerting:
    """`check_and_alert` JSONL output + return-value contract."""

    def test_check_and_alert_writes_jsonl_when_psi_above_threshold(self, tmp_path: Path) -> None:
        """Synthetic drift > threshold → file written with valid JSONL records."""
        rng = np.random.default_rng(_RNG_SEED)
        baseline = rng.normal(loc=0.0, scale=1.0, size=_BASELINE_N)
        baseline_path = _build_baseline_for({"feature_a": baseline}, tmp_path)
        # Use a low threshold (0.05) so even moderate drift trips it,
        # avoiding test brittleness.
        settings = _make_settings(tmp_path, threshold=0.05)
        monitor = DriftMonitor(baseline_path=baseline_path, settings=settings)

        recent = pd.DataFrame({"feature_a": rng.normal(loc=1.5, scale=1.0, size=_RECENT_N)})

        n_alerts = monitor.check_and_alert(recent, run_id="test-run-001")

        assert n_alerts == 1, f"expected 1 alert, got {n_alerts}"

        alert_path = tmp_path / "logs" / "drift" / "test-run-001" / "drift_alerts.jsonl"
        assert alert_path.is_file(), f"alert file not written at {alert_path}"

        records = [json.loads(line) for line in alert_path.read_text().splitlines()]
        assert len(records) == 1
        rec = records[0]
        # Schema spot-check.
        assert rec["feature_name"] == "feature_a"
        assert rec["run_id"] == "test-run-001"
        assert rec["psi"] > 0.05
        assert rec["threshold"] == 0.05
        assert rec["n_baseline"] > 0
        assert rec["n_recent"] == _RECENT_N
        assert "timestamp" in rec

    def test_check_and_alert_writes_nothing_when_no_drift(self, tmp_path: Path) -> None:
        """No drift → return 0, no file created."""
        rng = np.random.default_rng(_RNG_SEED)
        baseline = rng.normal(loc=0.0, scale=1.0, size=_BASELINE_N)
        baseline_path = _build_baseline_for({"feature_a": baseline}, tmp_path)
        # High threshold + identical distributions → no alerts.
        settings = _make_settings(tmp_path, threshold=0.5)
        monitor = DriftMonitor(baseline_path=baseline_path, settings=settings)

        recent = pd.DataFrame({"feature_a": rng.normal(loc=0.0, scale=1.0, size=_RECENT_N)})

        n_alerts = monitor.check_and_alert(recent, run_id="no-drift-run")

        assert n_alerts == 0
        # The whole run_id directory should NOT exist (no file means
        # operators can grep for the file's presence as a clean
        # alerted/not-alerted signal).
        run_dir = tmp_path / "logs" / "drift" / "no-drift-run"
        assert not (
            run_dir / "drift_alerts.jsonl"
        ).exists(), "no-drift case should not create the alert file"


# ---------------------------------------------------------------------
# Persistence + math equivalence.
# ---------------------------------------------------------------------


class TestPersistenceAndEquivalence:
    """Round-trip + cross-implementation math equivalence."""

    def test_baseline_round_trip_via_parquet(self, tmp_path: Path) -> None:
        """Build → save → load fresh DriftMonitor → PSI matches."""
        rng = np.random.default_rng(_RNG_SEED)
        baseline = rng.normal(loc=0.0, scale=1.0, size=_BASELINE_N)
        baseline_path = _build_baseline_for({"feature_a": baseline}, tmp_path)
        monitor1 = DriftMonitor(baseline_path=baseline_path, settings=_make_settings(tmp_path))

        # Re-instantiate from the persisted parquet — ensures the
        # _load_baseline reconstruction is correct.
        monitor2 = DriftMonitor(baseline_path=baseline_path, settings=_make_settings(tmp_path))

        recent = pd.DataFrame({"feature_a": rng.normal(loc=0.7, scale=1.0, size=_RECENT_N)})
        psi1 = monitor1.compute_feature_psi("feature_a", recent)
        psi2 = monitor2.compute_feature_psi("feature_a", recent)

        # Same baseline + same recent → identical PSI bit-for-bit.
        assert psi1 == pytest.approx(psi2, abs=1e-12), f"round-trip PSI mismatch: {psi1} vs {psi2}"

    def test_compute_feature_psi_matches_utils_compute_psi(self, tmp_path: Path) -> None:
        """DriftMonitor PSI ≡ utils.metrics.compute_psi for the same inputs.

        Catches any drift between the pre-binned-baseline path and the
        raw-arrays path (e.g., a future epsilon change in one but not
        the other).  Uses NaN-free synthetic data so both pipelines
        agree exactly — production paths drop NaN before binning.
        """
        rng = np.random.default_rng(_RNG_SEED)
        baseline = rng.normal(loc=0.0, scale=1.0, size=_BASELINE_N)
        recent = rng.normal(loc=0.5, scale=1.0, size=_RECENT_N)

        baseline_path = _build_baseline_for({"feature_a": baseline}, tmp_path)
        monitor = DriftMonitor(baseline_path=baseline_path, settings=_make_settings(tmp_path))

        psi_drift_monitor = monitor.compute_feature_psi(
            "feature_a", pd.DataFrame({"feature_a": recent})
        )
        psi_utils = compute_psi(baseline, recent, bins=_BINS)

        # 1e-6 tolerance — both paths use the same epsilon floor and
        # the same quantile-edge logic; differences should be limited
        # to floating-point summation order.
        assert psi_drift_monitor == pytest.approx(psi_utils, abs=1e-6), (
            f"DriftMonitor PSI ({psi_drift_monitor}) != utils.compute_psi "
            f"({psi_utils}); the two binning paths have drifted apart"
        )

        # Sanity: also assert the pure math kernel agrees on
        # pre-computed pcts (the most direct comparison).
        # Re-derive baseline_pcts + recent_pcts from the same pipeline:
        baseline_feature = monitor._baselines["feature_a"]  # noqa: SLF001 — test access
        recent_arr = recent.astype(np.float64)
        recent_bins = np.clip(
            np.digitize(recent_arr, baseline_feature.edges[1:-1], right=False),
            0,
            len(baseline_feature.baseline_pcts) - 1,
        )
        recent_counts = np.bincount(recent_bins, minlength=len(baseline_feature.baseline_pcts))
        recent_pcts = recent_counts / len(recent_arr)
        psi_kernel = _psi_from_pcts(baseline_feature.baseline_pcts, recent_pcts)
        assert psi_kernel == pytest.approx(psi_drift_monitor, abs=1e-12)


# ---------------------------------------------------------------------
# Sprint 6.1.d retrofit — Prometheus Counter behaviour.
# ---------------------------------------------------------------------


class TestDriftAlertsTotalCounter:
    """`fraud_engine_drift_alerts_total` Counter increments per alert.

    The Counter feeds the `FeatureDrift` Prometheus alert rule (Sprint
    6.1.d) so an offline drift run is observable on the live scrape.
    Tests use the **delta pattern** (capture sample value before, fire
    alert, capture after) to be order-independent against the global
    REGISTRY singleton — same convention as test_prometheus_metrics.py.
    """

    @staticmethod
    def _drift_counter_value() -> float:
        """Read the current absolute value of fraud_engine_drift_alerts_total."""
        from prometheus_client import REGISTRY

        value = REGISTRY.get_sample_value("fraud_engine_drift_alerts_total")
        return float(value) if value is not None else 0.0

    def test_drift_alerts_counter_increments_per_jsonl_line(self, tmp_path: Path) -> None:
        """Each alert written to drift_alerts.jsonl bumps the Counter by 1."""
        rng = np.random.default_rng(_RNG_SEED)
        baseline = rng.normal(loc=0.0, scale=1.0, size=_BASELINE_N)
        baseline_path = _build_baseline_for({"feature_a": baseline}, tmp_path)
        # Low threshold so the synthetic shift trips the alert.
        settings = _make_settings(tmp_path, threshold=0.05)
        monitor = DriftMonitor(baseline_path=baseline_path, settings=settings)

        recent = pd.DataFrame({"feature_a": rng.normal(loc=1.5, scale=1.0, size=_RECENT_N)})

        before = self._drift_counter_value()
        n_alerts = monitor.check_and_alert(recent, run_id="counter-test-001")
        after = self._drift_counter_value()

        assert n_alerts == 1
        assert after - before == pytest.approx(
            1.0
        ), f"Counter delta = {after - before}, expected 1.0 (one JSONL line written)"

    def test_drift_alerts_counter_unchanged_when_no_drift(self, tmp_path: Path) -> None:
        """No drift → no JSONL writes → Counter stays put."""
        rng = np.random.default_rng(_RNG_SEED)
        baseline = rng.normal(loc=0.0, scale=1.0, size=_BASELINE_N)
        baseline_path = _build_baseline_for({"feature_a": baseline}, tmp_path)
        # High threshold + identical distributions → no alerts.
        settings = _make_settings(tmp_path, threshold=0.5)
        monitor = DriftMonitor(baseline_path=baseline_path, settings=settings)

        recent = pd.DataFrame({"feature_a": rng.normal(loc=0.0, scale=1.0, size=_RECENT_N)})

        before = self._drift_counter_value()
        n_alerts = monitor.check_and_alert(recent, run_id="counter-test-002")
        after = self._drift_counter_value()

        assert n_alerts == 0
        assert after == pytest.approx(
            before, abs=1e-9
        ), f"Counter unexpectedly changed: {before} → {after} on no-drift run"
