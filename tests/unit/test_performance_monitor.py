"""Unit tests for the Sprint 6.1.c PerformanceMonitor.

Nine tests across five categories:

    Baseline comparison (3):
        - Alert fires when AUC degrades above the 5% threshold.
        - No alert fires when AUC degrades below the 5% threshold.
        - Each metric (AUC / AUC-PR / cost) alerts independently.

    Cost direction (1):
        - Cost uses (current - baseline) / baseline (lower-is-better
          sign convention) so a higher current cost triggers an alert.

    Contract (2):
        - When recent_window has a `decision` column, it's used directly
          for cost (no internal thresholding).
        - When recent_window lacks a `decision` column, scores are
          thresholded internally via Settings.decision_threshold.

    Edge cases (1):
        - Single-class window → AUC + AUC-PR are NaN; no false alerts.

    Alerting (2):
        - check_and_alert writes one JSONL record per degraded metric.
        - check_and_alert writes nothing when there's no degradation.

All tests use deterministic seeds (`np.random.default_rng(42)`); tests
write to `tmp_path` so they don't touch real `logs/performance/`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_engine.config.settings import Settings
from fraud_engine.monitoring.performance_monitor import PerformanceMonitor

# ---------------------------------------------------------------------
# Constants + helpers.
# ---------------------------------------------------------------------

_RNG_SEED = 42
_WINDOW_N = 1_000


def _make_settings(  # noqa: PLR0913 — test factory with one knob per Settings field; collapsing into a dict would obscure the test-side intent at every call site.
    tmp_path: Path,
    *,
    threshold: float = 0.05,
    baseline_auc: float = 0.85,
    baseline_auc_pr: float = 0.50,
    baseline_cost: float = 50_000.0,
    decision_threshold: float = 0.5,
) -> Settings:
    """Build a Settings instance with paths pointing into tmp_path.

    Avoids the lru-cached singleton — each test owns its own Settings
    so paths and baselines are isolated per-test.
    """
    return Settings(
        performance_alert_log_dir=tmp_path / "logs" / "performance",
        performance_degradation_threshold=threshold,
        performance_training_auc=baseline_auc,
        performance_training_auc_pr=baseline_auc_pr,
        performance_training_cost=baseline_cost,
        decision_threshold=decision_threshold,
    )


def _synth_window(  # noqa: PLR0913 — test factory with one knob per synthesis dimension; tests need to dial each independently.
    *,
    n: int = _WINDOW_N,
    fraud_rate: float = 0.035,
    score_separation: float = 2.0,
    rng_seed: int = _RNG_SEED,
    include_decision: bool = False,
    decision_threshold: float = 0.5,
) -> pd.DataFrame:
    """Build a synthetic (score, label) window with controlled separation.

    Higher `score_separation` → cleaner discrimination → higher AUC.
    Lower → noisier → lower AUC.  Used to dial the synthetic-degradation
    bands the tests expect.

    Args:
        n: Number of rows.
        fraud_rate: Fraction of label==1.  Default matches CLAUDE.md §1
            (3.5% fraud rate in IEEE-CIS).
        score_separation: Logistic shift between the legit and fraud
            score-distribution means (in σ units).  3.0+ → AUC ≈ 0.95;
            2.0 → AUC ≈ 0.85; 1.0 → AUC ≈ 0.75; 0.5 → AUC ≈ 0.65.
        rng_seed: For reproducibility.
        include_decision: If True, threshold scores at `decision_threshold`
            and add a `decision` column.
        decision_threshold: Threshold to apply if include_decision=True.
    """
    rng = np.random.default_rng(rng_seed)
    n_fraud = max(int(round(n * fraud_rate)), 2)
    n_legit = n - n_fraud
    legit_scores = 1.0 / (1.0 + np.exp(-rng.normal(loc=-score_separation, scale=1.0, size=n_legit)))
    fraud_scores = 1.0 / (1.0 + np.exp(-rng.normal(loc=score_separation, scale=1.0, size=n_fraud)))
    scores = np.concatenate([legit_scores, fraud_scores])
    labels = np.concatenate([np.zeros(n_legit, dtype=np.int64), np.ones(n_fraud, dtype=np.int64)])
    # Shuffle so the order doesn't tip off any downstream code.
    perm = rng.permutation(n)
    scores = scores[perm]
    labels = labels[perm]
    df = pd.DataFrame({"score": scores, "label": labels})
    if include_decision:
        df["decision"] = (scores >= decision_threshold).astype(np.int64)
    return df


# ---------------------------------------------------------------------
# Baseline comparison tests.
# ---------------------------------------------------------------------


class TestBaselineComparison:
    """Synthetic degradation triggers (or doesn't trigger) the alert."""

    def test_alert_when_auc_degrades_above_threshold(self, tmp_path: Path) -> None:
        """Recent AUC well below baseline → alert.

        sep=0.5 produces AUC ≈ 0.813 with the deterministic seed; setting
        baseline=0.95 gives degradation = (0.95 - 0.813) / 0.95 ≈ 14.4%,
        comfortably above the 5% threshold.
        """
        recent = _synth_window(score_separation=0.5)
        # Set AUC-PR + cost baselines such that they DON'T degrade so we
        # isolate the AUC alert. The probe pattern (run once, set to actual)
        # avoids brittle hard-coded numbers.
        probe_settings = _make_settings(tmp_path)
        probe_monitor = PerformanceMonitor(settings=probe_settings)
        actual = probe_monitor.compute_rolling_metrics(recent)

        settings = _make_settings(
            tmp_path,
            baseline_auc=0.95,  # > current ≈ 0.81 → degrade
            baseline_auc_pr=actual["auc_pr"],  # match → no degrade
            baseline_cost=actual["cost"] * 100.0,  # current much cheaper → no degrade
        )
        monitor = PerformanceMonitor(settings=settings)
        metrics = monitor.compute_rolling_metrics(recent)
        degradations = monitor.compare_to_baseline(metrics)

        # AUC must be in the degradations list.
        assert any(
            d.metric == "auc" for d in degradations
        ), f"expected auc degradation; got {[(d.metric, d.degradation) for d in degradations]}"
        auc_deg = next(d for d in degradations if d.metric == "auc")
        # Sanity: degradation > 5%.
        assert auc_deg.degradation > 0.05

    def test_no_alert_when_auc_degrades_below_threshold(self, tmp_path: Path) -> None:
        """Recent AUC <5% below baseline → no AUC alert.

        sep=0.5 produces AUC ≈ 0.813; setting baseline=0.84 gives
        degradation = (0.84 - 0.813) / 0.84 ≈ 3.2%, just below the 5%
        threshold.
        """
        recent = _synth_window(score_separation=0.5)
        probe_settings = _make_settings(tmp_path)
        probe_monitor = PerformanceMonitor(settings=probe_settings)
        actual = probe_monitor.compute_rolling_metrics(recent)

        # Baseline within the 5% threshold relative to current AUC.
        baseline_just_above = actual["auc"] * 1.03  # ≈ 3% relative degradation
        settings = _make_settings(
            tmp_path,
            baseline_auc=baseline_just_above,
            baseline_auc_pr=actual["auc_pr"],
            baseline_cost=actual["cost"] * 100.0,
        )
        monitor = PerformanceMonitor(settings=settings)

        degradations = monitor.compare_to_baseline(monitor.compute_rolling_metrics(recent))

        assert not any(
            d.metric == "auc" for d in degradations
        ), f"unexpected auc degradation: {[(d.metric, d.degradation) for d in degradations]}"

    def test_each_metric_alerts_independently(self, tmp_path: Path) -> None:
        """Setting only one baseline above current degrades only that metric.

        sep=0.5 → AUC ≈ 0.81, AUC-PR ≈ 0.155.  Setting AUC-PR baseline at
        2× current (≈ 0.31, comfortably under the [0,1] cap) gives ≈50%
        degradation; AUC + cost baselines stay at current → no degrade.
        """
        recent = _synth_window(score_separation=0.5)
        probe_settings = _make_settings(tmp_path)
        probe_monitor = PerformanceMonitor(settings=probe_settings)
        actual = probe_monitor.compute_rolling_metrics(recent)

        settings = _make_settings(
            tmp_path,
            baseline_auc=actual["auc"],  # match → no degrade
            baseline_auc_pr=min(actual["auc_pr"] * 2.0, 0.99),  # ~50% degrade
            baseline_cost=actual["cost"] * 100.0,  # current cheaper → no degrade
        )
        monitor = PerformanceMonitor(settings=settings)
        degradations = monitor.compare_to_baseline(monitor.compute_rolling_metrics(recent))

        degraded_metrics = {d.metric for d in degradations}
        assert degraded_metrics == {
            "auc_pr"
        }, f"expected only auc_pr to degrade; got {degraded_metrics}"


# ---------------------------------------------------------------------
# Cost-direction test.
# ---------------------------------------------------------------------


class TestCostDirection:
    """Cost uses (current - baseline) / baseline (lower-is-better)."""

    def test_cost_degradation_uses_higher_is_worse_sign(self, tmp_path: Path) -> None:
        """current_cost > baseline_cost → degradation > 0 → alert."""
        recent = _synth_window(score_separation=2.0)
        # Set cost baseline well below the synthesized current so the
        # ratio degrades clearly. Set AUC + AUC-PR baselines to current
        # (no degradation) so we isolate the cost path.
        probe_settings = _make_settings(tmp_path)
        probe_monitor = PerformanceMonitor(settings=probe_settings)
        actual = probe_monitor.compute_rolling_metrics(recent)

        # Cost baseline = half of current → (current - baseline) / baseline = 100%.
        settings = _make_settings(
            tmp_path,
            baseline_auc=actual["auc"],
            baseline_auc_pr=actual["auc_pr"],
            baseline_cost=actual["cost"] / 2.0,  # half → 100% degradation
        )
        monitor = PerformanceMonitor(settings=settings)
        degradations = monitor.compare_to_baseline(monitor.compute_rolling_metrics(recent))

        assert len(degradations) == 1
        cost_deg = degradations[0]
        assert cost_deg.metric == "cost"
        # current is 2x baseline → degradation = 1.0 (100%).
        assert cost_deg.degradation == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------
# Contract tests.
# ---------------------------------------------------------------------


class TestRecentWindowContract:
    """The recent_window column contract."""

    def test_compute_rolling_metrics_uses_decision_column_when_present(
        self, tmp_path: Path
    ) -> None:
        """A caller-provided `decision` column overrides internal thresholding.

        We synthesise a window where the score-derived decision (at
        threshold 0.5) and a deliberately-different `decision` column
        (everything blocked) produce different cost outcomes; assert
        the result reflects the column, not the threshold.
        """
        recent = _synth_window(score_separation=2.0, include_decision=False)
        # Override decision column: block EVERYTHING (decision=1 for all rows).
        # That maximises FP cost (all legit transactions blocked).
        recent_block_all = recent.copy()
        recent_block_all["decision"] = 1

        # Same window but with decision derived from threshold 0.5.
        recent_no_decision = recent.copy()

        settings = _make_settings(tmp_path, decision_threshold=0.5)
        monitor = PerformanceMonitor(settings=settings)

        cost_block_all = monitor.compute_rolling_metrics(recent_block_all)["cost"]
        cost_internal = monitor.compute_rolling_metrics(recent_no_decision)["cost"]
        # block-all should produce strictly higher cost (more FPs).
        assert cost_block_all > cost_internal, (
            f"caller-provided decision=1 column should produce higher FP cost; "
            f"got block_all={cost_block_all}, internal={cost_internal}"
        )

    def test_compute_rolling_metrics_thresholds_scores_when_decision_absent(
        self, tmp_path: Path
    ) -> None:
        """Without a `decision` column, scores are thresholded via Settings."""
        recent = _synth_window(score_separation=2.0, include_decision=False)
        # Two monitors with different thresholds → different decisions → different costs.
        settings_low_threshold = _make_settings(tmp_path, decision_threshold=0.1)
        settings_high_threshold = _make_settings(tmp_path, decision_threshold=0.9)

        cost_low = PerformanceMonitor(settings=settings_low_threshold).compute_rolling_metrics(
            recent
        )["cost"]
        cost_high = PerformanceMonitor(settings=settings_high_threshold).compute_rolling_metrics(
            recent
        )["cost"]

        # Different thresholds → different decisions → different costs.
        assert (
            cost_low != cost_high
        ), f"thresholds 0.1 and 0.9 should produce different costs; got both = {cost_low}"


# ---------------------------------------------------------------------
# Edge case test.
# ---------------------------------------------------------------------


class TestEdgeCases:
    """Single-class windows and other math-undefined situations."""

    def test_returns_nan_metrics_when_window_is_single_class(self, tmp_path: Path) -> None:
        """All-label=0 window → AUC + AUC-PR are NaN; no false alerts.

        Cost baseline is set well above current so it doesn't generate a
        spurious cost alert that would mask the AUC/AUC-PR NaN behaviour
        we're actually testing.
        """
        n = 100
        recent = pd.DataFrame(
            {
                "score": np.random.default_rng(_RNG_SEED).uniform(0, 1, n),
                "label": np.zeros(n, dtype=np.int64),
            }
        )
        # Probe to size the cost baseline appropriately.
        probe_settings = _make_settings(tmp_path)
        probe_monitor = PerformanceMonitor(settings=probe_settings)
        actual = probe_monitor.compute_rolling_metrics(recent)
        settings = _make_settings(
            tmp_path,
            baseline_cost=actual["cost"] * 100.0,  # current cheaper → no cost alert
        )
        monitor = PerformanceMonitor(settings=settings)

        metrics = monitor.compute_rolling_metrics(recent)
        # AUC + AUC-PR must be NaN; cost is still defined (just no FNs).
        assert math.isnan(metrics["auc"])
        assert math.isnan(metrics["auc_pr"])
        assert math.isfinite(metrics["cost"])

        # NaN metrics must NOT trigger alerts.
        degradations = monitor.compare_to_baseline(metrics)
        degraded_metric_names = {d.metric for d in degradations}
        assert "auc" not in degraded_metric_names
        assert "auc_pr" not in degraded_metric_names


# ---------------------------------------------------------------------
# Alerting tests.
# ---------------------------------------------------------------------


class TestAlerting:
    """check_and_alert JSONL output + return-value contract."""

    def test_check_and_alert_writes_jsonl_per_degraded_metric(self, tmp_path: Path) -> None:
        """All three metrics degrade → 3 JSONL lines, schema spot-checked.

        sep=0.5 keeps both AUC + AUC-PR well under 1.0, so multiplying
        them by 1.2 for the baseline (clipped at 0.99) doesn't hit the
        Pydantic le=1.0 cap.
        """
        recent = _synth_window(score_separation=0.5)
        probe_settings = _make_settings(tmp_path)
        probe_monitor = PerformanceMonitor(settings=probe_settings)
        actual = probe_monitor.compute_rolling_metrics(recent)

        settings = _make_settings(
            tmp_path,
            baseline_auc=min(actual["auc"] * 1.2, 0.99),  # ~17% degrade
            baseline_auc_pr=min(actual["auc_pr"] * 2.0, 0.99),  # ~50% degrade
            baseline_cost=actual["cost"] / 2.0,  # ~100% degrade
        )
        monitor = PerformanceMonitor(settings=settings)
        n_alerts = monitor.check_and_alert(recent, run_id="test-run-001")

        assert n_alerts == 3, f"expected 3 alerts; got {n_alerts}"

        alert_path = tmp_path / "logs" / "performance" / "test-run-001" / "performance_alerts.jsonl"
        assert alert_path.is_file(), f"alert file not written at {alert_path}"

        records = [json.loads(line) for line in alert_path.read_text().splitlines()]
        assert len(records) == 3
        # Schema spot-check on the first record.
        rec = records[0]
        for key in (
            "timestamp",
            "run_id",
            "metric",
            "baseline",
            "current",
            "degradation",
            "threshold",
            "n_recent",
        ):
            assert key in rec, f"missing key {key!r} in alert record: {rec}"
        # All three metric names are present across the 3 records.
        metric_names = {r["metric"] for r in records}
        assert metric_names == {"auc", "auc_pr", "cost"}
        # All record common values match the run.
        for r in records:
            assert r["run_id"] == "test-run-001"
            assert r["threshold"] == 0.05
            assert r["n_recent"] == _WINDOW_N

    def test_check_and_alert_writes_nothing_when_no_degradation(self, tmp_path: Path) -> None:
        """No metric degrades → return 0, no file created."""
        recent = _synth_window(score_separation=2.0)
        # Set baselines exactly at current values → no degradation
        # (sign-corrected fractional degradation is 0; threshold is 0.05).
        probe_settings = _make_settings(tmp_path)
        probe_monitor = PerformanceMonitor(settings=probe_settings)
        actual = probe_monitor.compute_rolling_metrics(recent)

        settings = _make_settings(
            tmp_path,
            baseline_auc=actual["auc"],
            baseline_auc_pr=actual["auc_pr"],
            baseline_cost=actual["cost"],
        )
        monitor = PerformanceMonitor(settings=settings)
        n_alerts = monitor.check_and_alert(recent, run_id="no-degrade-run")

        assert n_alerts == 0
        run_dir = tmp_path / "logs" / "performance" / "no-degrade-run"
        # File NOT created (signals "all clear" via absence).
        assert not (
            run_dir / "performance_alerts.jsonl"
        ).exists(), "no-degradation case should not create the alert file"
