"""Champion-vs-challenger comparison primitive for shadow-mode analysis.

Sprint 5 prompt 5.2.c: the analytical follow-on to Sprint 5.2.b's
shadow surface. Takes a DataFrame of predictions where each row carries
both the champion (Model A) and challenger (Model B) scores +
decisions, computes the standard champion-vs-challenger metrics
(agreement rate, score correlation, economic cost on labeled subset,
bootstrap significance), and returns a structured `ComparisonReport`
with a clear `PromotionVerdict` against the spec's three promotion
criteria:

    1. cost_improvement > 2%           (challenger cheaper)
    2. p_value < 0.05                  (cost difference statistically significant)
    3. agreement_rate > 0.85           (decisions don't diverge wildly)

ALL three must hold to promote the challenger to production. The
verdict structure carries per-criterion outcomes so the rendered
report explains WHY a promotion is blocked (rather than presenting a
single bool).

Module surface (re-exported from `fraud_engine.evaluation`):
    - ShadowComparison
    - ComparisonReport
    - PromotionVerdict
    - EconomicCosts

Business rationale:
    Shadow mode (Sprint 5.2.b) captures challenger predictions
    alongside production predictions during normal traffic. Without an
    analytical layer to compare them, the data sits in the structlog
    stream untouched — the value of running the challenger evaporates.
    This module is the bridge: it turns raw shadow events into
    actionable promotion decisions backed by statistical evidence.

Trade-offs considered:
    - **DataFrame-input contract.** The module is data-source-agnostic:
      it doesn't load from Postgres, doesn't parse JSONL, doesn't
      know about the production wire format. The CLI script
      (`scripts/shadow_compare_report.py`) handles loading; this
      module handles analysis. Mirrors the project's existing
      separation: `evaluation/economic.py` doesn't load parquets;
      `scripts/run_economic_evaluation.py` does. Keeps tests trivial
      (build small DataFrames inline; assert outputs).

    - **Population-level agreement + correlation; subset-level cost.**
      Agreement and correlation don't need labels — they're properties
      of the two scoring distributions, computable over the full N.
      Cost requires labels (production chargebacks; or backtest labels
      from `tier5_test.parquet`). Splitting the metrics this way means
      the report is informative even when the labeled subset is small
      (~5% of predictions, typical real-world chargeback rate).

    - **In-module bootstrap, NOT scipy.stats.bootstrap.** The standard
      non-parametric bootstrap (resample indices with replacement,
      compute statistic per iteration, report mean + CI + two-sided
      p-value) is ~15 LOC. Inlining keeps the logic visible and the
      dependency surface minimal. Parametric tests (t-test, Wilcoxon)
      were rejected — fraud-cost data violates their distributional
      assumptions (heavy-tailed, with mass at 0 for true negatives).
      Determinism via fixed seed lets tests assert exact p-values.

    - **`EconomicCosts` dataclass mirroring Settings's cost fields.**
      A small frozen dataclass (fraud_cost, fp_cost, tp_cost) — no
      tn_cost since true negatives are always free in this project's
      cost model (matches Sprint 4.1's `EconomicCostModel` convention).
      Keeps this module Settings-free for testability; the CLI script
      constructs the `EconomicCosts` from `Settings`.

    - **`PromotionVerdict` carries per-criterion outcomes, not just
      a bool.** Each of the three criteria has its own pass/fail flag
      + a human-readable reason string. The `summary` property renders
      a compact line for the markdown report. An analyst can act on
      "FAIL: agreement_rate=0.78 below 0.85 threshold" instead of
      staring at `promote=False`.

    - **`is_fraud` column is optional.** When present (labeled subset),
      cost + bootstrap path runs. When absent, those fields are None
      in the report and the verdict short-circuits to False with a
      "labels required" reason. Lets the report be useful even before
      chargeback labels land (agreement + correlation still compute).

Cross-references:
    - `src/fraud_engine/evaluation/economic.py` — Sprint 4.1's
      `EconomicCostModel` (the cost matrix this module reuses).
    - `src/fraud_engine/api/shadow.py:444-454` — the `shadow.scored`
      structlog event schema (the input columns this module expects).
    - `scripts/create_predictions_table.sql` — Sprint 5.2.a's
      `predictions` Postgres table (the production join target).
    - `scripts/shadow_compare_report.py` — the CLI consumer.
    - `CLAUDE.md` §3 (production-API stack), §8 (cost defaults).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Final

import numpy as np
import pandas as pd

from fraud_engine.utils.logging import get_logger, log_call

# ---------------------------------------------------------------------
# Module constants.
# ---------------------------------------------------------------------

# Spec-mandated promotion thresholds. All three must hold for promotion.
_DEFAULT_COST_IMPROVEMENT_THRESHOLD: Final[float] = 0.02  # 2% relative improvement
_DEFAULT_P_VALUE_THRESHOLD: Final[float] = 0.05
_DEFAULT_AGREEMENT_THRESHOLD: Final[float] = 0.85

# Bootstrap configuration. 10K iterations gives stable p-value estimates
# at the resolution we care about (0.05 ± 0.005). Fixed seed makes tests
# deterministic.
_DEFAULT_BOOTSTRAP_N_ITER: Final[int] = 10_000
_DEFAULT_BOOTSTRAP_SEED: Final[int] = 42

# Confidence-interval percentiles. Standard 95% two-sided.
_CI_LOW_PERCENTILE: Final[float] = 2.5
_CI_HIGH_PERCENTILE: Final[float] = 97.5

# Minimum unique values for a meaningful Pearson correlation. With <2
# unique values either column is constant and correlation is undefined.
_MIN_UNIQUE_FOR_CORR: Final[int] = 2

# Required input DataFrame columns.
_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "request_id",
    "champion_score",
    "shadow_score",
    "champion_decision",
    "shadow_decision",
)
# Optional column — when present, enables the cost + bootstrap path.
_LABEL_COLUMN: Final[str] = "is_fraud"

_logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Public dataclasses.
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class EconomicCosts:
    """Cost constants for champion-vs-challenger cost computation.

    Mirrors Sprint 4.1's `EconomicCostModel(fraud_cost, fp_cost, tp_cost,
    tn_cost=0)`. Kept as a separate dataclass so this module doesn't
    need to construct an EconomicCostModel instance (which carries
    additional state we don't use for the bootstrap path).

    Attributes:
        fraud_cost: USD cost of a missed fraud (false negative).
            Default project value: $450 (per CLAUDE.md §8).
        fp_cost: USD cost of a false-positive block (customer friction
            + churn risk). Default: $35.
        tp_cost: USD cost of a true-positive (analyst investigation
            time). Default: $5.
    """

    fraud_cost: float
    fp_cost: float
    tp_cost: float


@dataclasses.dataclass(frozen=True, slots=True)
class PromotionVerdict:
    """Per-criterion + aggregate verdict on whether to promote the challenger.

    All three criteria must pass for `promote=True`. The reason
    strings explain each criterion's outcome for the markdown report.

    Attributes:
        promote: True iff ALL three criteria pass.
        cost_improvement_pass: cost_improvement > threshold (default 2%).
        p_value_pass: p_value < threshold (default 0.05).
        agreement_pass: agreement_rate > threshold (default 0.85).
        reasons: ordered list of three human-readable reason strings,
            one per criterion. Each starts with "PASS:" or "FAIL:".
    """

    promote: bool
    cost_improvement_pass: bool
    p_value_pass: bool
    agreement_pass: bool
    reasons: list[str]

    @property
    def summary(self) -> str:
        """Compact one-line verdict for the report header."""
        prefix = "PROMOTE" if self.promote else "DO NOT PROMOTE"
        return f"{prefix} — {' | '.join(self.reasons)}"


@dataclasses.dataclass(frozen=True, slots=True)
class ComparisonReport:
    """Structured output of `ShadowComparison.run()`.

    Cost and bootstrap fields are None when the input DataFrame lacks
    the `is_fraud` column. The verdict's `promote` flag is False in
    that case (with a "labels required" reason).

    Attributes:
        n_total: Total predictions analysed.
        n_labeled: Predictions with non-null `is_fraud` (None if no
            label column at all).
        agreement_rate: Fraction where champion_decision == shadow_decision.
        score_correlation: Pearson correlation of champion_score vs
            shadow_score. NaN if either is constant (degenerate input).
        champion_cost_per_txn: Mean cost per labeled prediction under
            champion's decision. None if no labels.
        shadow_cost_per_txn: Mean cost per labeled prediction under
            shadow's decision. None if no labels.
        cost_improvement: (champion - shadow) / champion. Positive means
            shadow is cheaper. None if no labels OR champion cost is 0.
        bootstrap_mean_diff: Mean of the bootstrap distribution of
            (champion_cost - shadow_cost). None if no labels.
        bootstrap_ci_95: 95% CI on the bootstrap mean difference.
            None if no labels.
        bootstrap_p_value: Two-sided p-value testing whether the cost
            difference is consistent with zero. None if no labels.
        verdict: Per-criterion + aggregate promotion decision.
    """

    n_total: int
    n_labeled: int | None
    agreement_rate: float
    score_correlation: float
    champion_cost_per_txn: float | None
    shadow_cost_per_txn: float | None
    cost_improvement: float | None
    bootstrap_mean_diff: float | None
    bootstrap_ci_95: tuple[float, float] | None
    bootstrap_p_value: float | None
    verdict: PromotionVerdict


# ---------------------------------------------------------------------
# Module-private helpers.
# ---------------------------------------------------------------------


def _per_row_costs(
    decisions: pd.Series[Any],
    labels: pd.Series[Any],
    costs: EconomicCosts,
) -> np.ndarray[Any, Any]:
    """Return the per-row USD cost array given decisions + labels + costs.

    Cost matrix:
        decision="block", label=fraud (1)     → tp_cost (caught + investigated)
        decision="block", label=not_fraud (0) → fp_cost (customer friction)
        decision="allow", label=fraud (1)     → fraud_cost (full loss)
        decision="allow", label=not_fraud (0) → 0 (correct allow; free)

    Mirrors Sprint 4.1's `EconomicCostModel.compute_cost` matrix
    exactly. Returns per-row costs (not aggregated) so the caller can
    bootstrap.

    Args:
        decisions: Series of 'block'/'allow' strings, length N.
        labels: Series of 0/1 binary labels, length N. Must align with
            decisions index-wise (caller's responsibility).
        costs: EconomicCosts dataclass with fraud / fp / tp values.

    Returns:
        np.ndarray of shape (N,) with per-row USD cost.
    """
    decisions_arr = decisions.to_numpy()
    labels_arr = labels.to_numpy().astype(int)
    is_block = decisions_arr == "block"
    is_fraud = labels_arr == 1

    cost = np.zeros(len(decisions_arr), dtype=np.float64)
    # Block + fraud → tp; block + not_fraud → fp; allow + fraud → fraud_cost; allow + not_fraud → 0
    cost[is_block & is_fraud] = costs.tp_cost
    cost[is_block & ~is_fraud] = costs.fp_cost
    cost[~is_block & is_fraud] = costs.fraud_cost
    return cost


def _bootstrap_cost_diff(
    champion_costs: np.ndarray[Any, Any],
    shadow_costs: np.ndarray[Any, Any],
    n_iter: int = _DEFAULT_BOOTSTRAP_N_ITER,
    seed: int = _DEFAULT_BOOTSTRAP_SEED,
) -> tuple[float, tuple[float, float], float]:
    """Non-parametric bootstrap on the (champion - shadow) cost difference.

    Resamples indices with replacement N times; for each iteration
    computes mean(champion_costs[idx]) - mean(shadow_costs[idx]).
    Returns the mean of the bootstrap distribution, a 95% CI, and
    a two-sided p-value testing whether the difference is consistent
    with zero.

    Diff convention: positive means shadow is CHEAPER (champion - shadow > 0).

    Args:
        champion_costs: Per-row champion costs (length N_labeled).
        shadow_costs: Per-row shadow costs (length N_labeled). Must
            align with champion_costs index-wise.
        n_iter: Bootstrap iterations. Default 10K; tested deterministic.
        seed: RNG seed. Default 42; tests assert exact p-values via
            this seed.

    Returns:
        Tuple of (mean_diff, (ci_low, ci_high), p_value).

    Raises:
        ValueError: If the cost arrays have different lengths or are empty.
    """
    if len(champion_costs) != len(shadow_costs):
        raise ValueError(
            f"_bootstrap_cost_diff: arrays must have equal length; "
            f"got champion={len(champion_costs)}, shadow={len(shadow_costs)}"
        )
    if len(champion_costs) == 0:
        raise ValueError("_bootstrap_cost_diff: cost arrays are empty")

    rng = np.random.default_rng(seed)
    n = len(champion_costs)
    diffs = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        idx = rng.integers(0, n, n)
        diffs[i] = champion_costs[idx].mean() - shadow_costs[idx].mean()

    ci_low = float(np.percentile(diffs, _CI_LOW_PERCENTILE))
    ci_high = float(np.percentile(diffs, _CI_HIGH_PERCENTILE))
    # Two-sided p-value: probability the observed mean diff is
    # consistent with zero under the null hypothesis.
    p_value = 2.0 * float(min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return float(diffs.mean()), (ci_low, ci_high), p_value


def _validate_input(predictions: pd.DataFrame) -> None:
    """Raise ValueError if required columns are missing.

    Required columns: request_id, champion_score, shadow_score,
    champion_decision, shadow_decision. Optional: is_fraud.
    """
    missing = [col for col in _REQUIRED_COLUMNS if col not in predictions.columns]
    if missing:
        raise ValueError(
            f"ShadowComparison: predictions DataFrame missing required "
            f"columns: {missing}. Required: {list(_REQUIRED_COLUMNS)}; "
            f"got: {list(predictions.columns)}"
        )


# ---------------------------------------------------------------------
# Public class.
# ---------------------------------------------------------------------


class ShadowComparison:
    """Champion-vs-challenger offline comparison.

    Public API:
        - `run()` — compute all metrics + verdict; return `ComparisonReport`.

    Inputs (passed at construction):
        - `predictions`: pandas DataFrame with required columns
          (request_id, champion_score, shadow_score, champion_decision,
          shadow_decision) + optional `is_fraud` column.
        - `costs`: EconomicCosts dataclass (fraud / fp / tp).
        - Threshold knobs (kw-only): cost_improvement_threshold (default
          2%), p_value_threshold (default 0.05), agreement_threshold
          (default 85%), bootstrap_n_iter (default 10K), seed (default 42).

    Business rationale + trade-offs considered: see module docstring.
    """

    def __init__(  # noqa: PLR0913 — knobs map 1:1 to the spec's three promotion thresholds + bootstrap config; folding into a config dataclass would obscure the per-knob defaults
        self,
        predictions: pd.DataFrame,
        costs: EconomicCosts,
        *,
        cost_improvement_threshold: float = _DEFAULT_COST_IMPROVEMENT_THRESHOLD,
        p_value_threshold: float = _DEFAULT_P_VALUE_THRESHOLD,
        agreement_threshold: float = _DEFAULT_AGREEMENT_THRESHOLD,
        bootstrap_n_iter: int = _DEFAULT_BOOTSTRAP_N_ITER,
        seed: int = _DEFAULT_BOOTSTRAP_SEED,
    ) -> None:
        """Configure the comparison; does NO computation until `run()`.

        Args:
            predictions: DataFrame with required + optional columns.
            costs: Per-outcome cost constants.
            cost_improvement_threshold: Minimum relative improvement
                for promotion (default 0.02 = 2%).
            p_value_threshold: Maximum p-value for promotion (default 0.05).
            agreement_threshold: Minimum agreement rate for promotion
                (default 0.85).
            bootstrap_n_iter: Bootstrap iterations (default 10K).
            seed: RNG seed for reproducibility (default 42).

        Raises:
            ValueError: If `predictions` is missing required columns.
        """
        _validate_input(predictions)
        self._predictions: pd.DataFrame = predictions.copy()
        self._costs: EconomicCosts = costs
        self._cost_improvement_threshold: float = cost_improvement_threshold
        self._p_value_threshold: float = p_value_threshold
        self._agreement_threshold: float = agreement_threshold
        self._bootstrap_n_iter: int = bootstrap_n_iter
        self._seed: int = seed

    @log_call
    def run(self) -> ComparisonReport:
        """Compute all metrics + the promotion verdict.

        Always-computable (no labels needed):
            - n_total, agreement_rate, score_correlation

        Labels-required (only if `is_fraud` column present):
            - n_labeled, champion_cost_per_txn, shadow_cost_per_txn,
              cost_improvement, bootstrap_mean_diff, bootstrap_ci_95,
              bootstrap_p_value

        The verdict short-circuits to `promote=False` when labels are
        absent (one of the criteria — cost improvement — is undefined).

        Returns:
            Fully-populated `ComparisonReport`.
        """
        n_total = len(self._predictions)
        agreement_rate = self._compute_agreement()
        score_correlation = self._compute_correlation()

        # Cost path — only when labels present.
        has_labels = _LABEL_COLUMN in self._predictions.columns
        if has_labels:
            labeled = self._predictions.dropna(subset=[_LABEL_COLUMN])
            n_labeled: int | None = len(labeled)
            if n_labeled and n_labeled > 0:
                champion_costs = _per_row_costs(
                    labeled["champion_decision"], labeled[_LABEL_COLUMN], self._costs
                )
                shadow_costs = _per_row_costs(
                    labeled["shadow_decision"], labeled[_LABEL_COLUMN], self._costs
                )
                champ_mean = float(champion_costs.mean())
                shadow_mean = float(shadow_costs.mean())
                champion_cost_per_txn: float | None = champ_mean
                shadow_cost_per_txn: float | None = shadow_mean
                # Avoid division-by-zero when champion has zero cost
                # (only if all labeled predictions are correct
                # allow-non-fraud — possible for tiny test inputs).
                cost_improvement: float | None
                if champ_mean > 0:
                    cost_improvement = (champ_mean - shadow_mean) / champ_mean
                else:
                    cost_improvement = None
                mean_diff, ci, p_value = _bootstrap_cost_diff(
                    champion_costs, shadow_costs, self._bootstrap_n_iter, self._seed
                )
                bootstrap_mean_diff: float | None = mean_diff
                bootstrap_ci_95: tuple[float, float] | None = ci
                bootstrap_p_value: float | None = p_value
            else:
                champion_cost_per_txn = None
                shadow_cost_per_txn = None
                cost_improvement = None
                bootstrap_mean_diff = None
                bootstrap_ci_95 = None
                bootstrap_p_value = None
        else:
            n_labeled = None
            champion_cost_per_txn = None
            shadow_cost_per_txn = None
            cost_improvement = None
            bootstrap_mean_diff = None
            bootstrap_ci_95 = None
            bootstrap_p_value = None

        verdict = self._verdict(
            agreement_rate=agreement_rate,
            cost_improvement=cost_improvement,
            p_value=bootstrap_p_value,
        )

        return ComparisonReport(
            n_total=n_total,
            n_labeled=n_labeled,
            agreement_rate=agreement_rate,
            score_correlation=score_correlation,
            champion_cost_per_txn=champion_cost_per_txn,
            shadow_cost_per_txn=shadow_cost_per_txn,
            cost_improvement=cost_improvement,
            bootstrap_mean_diff=bootstrap_mean_diff,
            bootstrap_ci_95=bootstrap_ci_95,
            bootstrap_p_value=bootstrap_p_value,
            verdict=verdict,
        )

    # ---------- internal computations ----------------------------------

    def _compute_agreement(self) -> float:
        """Fraction of predictions where champion and shadow decide the same."""
        match = self._predictions["champion_decision"] == self._predictions["shadow_decision"]
        return float(match.mean())

    def _compute_correlation(self) -> float:
        """Pearson correlation of champion_score vs shadow_score.

        Returns NaN if either column is constant (degenerate input
        for which correlation is undefined; we surface NaN rather
        than raise so the report doesn't crash on edge-case data).
        """
        champion = self._predictions["champion_score"]
        shadow = self._predictions["shadow_score"]
        # Pandas' .corr handles NaN propagation; explicit constant check
        # avoids the warning + ambiguous NaN that would otherwise emerge.
        if (
            champion.nunique(dropna=True) < _MIN_UNIQUE_FOR_CORR
            or shadow.nunique(dropna=True) < _MIN_UNIQUE_FOR_CORR
        ):
            return float("nan")
        return float(champion.corr(shadow))

    def _verdict(
        self,
        *,
        agreement_rate: float,
        cost_improvement: float | None,
        p_value: float | None,
    ) -> PromotionVerdict:
        """Apply the three promotion criteria; return structured verdict."""
        # Criterion 1: agreement (always computable).
        agreement_pass = agreement_rate > self._agreement_threshold
        agreement_reason = (
            f"PASS: agreement_rate={agreement_rate:.4f} > {self._agreement_threshold:.2f}"
            if agreement_pass
            else f"FAIL: agreement_rate={agreement_rate:.4f} not > {self._agreement_threshold:.2f}"
        )

        # Criterion 2: cost improvement (requires labels).
        if cost_improvement is None:
            cost_improvement_pass = False
            cost_improvement_reason = "FAIL: cost_improvement undefined (labels required)"
        else:
            cost_improvement_pass = cost_improvement > self._cost_improvement_threshold
            cost_improvement_reason = (
                f"PASS: cost_improvement={cost_improvement * 100:.2f}% "
                f"> {self._cost_improvement_threshold * 100:.0f}%"
                if cost_improvement_pass
                else f"FAIL: cost_improvement={cost_improvement * 100:.2f}% "
                f"not > {self._cost_improvement_threshold * 100:.0f}%"
            )

        # Criterion 3: p-value (requires labels).
        if p_value is None:
            p_value_pass = False
            p_value_reason = "FAIL: p_value undefined (labels required)"
        else:
            p_value_pass = p_value < self._p_value_threshold
            p_value_reason = (
                f"PASS: p_value={p_value:.4f} < {self._p_value_threshold:.2f}"
                if p_value_pass
                else f"FAIL: p_value={p_value:.4f} not < {self._p_value_threshold:.2f}"
            )

        promote = agreement_pass and cost_improvement_pass and p_value_pass
        return PromotionVerdict(
            promote=promote,
            cost_improvement_pass=cost_improvement_pass,
            p_value_pass=p_value_pass,
            agreement_pass=agreement_pass,
            reasons=[agreement_reason, cost_improvement_reason, p_value_reason],
        )


__all__ = [
    "ComparisonReport",
    "EconomicCosts",
    "PromotionVerdict",
    "ShadowComparison",
]
