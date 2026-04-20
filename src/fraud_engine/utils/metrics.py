"""Shared metric implementations: economic cost, PR@K, recall@FPR, PSI.

Every metric in this module is called from both evaluation (Sprint 4
threshold optimisation) and monitoring (Sprint 6 drift dashboards).
Defining them here means the two pipelines can never drift apart on
metric definition — a change to `economic_cost` ripples into both
places on the same deploy.

Business rationale:
    Fraud-ML evaluation is not about AUC. A 0.95-AUC model that trips
    at the wrong threshold can still bankrupt a team via
    false-positive chargeback costs, while a 0.90-AUC model tuned on
    real cost ratios ships profitably. These four metrics are the
    ones that matter operationally:
        - `economic_cost` — the expected-cost loss function that
          drives threshold selection (Sprint 4).
        - `precision_recall_at_k` — the top-K flagging operating
          point (analyst-capacity constrained).
        - `recall_at_fpr` — the FPR-constrained operating point
          (customer-friction constrained).
        - `compute_psi` — the population-stability index, the
          fraud-industry standard drift signal.

Trade-offs considered:
    - Defaults on `economic_cost` are resolved from `get_settings()`
      so production code picks up the configured USD costs, while
      tests can override them per-call for sensitivity analysis.
    - `recall_at_fpr` uses `sklearn.metrics.roc_curve` to enumerate
      thresholds — O(N log N) — rather than a custom binary search.
      Operating-point selection runs once per evaluation, not in a
      hot loop.
    - `compute_psi` replaces zero fractions with 1e-4 per industry
      convention. Alternatives (Laplace smoothing, additive-1) give
      different magnitudes but equivalent alert decisions at the
      standard <0.1 / 0.1-0.25 / >0.25 bands.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import roc_curve

from fraud_engine.config.settings import get_settings

# Industry convention: PSI floor for empty bins. Avoids log(0) without
# distorting the alert-band magnitudes. Referenced in multiple places
# (PSI docstring, test assertions) so it lives as a module constant.
_PSI_EPSILON: float = 1e-4


def economic_cost(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    fraud_cost: float | None = None,
    fp_cost: float | None = None,
    tp_cost: float | None = None,
) -> float:
    """Compute total expected USD cost for a set of binary predictions.

    Formula: `FN * fraud_cost + FP * fp_cost + TP * tp_cost`.

    Business rationale:
        The "which threshold wins" decision must be made in the units
        the business cares about — dollars, not F1. A false negative
        costs ~13x more than a false positive in this dataset
        (chargeback + reputation vs. a support call), which means a
        purely-symmetric metric like F1 picks the wrong threshold.
        This function is the loss that Sprint 4's threshold optimiser
        minimises and that Sprint 6's drift monitor tracks in
        production.

    Trade-offs considered:
        - Each cost is overridable per call so Sprint 4's sensitivity
          analysis can sweep ±20% around the defaults and confirm the
          chosen threshold is robust.
        - We compute TN implicitly (true negatives have zero marginal
          cost — not blocking a legit txn is free). This matches how
          acquirers model the space in practice.
        - `np.asarray` on both inputs so ndarrays, pandas Series, and
          python lists all work.

    Args:
        y_true: Ground-truth binary labels in {0, 1}.
        y_pred: Predicted binary labels in {0, 1}. Apply thresholding
            to probabilities before calling.
        fraud_cost: USD cost of a false negative (missed fraud).
            Defaults to `settings.fraud_cost_usd`.
        fp_cost: USD cost of a false positive (blocked legit txn).
            Defaults to `settings.fp_cost_usd`.
        tp_cost: USD cost of a true positive (analyst review).
            Defaults to `settings.tp_cost_usd`.

    Returns:
        Total expected cost in USD.
    """
    settings = get_settings()
    fn_usd = settings.fraud_cost_usd if fraud_cost is None else fraud_cost
    fp_usd = settings.fp_cost_usd if fp_cost is None else fp_cost
    tp_usd = settings.tp_cost_usd if tp_cost is None else tp_cost

    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)

    fn = int(((y_true_arr == 1) & (y_pred_arr == 0)).sum())
    fp = int(((y_true_arr == 0) & (y_pred_arr == 1)).sum())
    tp = int(((y_true_arr == 1) & (y_pred_arr == 1)).sum())

    return float(fn * fn_usd + fp * fp_usd + tp * tp_usd)


def precision_recall_at_k(
    y_true: ArrayLike,
    y_scores: ArrayLike,
    k: int,
) -> tuple[float, float]:
    """Precision and recall when the top-K scored samples are flagged.

    Business rationale:
        Analyst review teams have finite capacity — they can only work
        K alerts per hour. The "top-K" operating point is the honest
        way to report how well the model supports that workflow.
        Raw precision and recall at a fixed threshold miss this — if
        the model upscales, the analyst queue overflows.

    Trade-offs considered:
        - We use `np.argpartition` rather than a full sort because K
          is typically much smaller than N. Correctness is identical;
          the constant factor saves time on 500K-row validation sets.
        - Ties at rank K are broken by index order. In practice, score
          ties in LightGBM are rare enough (float64 probabilities)
          that this doesn't matter; if it does, pre-permute the input.

    Args:
        y_true: Ground-truth binary labels.
        y_scores: Continuous scores (higher = more fraud-like).
        k: Top-K cutoff. Must be >= 1 and <= len(y_true).

    Returns:
        `(precision, recall)` — precision among top-K predictions,
        recall across all positives.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_scores_arr = np.asarray(y_scores).astype(float)
    n = len(y_true_arr)
    if k < 1 or k > n:
        raise ValueError(f"k={k} out of range [1, {n}]")

    top_k_idx = np.argpartition(-y_scores_arr, kth=k - 1)[:k]
    top_k_labels = y_true_arr[top_k_idx]
    tp = int(top_k_labels.sum())
    precision = tp / k
    total_positives = int(y_true_arr.sum())
    recall = tp / total_positives if total_positives > 0 else 0.0
    return precision, recall


def recall_at_fpr(
    y_true: ArrayLike,
    y_scores: ArrayLike,
    target_fpr: float,
) -> float:
    """Recall (TPR) at the highest threshold where FPR <= target.

    Business rationale:
        Operational fraud teams express risk appetite as "we cannot
        exceed X% false-positive rate without breaking customer
        experience." Reporting recall at a fixed FPR tells them
        exactly how much fraud the model catches inside that budget —
        a number that maps directly onto the product target, unlike
        raw AUC (which averages across all operating points,
        including ones that violate the FPR budget).

    Trade-offs considered:
        - `sklearn.metrics.roc_curve` gives all thresholds in one O(N
          log N) call. A custom binary search would be faster
          asymptotically but less readable; we run this once per
          evaluation.
        - If no threshold satisfies the budget (degenerate case for
          tiny targets on small datasets), return 0.0 — a "no recall
          available inside budget" signal. The alternative (raising)
          forces callers to wrap every call in a try/except.

    Args:
        y_true: Ground-truth binary labels.
        y_scores: Continuous scores (higher = more fraud-like).
        target_fpr: Maximum allowed false-positive rate, in [0, 1].

    Returns:
        The TPR at the highest threshold with FPR <= target_fpr, or
        0.0 if no such threshold exists.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_scores_arr = np.asarray(y_scores).astype(float)
    fpr, tpr, _ = roc_curve(y_true_arr, y_scores_arr)

    # roc_curve returns thresholds in monotonically decreasing order.
    # We want the largest TPR among operating points where fpr <=
    # target. If none qualify (target lower than the smallest
    # achievable FPR), return 0.0.
    mask = fpr <= target_fpr
    if not mask.any():
        return 0.0
    return float(tpr[mask].max())


def compute_psi(
    baseline: ArrayLike,
    current: ArrayLike,
    *,
    bins: int = 10,
) -> float:
    """Population Stability Index between two distributions.

    Alert bands (industry convention):
        - PSI < 0.10   — no significant population shift.
        - 0.10 to 0.25 — moderate shift; investigate.
        - PSI > 0.25   — significant shift; model re-fit likely needed.

    Business rationale:
        PSI is the fraud-industry standard for distribution drift
        because it's invariant to bin count (within reason) and
        produces a single summary number. Sprint 6's monitoring
        dashboard alerts on PSI > 0.1 per feature; the same function
        is used in Sprint 4 when we check that the validation set's
        score distribution matches the training set's.

    Trade-offs considered:
        - We bin on the baseline's empirical quantiles (equal-
          frequency) rather than on equal-width ranges. Equal-width
          bins are unstable when the distribution is skewed — a
          single outlier reshuffles every bin. Equal-frequency is
          stable on fraud-domain tail distributions.
        - Empty bins are floored at `_PSI_EPSILON` (1e-4) to avoid
          `log(0)`. Laplace / additive-1 smoothing gives a slightly
          different magnitude but identical alert decisions.
        - `bins=10` is the industry default; 5 is too coarse to catch
          tail drift, 20+ generates noise on small samples.

    Args:
        baseline: Reference distribution (e.g. training scores).
        current: New distribution to compare (e.g. production scores
            over the past week).
        bins: Number of quantile buckets.

    Returns:
        PSI value. See alert bands above for interpretation.
    """
    baseline_arr = np.asarray(baseline).astype(float)
    current_arr = np.asarray(current).astype(float)

    # Equal-frequency bin edges from the baseline quantiles. `np.unique`
    # collapses ties (important when the baseline is heavily discrete,
    # e.g. a boolean feature); otherwise `np.digitize` crashes.
    quantile_edges: NDArray[np.float64] = np.quantile(
        baseline_arr,
        q=np.linspace(0, 1, bins + 1),
    )
    quantile_edges = np.unique(quantile_edges)
    if len(quantile_edges) < 2:
        # Degenerate baseline — single value everywhere. No meaningful
        # PSI can be computed; treat as zero drift.
        return 0.0

    # `right=False` so a value equal to an interior edge falls in the
    # upper bin; edge cases at min/max are clipped to [1, len(edges)-1].
    baseline_bins = np.clip(
        np.digitize(baseline_arr, quantile_edges[1:-1], right=False),
        0,
        len(quantile_edges) - 2,
    )
    current_bins = np.clip(
        np.digitize(current_arr, quantile_edges[1:-1], right=False),
        0,
        len(quantile_edges) - 2,
    )

    n_bins = len(quantile_edges) - 1
    baseline_frac = np.array(
        [
            max(float((baseline_bins == i).sum()) / len(baseline_arr), _PSI_EPSILON)
            for i in range(n_bins)
        ]
    )
    current_frac = np.array(
        [
            max(float((current_bins == i).sum()) / len(current_arr), _PSI_EPSILON)
            for i in range(n_bins)
        ]
    )

    psi = float(((current_frac - baseline_frac) * np.log(current_frac / baseline_frac)).sum())
    return psi


__all__ = [
    "compute_psi",
    "economic_cost",
    "precision_recall_at_k",
    "recall_at_fpr",
]
