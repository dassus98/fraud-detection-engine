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
        - `precision_recall_at_k` — the top-fraction flagging
          operating point (analyst-capacity constrained).
        - `recall_at_fpr` — the FPR-constrained operating point
          (customer-friction constrained).
        - `compute_psi` — the population-stability index, the
          fraud-industry standard drift signal.

Trade-offs considered:
    - `economic_cost` accepts `None` for the USD costs and falls back
      to `get_settings()` so production code picks up the configured
      defaults. Tests and sensitivity analyses pass explicit values to
      sweep around those defaults.
    - `economic_cost` returns a dict (rather than a bare float) so
      the same call exposes `total_cost`, `cost_per_txn`, and the
      individual FN/FP/TP/TN counts. Sprint 4's threshold sweep keys
      off `total_cost`; the Sprint 5 MLflow logger persists the
      per-class counts alongside it so a failing regression is easy
      to diagnose.
    - `precision_recall_at_k` takes `k` as a *fraction in (0, 1]*
      (what the spec names). Analyst capacity is naturally expressed
      as a percentage of daily volume, not a raw count that has to
      be rescaled whenever volume drifts. The implementation
      converts to a count via `ceil(k * N)` internally.
    - `recall_at_fpr` uses `sklearn.metrics.roc_curve` to enumerate
      thresholds — O(N log N) — rather than a custom binary search.
      Operating-point selection runs once per evaluation, not in a
      hot loop.
    - `compute_psi` replaces zero fractions with a caller-tunable
      `epsilon` (default 1e-6 per the Sprint 0.3.b spec). A smaller
      epsilon preserves the symmetry of PSI under argument swap; a
      larger epsilon (1e-4 is the common industry choice) is more
      forgiving on sparse baseline bins but breaks symmetry visibly.
      The default favours symmetry; the epsilon kwarg lets the
      Sprint 6 drift dashboard pick a larger floor if its operational
      data is sparser than the training distribution.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import roc_curve

from fraud_engine.config.settings import get_settings

# Minimum distinct quantile edges that yield a meaningful PSI. One edge
# (i.e. every baseline value identical) collapses to a single bin,
# which makes the drift computation trivially zero.
_MIN_QUANTILE_EDGES: int = 2


def economic_cost(  # noqa: PLR0913 — the four cost parameters plus two label arrays are the business contract (see docstring); collapsing into a dict would hide the cost-model semantics at every call site.
    y_true: ArrayLike,
    y_pred: ArrayLike,
    fraud_cost: float | None = None,
    fp_cost: float | None = None,
    tp_cost: float | None = None,
    tn_cost: float = 0.0,
) -> dict[str, float]:
    """Compute expected USD cost for a set of binary predictions.

    Cost model:
        `total = FN * fraud_cost + FP * fp_cost + TP * tp_cost
                 + TN * tn_cost`

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
        - Each USD cost is overridable per call so Sprint 4's
          sensitivity analysis can sweep ±20% around the defaults and
          confirm the chosen threshold is robust.
        - Passing `None` (the default) for `fraud_cost`/`fp_cost`/
          `tp_cost` resolves the value from `get_settings()`. A
          mandatory-positional signature (as the spec sketches) would
          force every caller to thread the cost triple through every
          call site — painful and inconsistent with CLAUDE.md §5.4
          ("no hardcoded values outside config"). `tn_cost` has no
          Settings analogue (TN cost is zero by convention in
          fraud-ML) so it defaults to a literal 0.0.
        - Return value is a dict rather than a bare float so the
          same call surfaces the per-class counts the Sprint 4
          threshold sweep writes to MLflow. A dict return also lets
          later sprints add fields (e.g. `expected_chargeback_count`)
          without a breaking-signature change.
        - `np.asarray` on both inputs so ndarrays, pandas Series, and
          Python lists all work.

    Args:
        y_true: Ground-truth binary labels in {0, 1}.
        y_pred: Predicted binary labels in {0, 1}. Apply thresholding
            to probabilities before calling.
        fraud_cost: USD cost of a false negative (missed fraud). If
            None, resolved from `settings.fraud_cost_usd`.
        fp_cost: USD cost of a false positive (blocked legit txn). If
            None, resolved from `settings.fp_cost_usd`.
        tp_cost: USD cost of a true positive (analyst review time).
            If None, resolved from `settings.tp_cost_usd`.
        tn_cost: USD cost of a true negative (nothing happens).
            Defaults to 0.0 — no Settings field since TN cost is
            zero by convention.

    Returns:
        A dict with keys:
            - `total_cost`: total expected USD cost
            - `cost_per_txn`: total_cost / len(y_true)
            - `fn`, `fp`, `tp`, `tn`: raw counts of each outcome,
              floats so downstream MLflow logs accept them without
              coercion.
    """
    settings = get_settings()
    fn_usd = settings.fraud_cost_usd if fraud_cost is None else fraud_cost
    fp_usd = settings.fp_cost_usd if fp_cost is None else fp_cost
    tp_usd = settings.tp_cost_usd if tp_cost is None else tp_cost

    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)
    n = len(y_true_arr)

    fn = int(((y_true_arr == 1) & (y_pred_arr == 0)).sum())
    fp = int(((y_true_arr == 0) & (y_pred_arr == 1)).sum())
    tp = int(((y_true_arr == 1) & (y_pred_arr == 1)).sum())
    tn = int(((y_true_arr == 0) & (y_pred_arr == 0)).sum())

    total_cost = fn * fn_usd + fp * fp_usd + tp * tp_usd + tn * tn_cost
    # Guard against an empty input producing ZeroDivisionError. An
    # empty input is degenerate but callers (notably property-based
    # tests) may still reach here.
    cost_per_txn = total_cost / n if n > 0 else 0.0

    return {
        "total_cost": float(total_cost),
        "cost_per_txn": float(cost_per_txn),
        "fn": float(fn),
        "fp": float(fp),
        "tp": float(tp),
        "tn": float(tn),
    }


def precision_recall_at_k(
    y_true: ArrayLike,
    y_scores: ArrayLike,
    k: float,
) -> tuple[float, float]:
    """Precision and recall when the top-`k` fraction of scores is flagged.

    Business rationale:
        Analyst review teams have finite capacity — they can only work
        a fixed percentage of daily volume. The "top-K" operating
        point is the honest way to report how well the model supports
        that workflow. Raw precision/recall at a fixed probability
        threshold miss this — if the model upscales, the analyst
        queue overflows.

    Trade-offs considered:
        - `k` is a fraction in (0, 1] (not a count). Analyst capacity
          is naturally expressed as a percentage of daily volume;
          expressing it as an integer count is brittle when the
          daily volume fluctuates (e.g. weekend dips).
        - The count of items to flag is `ceil(k * N)` — so the
          smallest K always flags at least one item (a strict
          floor of 1 prevents `k * N = 0.7` from collapsing to zero
          and reporting a nonsense ratio).
        - We use `np.argpartition` rather than a full sort because
          the flagged fraction is typically much smaller than N.
          Correctness is identical; the constant factor saves time
          on 500K-row validation sets.
        - Ties at rank K are broken by index order. In practice,
          score ties in LightGBM are rare enough (float64
          probabilities) that this doesn't matter; if it does,
          pre-permute the input.

    Args:
        y_true: Ground-truth binary labels.
        y_scores: Continuous scores (higher = more fraud-like).
        k: Fraction of items to flag, in (0, 1].

    Returns:
        `(precision, recall)` — precision among flagged predictions,
        recall across all positives.

    Raises:
        ValueError: If `k` is not in (0, 1].
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_scores_arr = np.asarray(y_scores).astype(float)
    n = len(y_true_arr)
    if not (0.0 < k <= 1.0):
        raise ValueError(f"k={k} must be a fraction in (0, 1]")

    # Minimum 1 item — prevents k=0.01 * N=50 = 0.5 → 0 from returning
    # division-by-zero. Maximum N (the full dataset) when k=1.0.
    k_count = max(1, int(np.ceil(k * n)))

    top_k_idx = np.argpartition(-y_scores_arr, kth=k_count - 1)[:k_count]
    top_k_labels = y_true_arr[top_k_idx]
    tp = int(top_k_labels.sum())
    precision = tp / k_count
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
        - `sklearn.metrics.roc_curve` gives all thresholds in one
          O(N log N) call. A custom binary search would be faster
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
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Population Stability Index between two distributions.

    PSI formula:
        `sum over bins of (p_curr - p_base) * log(p_curr / p_base)`

    Alert bands (industry convention):
        - PSI < 0.10   — no significant population shift.
        - 0.10 to 0.25 — moderate shift; investigate.
        - PSI > 0.25   — significant shift; model re-fit likely
          needed.

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
        - Empty bins are floored at `epsilon` to avoid `log(0)`. The
          default 1e-6 preserves symmetry (`psi(a, b) ≈ psi(b, a)`)
          to within floating-point noise; raising epsilon to 1e-4
          breaks that symmetry visibly on sparse baselines but is
          kinder on very small samples. Callers who care about
          symmetry (e.g. A/B drift comparisons) keep the default;
          callers who trade symmetry for smoothing bump epsilon up.
        - `bins=10` is the industry default; 5 is too coarse to
          catch tail drift, 20+ generates noise on small samples.

    Args:
        baseline: Reference distribution (e.g. training scores).
        current: New distribution to compare (e.g. production scores
            over the past week).
        bins: Number of quantile buckets.
        epsilon: Floor for empty-bin fractions. Keep the default
            (1e-6) for symmetric comparisons; raise to 1e-4 for
            sparse-baseline smoothing.

    Returns:
        PSI value. See alert bands above for interpretation.
    """
    baseline_arr = np.asarray(baseline).astype(float)
    current_arr = np.asarray(current).astype(float)

    # Equal-frequency bin edges from the baseline quantiles.
    # `np.unique` collapses ties (important when the baseline is
    # heavily discrete, e.g. a boolean feature); otherwise
    # `np.digitize` crashes on the equal edges.
    quantile_edges: NDArray[np.float64] = np.quantile(
        baseline_arr,
        q=np.linspace(0, 1, bins + 1),
    )
    quantile_edges = np.unique(quantile_edges)
    if len(quantile_edges) < _MIN_QUANTILE_EDGES:
        # Degenerate baseline — single value everywhere. No
        # meaningful PSI can be computed; treat as zero drift.
        return 0.0

    # `right=False` so a value equal to an interior edge falls in
    # the upper bin; edge cases at min/max are clipped to
    # [0, len(edges)-2].
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
        [max(float((baseline_bins == i).sum()) / len(baseline_arr), epsilon) for i in range(n_bins)]
    )
    current_frac = np.array(
        [max(float((current_bins == i).sum()) / len(current_arr), epsilon) for i in range(n_bins)]
    )

    return float(((current_frac - baseline_frac) * np.log(current_frac / baseline_frac)).sum())


__all__ = [
    "compute_psi",
    "economic_cost",
    "precision_recall_at_k",
    "recall_at_fpr",
]
