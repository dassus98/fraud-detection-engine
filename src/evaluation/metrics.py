"""
Economic cost functions for threshold optimisation.

Plain-English summary
---------------------
A fraud model makes two types of mistakes:

  False negative (FN) — fraud that slips through undetected.
      Cost: $500 average fraud loss + $25 chargeback fee = $525 per incident.

  False positive (FP) — a legitimate purchase that gets blocked.
      Cost: 5% chance the customer churns × $2,500 lifetime value
            + $25 support call = $150 per incident.

Because missing fraud ($525) is more than 3× as expensive as blocking a
good customer ($150), the model should cast a wider net than the default
0.50 threshold.  find_optimal_threshold() scans the full threshold range
and returns the cut-point that minimises total expected cost.

Sources
-------
Chargeback fee    : https://stripe.com/en-ca/pricing
                    https://www.paypal.com/us/business/paypal-business-fees
Chargeback impact : https://b2b.mastercard.com/news-and-insights/blog/
                      what-s-the-true-cost-of-a-chargeback-in-2025/
Customer LTV      : https://www.mx.com/blog/
                      customer-lifetime-value-in-banking-hinges-on-advocacy-and-engagement/
"""

import numpy as np


def calculate_economic_cost(
    y_true,
    y_pred_proba,
    threshold: float,
    cost_fn: float = 525,
    cost_fp: float = 150,
):
    """
    Compute the total and per-transaction economic cost at a given threshold.

    Args:
        y_true:       Binary ground-truth labels (1 = fraud, 0 = legitimate).
        y_pred_proba: Model-predicted fraud probabilities in [0, 1].
        threshold:    Decision cut-point — transactions with probability ≥
                      threshold are predicted as fraud (BLOCK).
        cost_fn:      Dollar cost of each missed fraud (false negative).
                      Default $525 = $500 fraud loss + $25 chargeback fee.
        cost_fp:      Dollar cost of each wrongly blocked transaction
                      (false positive).
                      Default $150 = 5% churn × $2,500 LTV + $25 support call.

    Returns:
        (total_cost, cost_per_txn) — total dollar cost and the same figure
        normalised per transaction.
    """
    # Classify each transaction as fraud (1) or legitimate (0).
    y_pred = (y_pred_proba >= threshold).astype(int)

    # False negatives: real fraud that the model let through.
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    # False positives: legitimate transactions that the model blocked.
    fp = ((y_true == 0) & (y_pred == 1)).sum()

    total_cost = (fn * cost_fn) + (fp * cost_fp)

    # Normalise by dataset size so costs are comparable across different
    # validation set sizes.
    cost_per_txn = total_cost / len(y_true)

    return total_cost, cost_per_txn


def find_optimal_threshold(
    y_true,
    y_pred_proba,
    cost_fn: float = 525,
    cost_fp: float = 150,
):
    """
    Scan thresholds from 0.01 to 0.99 and return the one that minimises
    expected cost per transaction.

    A default threshold of 0.50 is arbitrary and ignores cost asymmetry.
    Because FN is 3.5× more expensive than FP in our model, the optimal
    threshold sits well below 0.50 — the model should block more
    aggressively to avoid missing fraud.

    Args:
        y_true:       Binary ground-truth labels.
        y_pred_proba: Model-predicted fraud probabilities.
        cost_fn:      Cost per false negative (missed fraud).
        cost_fp:      Cost per false positive (blocked legitimate transaction).

    Returns:
        (optimal_threshold, min_cost_per_txn)
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    costs = [
        calculate_economic_cost(y_true, y_pred_proba, t, cost_fn, cost_fp)[1]
        for t in thresholds
    ]

    optimal_idx = np.argmin(costs)
    return thresholds[optimal_idx], costs[optimal_idx]
