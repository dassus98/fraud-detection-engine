import numpy as np
import pandas as pd

def calculate_economic_cost(y_true, y_pred_proba, threshold, cost_fn = 525, cost_fp = 150):
    """
    Calculating the expected economic cost of a fraud model based on a decision threshold.
    Assumptions for the values:
    - Chargeback fee ($25): Stripe Canada charges $15 per VISA chargeback, PayPal charges $20. 
      Let's take $25 as an assumption and leave out the extremes (e.g. crypto).
    - Support call cost ($25): Let's assume a FP leads to support ticket 100% 
      of the time (very conservative estimate). Google states that average support call 
      cost in fintech is roughly $15-$25 per call.
    - Average fraud loss ($500): Fraudsters will eventually try to steal a large amount. 
      Let's set it to $500 (within lower credit limits) since detecting a $30 minor fraud would be genuinely difficult.
    - CLV (Customer Lifetime Value): Assumed to be roughly $2500. Only found one source 
      for justifying CLV for banks.
    - Source for chargeback fee (Stripe): https://stripe.com/en-ca/pricing 
    - Source for chargeback fee (PayPal): https://www.paypal.com/us/business/paypal-business-fees#statement-12
    - Source for chargeback justification: https://b2b.mastercard.com/news-and-insights/blog/what-s-the-true-cost-of-a-chargeback-in-2025/
    - Source for CLV: https://www.mx.com/blog/customer-lifetime-value-in-banking-hinges-on-advocacy-and-engagement/
    
    Business Justification:
    - Reducing cost is more important than accuracy
    - cost_fn ($525): Fraud amount + Chargeback fee (+ overhead?)
    - cost_fp ($150): (5% chance of churn * $2500 LTV) + Support call cost
    """
    # Convert probabilities to binary decisions based on threshold
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Confusion Matrix Components
    # fn = Fraud we missed (y_true=1, y_pred=0)
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    
    # fp = Legitimate users we blocked (y_true=0, y_pred=1)
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    
    # Total Cost Calculation
    total_cost = (fn * cost_fn) + (fp * cost_fp)
    
    # Cost per transaction (normalized metric for comparing models)
    cost_per_txn = total_cost / len(y_true)
    
    return total_cost, cost_per_txn

def find_optimal_threshold(y_true, y_pred_proba, cost_fn=525, cost_fp=150):
    """
    Scans thresholds from 0.01 to 0.99 to find the point that minimizes economic cost.
    Business justification: 
    - A default threshold of 0.5 is arbitrary and ignores the fact that fraud is more expensive 
      than friction in our matrix ($525 vs $150)
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    costs = []
    
    for t in thresholds:
        _, cost_per_txn = calculate_economic_cost(y_true, y_pred_proba, t, cost_fn, cost_fp)
        costs.append(cost_per_txn)
        
    optimal_idx = np.argmin(costs)
    return thresholds[optimal_idx], costs[optimal_idx]