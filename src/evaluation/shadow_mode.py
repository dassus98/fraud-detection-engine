import os
import logging
import joblib
import pandas as pd
import numpy as np

from src.config import DATA_PATH, MODEL_SAVE_PATH, PIPELINE_SAVE_PATH
from src.utils import reduce_mem_usage
from src.evaluation.metrics import calculate_economic_cost, find_optimal_threshold

logger = logging.getLogger(__name__)

COST_FN = 525   # cost per missed fraud (chargeback + overhead)
COST_FP = 150   # cost per blocked legitimate transaction (CLV churn risk + support)


def load_validation_set():
    """Load and return the same temporal 20% validation split used during training."""
    logger.info(f'Loading data from {DATA_PATH}')
    df = pd.read_csv(DATA_PATH)
    df = reduce_mem_usage(df)
    df = df.sort_values('TransactionDT').reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    val_df = df.iloc[split_idx:].copy()
    logger.info(f'Validation set: {val_df.shape[0]} transactions')
    return val_df


def run_shadow_simulation():
    """Simulate batch scoring on the validation set and write an economic summary report."""
    for path, label in [(MODEL_SAVE_PATH, 'model'), (PIPELINE_SAVE_PATH, 'pipeline')]:
        if not os.path.exists(path):
            logger.error(f'Missing {label} at {path} — run training first.')
            return

    logger.info(f'Loading model from {MODEL_SAVE_PATH}')
    model = joblib.load(MODEL_SAVE_PATH)

    logger.info(f'Loading pipeline from {PIPELINE_SAVE_PATH}')
    pipeline = joblib.load(PIPELINE_SAVE_PATH)

    val_df = load_validation_set()
    y_true = val_df['isFraud'].values
    amounts = val_df['TransactionAmt'].values

    logger.info('Transforming validation data...')
    X_val = pipeline.transform(val_df)

    logger.info('Scoring transactions...')
    y_proba = model.predict(X_val)

    # Find the threshold that minimises economic cost
    threshold, min_cost_per_txn = find_optimal_threshold(y_true, y_proba, COST_FN, COST_FP)
    logger.info(f'Optimal threshold: {threshold:.2f}  (cost/txn: ${min_cost_per_txn:.4f})')

    y_pred = (y_proba >= threshold).astype(int)

    # Confusion matrix masks
    tp_mask = (y_true == 1) & (y_pred == 1)   # fraud caught
    fn_mask = (y_true == 1) & (y_pred == 0)   # fraud missed
    fp_mask = (y_true == 0) & (y_pred == 1)   # legitimate transactions blocked
    tn_mask = (y_true == 0) & (y_pred == 0)   # legitimate transactions passed

    tp = tp_mask.sum()
    fn = fn_mask.sum()
    fp = fp_mask.sum()
    tn = tn_mask.sum()
    total = len(y_true)
    total_fraud = (y_true == 1).sum()

    # Dollar figures using actual TransactionAmt
    fraud_caught_dollars = amounts[tp_mask].sum()
    fraud_missed_dollars = amounts[fn_mask].sum()

    # Economic cost: what the model costs vs doing nothing
    total_cost_with_model, _ = calculate_economic_cost(y_true, y_proba, threshold, COST_FN, COST_FP)
    # Baseline: catch nothing — every fraud case is a false negative
    baseline_cost = total_fraud * COST_FN
    net_revenue_saved = baseline_cost - total_cost_with_model

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    lines = [
        '=' * 60,
        '          SHADOW MODE SIMULATION REPORT',
        '=' * 60,
        '',
        f'  Transactions evaluated   : {total:,}',
        f'  Total fraud cases        : {total_fraud:,} ({100 * total_fraud / total:.2f}%)',
        f'  Decision threshold       : {threshold:.2f}',
        '',
        '--- Confusion Matrix ---',
        f'  True positives (TP)      : {tp:,}',
        f'  False negatives (FN)     : {fn:,}',
        f'  False positives (FP)     : {fp:,}',
        f'  True negatives (TN)      : {tn:,}',
        '',
        '--- Dollar Impact ---',
        f'  Fraud caught             : ${fraud_caught_dollars:>12,.2f}',
        f'  Fraud missed             : ${fraud_missed_dollars:>12,.2f}',
        f'  Fraud catch rate         : {100 * fraud_caught_dollars / (fraud_caught_dollars + fraud_missed_dollars):.1f}%' if (fraud_caught_dollars + fraud_missed_dollars) > 0 else '  Fraud catch rate         : N/A',
        '',
        '--- Operational Impact ---',
        f'  Legitimate txns blocked  : {fp:,}',
        f'  Block rate (FP rate)     : {100 * fp / (fp + tn):.2f}%' if (fp + tn) > 0 else '  Block rate               : N/A',
        '',
        '--- Economic Cost Model ---',
        f'  Cost per FN (missed)     : ${COST_FN}',
        f'  Cost per FP (blocked)    : ${COST_FP}',
        f'  Baseline cost (no model) : ${baseline_cost:>12,}',
        f'  Model total cost         : ${total_cost_with_model:>12,.0f}',
        f'  Net revenue saved        : ${net_revenue_saved:>12,.0f}',
        f'  Cost per transaction     : ${min_cost_per_txn:.4f}',
        '',
        '--- Classification Metrics ---',
        f'  Precision                : {precision:.4f}',
        f'  Recall                   : {recall:.4f}',
        '',
        '=' * 60,
    ]

    report = '\n'.join(lines)
    print(report)

    os.makedirs('reports', exist_ok=True)
    report_path = os.path.join('reports', 'shadow_mode_results.txt')
    with open(report_path, 'w') as f:
        f.write(report + '\n')
    logger.info(f'Report saved to {report_path}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    run_shadow_simulation()
