"""
Stratified economic-cost analysis segmented by transaction amount.

Splits the validation set into three spend bands (low / medium / high) and
reports the economic cost per band.  This surfaces whether the model
disproportionately misses large-value fraud — the category where a false
negative is most expensive.

Usage (after training):
    from src.evaluation.stratified import StratifiedEvaluator
    evaluator = StratifiedEvaluator(model=fraud_model, threshold=0.16)
    report_df = evaluator.analyze(val_df_processed, target_col='isFraud')
"""

import logging

import pandas as pd

from src.evaluation.metrics import calculate_economic_cost

logger = logging.getLogger(__name__)


class StratifiedEvaluator:
    """
    Breaks down model performance and economic cost by transaction-amount band.

    Three bands are used:
        - Low    : TransactionAmt < $50
        - Medium : $50 ≤ TransactionAmt ≤ $200
        - High   : TransactionAmt > $200

    For each band the evaluator reports:
        - Total transactions and fraud rate
        - Economic cost per transaction (FN × $525 + FP × $150)
        - Total cost for that segment

    This helps identify whether missed fraud is concentrated in high-value
    transactions, which would warrant a tighter threshold or manual review queue
    for that segment.
    """

    def __init__(self, model, threshold: float):
        """
        Args:
            model:     A fitted FraudModel (or any object with a .predict() method
                       that accepts a numeric DataFrame and returns probabilities).
            threshold: Decision threshold — transactions with predicted fraud
                       probability ≥ threshold are blocked.
        """
        self.model = model
        self.threshold = threshold

    def analyze(self, df: pd.DataFrame, target_col: str = 'isFraud') -> pd.DataFrame:
        """
        Run the stratified analysis and log a per-band cost table.

        Args:
            df:         DataFrame that has already been through FraudPipeline.transform()
                        *plus* still carries 'TransactionAmt' and ``target_col``.
            target_col: Name of the binary fraud label column (1 = fraud, 0 = legitimate).

        Returns:
            DataFrame with one row per amount band and columns:
            Segment, Count, Fraud_Rate, Cost_Per_Txn, Total_Cost.
        """
        logger.info('Starting stratified analysis...')

        df = df.copy().reset_index(drop=True)

        # Bin transaction amounts into three spend bands.
        # The upper bound (100_000) is deliberately far above the dataset maximum
        # so every row falls into exactly one bucket.
        df['Amt_Bin'] = pd.cut(
            df['TransactionAmt'],
            bins=[-1, 50, 200, 100_000],
            labels=['Low (<$50)', 'Med ($50–$200)', 'High (>$200)'],
        )

        # Identify the numeric feature columns the model was trained on.
        # We exclude meta-columns that are not model features.
        exclude_cols = [target_col, 'TransactionID', 'TransactionDT', 'Amt_Bin']
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        numeric_cols = df[feature_cols].select_dtypes(include=['number', 'bool']).columns.tolist()

        # Score the full dataset in one pass, then slice by band.
        y_pred_proba = self.model.predict(df[numeric_cols])

        report = []

        for bin_name in ['Low (<$50)', 'Med ($50–$200)', 'High (>$200)']:
            mask = df['Amt_Bin'] == bin_name
            subset = df[mask]
            if len(subset) == 0:
                continue

            y_true = subset[target_col]
            y_prob = y_pred_proba[mask.values]   # align by boolean mask

            total_cost, cost_per_txn = calculate_economic_cost(
                y_true, y_prob, self.threshold
            )

            report.append({
                'Segment':      bin_name,
                'Count':        len(subset),
                'Fraud_Rate':   f'{y_true.mean():.1%}',
                'Cost_Per_Txn': f'${cost_per_txn:.2f}',
                'Total_Cost':   f'${total_cost:,.0f}',
            })

        report_df = pd.DataFrame(report)
        logger.info('\n' + report_df.to_string(index=False))

        return report_df
