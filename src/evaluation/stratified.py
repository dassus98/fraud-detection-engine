import pandas as pd
import numpy as np
import logging
from src.evaluation.metrics import calculate_economic_cost

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class StratifiedEvaluator:
    """
    Trying to figure out where money is being lost.
    """
    
    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold
        
    def analyze(self, df, target_col = 'isFraud'):
        logging.info("Starting Stratified Analysis...")
        
        df = df.copy().reset_index(drop = True)

        # Binning transaction amount
        # <$50 = low risk, $50-$200 = medium risk, >$200 = high risk
        df = df.copy()
        df['Amt_Bin'] = pd.cut(df['TransactionAmt'], 
                               bins=[-1, 50, 200, 100000], 
                               labels=['Low (<$50)', 'Med ($50-$200)', 'High (>$200)'])
        
        # Identify numeric columns for prediction
        exclude_cols = [target_col, 'TransactionID', 'TransactionDT', 'Amt_Bin']
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        numeric_cols = df[feature_cols].select_dtypes(include=['number', 'bool']).columns.tolist()
        
        y_pred_proba = self.model.predict(df[numeric_cols])
        
        # Calculating metrics
        report = []
        
        for bin_name in df['Amt_Bin'].unique():
            subset = df[df['Amt_Bin'] == bin_name]
            if len(subset) == 0:
                continue
                
            y_true = subset[target_col]
            y_prob = y_pred_proba[subset.index]
            
            total_cost, cost_per_txn = calculate_economic_cost(
                y_true, y_prob, self.threshold
            )
            
            fraud_rate = y_true.mean()
            
            report.append({
                'Segment': bin_name,
                'Count': len(subset),
                'Fraud_Rate': f"{fraud_rate:.1%}",
                'Cost_Per_Txn': f"${cost_per_txn:.2f}",
                'Total_Cost': f"${total_cost:,.0f}"
            })
            
        # Converting to table
        report_df = pd.DataFrame(report).sort_values('Segment')
        logging.info("\n" + report_df.to_string(index=False))
        
        return report_df