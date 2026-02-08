import pandas as pd
import numpy as np
import gc
from sklearn.base import BaseEstimator, TransformerMixin
from src.config import SELECTED_FEATURES, DROP_COLS
from src.features.v_features import VFeatureCleaner

class FraudPipeline(BaseEstimator, TransformerMixin):
    """
    Production pipeline intended to compelete four tasks:
    - Cleaning up the V variables (done here to prevent data leakage)
    - Feature selection
    - Categorical encoding of object variables
    - Null value handling
    """
    def __init__(self):
        self.v_cleaner = VFeatureCleaner()
        self.cat_encoders = {}
        self.string_cols = [
            'ProductCD', 'card4', 'card6', 
            'P_emaildomain', 'R_emaildomain',
            'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9'
        ]

    def fit(self, X, y=None):
        """
        Learns from training data.
        """

        print('--- Fitting Pipeline ---')

        v_cols = [col for col in X.columns if col.startswith('V')]
        if v_cols:
            print(f'Optimizing {len(v_cols)} V variables.')
            self.v_cleaner.fit(X[v_cols])

        print('Encoding categorical variables.')
        self.cat_cols = [col for col in self.string_cols if col in X.columns and col in SELECTED_FEATURES]
        for col in self.cat_cols:
            self.cat_encoders[col] = {k: i for i, k in enumerate(X[col].astype(str).value_counts().index)}

        return self

    def transform(self, X):
        """
        Transforming the data.
        """
        X = X.copy()
        
        # Drop features learned as redundant during fit.
        X = self.v_cleaner.transform(X)
        remaining_v_cols = [col for col in X.columns if col.startswith('V')]

        # Feature selection
        base_features_present = [col for col in SELECTED_FEATURES if col in X.columns]
        final_cols = base_features_present + remaining_v_cols
        X_final = X[final_cols].copy()

        # Categorical encoding + handling null values
        for col, mapping in self.cat_encoders.items():
            if col in X_final.columns:
                # Change to int
                X_final[col] = X_final[col].astype(str).map(mapping).fillna(-1).astype(int)
                # Change to category for LightGBM to train (so it doesn't confuse it for numeric values)
                X_final[col] = X_final[col].astype('category')
        
        for col in SELECTED_FEATURES:
            if col not in X_final.columns:
                X_final[col] = np.nan

        return X_final
    
if __name__ == '__main__':
    print('Loading data to test pipeline...')
    try:
        df_test = pd.read_csv('data/raw/train_transaction.csv', nrows=1000)
        test_pipeline = FraudPipeline()
        test_pipeline.fit(df_test)
        X_processed = test_pipeline.transform(df_test)
        print(f'Input shape: {df_test.shape}')
        print(f'Output shape: {X_processed.shape}')
        print(f'Remaining columns: {X_processed.columns.tolist()}')
    except FileNotFoundError:
        print('File not found - check training data file path.')
    pass