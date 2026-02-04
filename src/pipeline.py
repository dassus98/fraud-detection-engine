import pandas as pd
import gc
from src.features.v_features import VFeatureCleaner

class FraudPipeline:
    def __init__(self, selected_features = None, v_threshold = 0.90):
        """
        Docstring for __init__
        
        :param self: Description
        :param selected_features: Description
        :param v_threshold: Description
        """

        self.manual_selection = selected_features or [
            'TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
            'addr1', 'addr2', 'dist1', 'dist2', 'P_emaildomain', 'R_emaildomain',
            'C1', 'C2', 'C5', 'C8', 'C9', 'C12', # Choosing Cs with >0.30 correlation with Fraud
            'D1', 'D3', 'D4', 'D5', 'D8', 'D10', 'D11', 'D13', 'D14', 'D15', # Choosing non-collinear D variables
            'M1', 'M4', 'M5', 'M6', 'M7',
        ]

        self.v_cleaner = VFeatureCleaner(threshold=v_threshold)

    def fit_transform(self, df):
        """
        Cleans V variables.
        """

        print('--- Fitting Pipeline ---')

        cols_to_keep = [col for col in self.manual_selection if col in df.columns]
        # cols_to_keep = self.manual_selection + v_cols
        X = df[cols_to_keep].copy()

        print('Optimizing V features...')
        self.v_cleaner.fit(X)
        X = self.v_cleaner.transform(X)

        print('Pipeline complete.')
        return X
    
    def transform(self, df):
        """
        Docstring for transform
        
        :param self: Description
        :param df: Description
        """
        print("--- Pipeline: Transform ---")
        
        # v_cols = [c for c in df.columns if c.startswith('V')]
        cols_to_keep = [col for col in self.manual_selection if col in df.columns]
        # cols_to_keep = [c for c in cols_to_keep if c in df.columns]
        
        X = df[cols_to_keep].copy()
        X = self.v_cleaner.transform(X)
        
        return X
    
if __name__ == '__main__':
    print('Loading data to test pipeline...')
    df_test = pd.read_csv('data/raw/train_transaction.csv')
    test_features = ['ProductCD', 'card1', 'C2', 'D3', 'M4', 'V5', 'V6', 'V7', 'V8']
    test_pipeline = FraudPipeline(selected_features=test_features)

    df_processed = test_pipeline.fit_transform(df_test)

    print('\nRemaining columns: ', df_processed.columns.tolist())