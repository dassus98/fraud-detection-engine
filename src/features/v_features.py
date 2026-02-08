import pandas as pd
import numpy as np

class VFeatureCleaner:
    """
    Cleans up the V variables. There are 339 V variables with a lot of high collinearity.
    This removes any with a collinearity > 0.90.
    """

    def __init__(self, threshold=0.90):
        self.threshold = threshold
        self.drop_cols = []
        self.top_features = []

    def fit(self, df):
        """
        Learns which columns to drop.
        """
        v_cols = [col for col in df.columns if col.startswith('V')]

        print(f'Exploring redundant V features...')
        corr_matrix = df[v_cols].corr().abs()

        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.drop_cols = [col for col in upper.columns if any(upper[col] > self.threshold)]
        self.top_features = [col for col in v_cols if col not in self.drop_cols]

        print(f'{len(self.drop_cols)} redundant features.')
        return self

    def transform(self, df):
        """
        Drop features learned in fit function.
        """

        cols_to_remove = [col for col in self.drop_cols if col in df.columns]

        if cols_to_remove:
            print(f'Dropping {len(cols_to_remove)} V variables.')
            return df.drop(columns=cols_to_remove)
        
        return df
    
if __name__ == '__main__':
    try:
        df_test = pd.read_csv('../data/raw/train_transaction.csv')
        cleaner = VFeatureCleaner(threshold=0.90)
        cleaner.fit(df_test)
        df_transformed = cleaner.transform(df_test)
        print(f'Original shape: {df_test.shape}, new shape: {df_transformed.shape}')
    except:
        pass