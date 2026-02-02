import pandas as pd
import numpy as np

class VFeatureCleaner:
    """
    Docstring for VFeatureCleaner
    """

    def __init__(self, threshold=0.90):
        self.threshold = threshold
        self.drop_cols = []
        self.top_features = []

    def fit(self, df):
        """
        Docstring for fit
        
        :param self: Description
        :param df: Description
        """
        v_cols = [col for col in df.columns if col.startswith('V')]

        corr_matrix = df.corr().abs()

        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.drop_cols = [col for col in upper.columns if any(upper[col] > self.threshold)]
        self.top_features = [col for col in v_cols if col not in self.drop_cols]

        print(f'{len(self.drop_cols)} redundant features.')

    def transform(self, df):
        """
        Docstring for transform
        
        :param self: Description
        :param df: Description
        """

        cols_to_remove = [col for col in self.drop_cols if col in df.columns]

        if cols_to_remove:
            print(f'Dropping {len(cols_to_remove)} V variables.')
            return df.drop(columns=cols_to_remove)
        
        return df
    
if __name__ == '__main__':
    df_test = pd.read_csv('../data/raw/train_transaction.csv')
    cleaner = VFeatureCleaner(threshold=0.90)
    cleaner.fit(df_test)
