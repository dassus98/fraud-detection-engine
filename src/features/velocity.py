import pandas as pd
import numpy as np

class VelocityFeatureEngineer:
    """
    Tier 4: Exponential decay features

    Business justification:
    - Fraud is unpredictable and can happen in bursts. It's probably beneficial to place a higher
      importance on more recent transactions. Here a half-life decay will be established to provide
      more weight to recent activity.
    """

    def __init__(self, time_col = 'TransactionDT', key_col = 'card1'):
        self.time_col = time_col
        self.key_col = key_col

    def fit_transform(self, df):
        print('Creating Tier 4 velocity features...')

        df = df.sort_values([self.key_col, self.time_col])

        # Feature 1: Time since last transaction
        # Tries to detect fraud which happens just after a client has had their information stolen.
        df['time_since_last_txn'] = df.groupby(self.key_col)[self.time_col].diff()

        # Fill null values with high value so they don't generate noise
        df['time_since_last_txn'] = df['time_since_last_txn'].fillna(10000)

        # Feature 2: Acceleration of transactions
        # Tries to detect if the time between transactions is shrinking as multiple instances of 
        # fraud are happening in quick succession
        df['time_since_prev_diff'] = df.groupby(self.key_col)['time_since_last_txn'].diff()

        return df