import pandas as pd
import numpy as np

class BehavioralFeatureEngineer:
    """
    Tier 3: Behavioral Deviation Features

    Business justification:
    - Static rules (e.g. transaction > $500) can fail because how normal a $500 purchase is depends
      on the client. This class will use Z-scores to normalize and identify how uncommon a large
      purchase is for X individual.
    """

    def __init__(self, key_entity = 'card1'):
        self.key_entity = key_entity

    def fit_transform(self, df):
        df = df.copy()
        print(f'Creating Tier 3 behavioral features grouped by {self.key_entity}...')

        # Feature 1: Transaction amount deviation
        # Calculating mean and std per user
        # In production, this can be stored in Redis rather than a database for low latency
        card_mean = df.groupby(self.key_entity)['TransactionAmt'].transform('mean')
        card_std = df.groupby(self.key_entity)['TransactionAmt'].transform('std')

        # Avoid dividing by zero
        card_std = card_std.replace(0, 1)

        # Z-score
        df['transaction_zscore'] = (df['TransactionAmt'] - card_mean) / card_std

        # Feature 2: Time consistency
        # Exploring whether the client usually makes a purchase at this hour
        # Same logic as transaction_zscore, can also be stored in Redis
        df['hour_of_day'] = (df['TransactionDT'] // 3600) % 24
        user_hour_mean = df.groupby(self.key_entity)['hour_of_day'].transform('mean')
        df['hour_deviation'] = np.abs(df['hour_of_day'] - user_hour_mean)

        return df