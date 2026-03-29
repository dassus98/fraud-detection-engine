import logging

import pandas as pd
import numpy as np
import gc
from sklearn.base import BaseEstimator, TransformerMixin
from src.config import SELECTED_FEATURES, DROP_COLS
from src.features.v_features import VFeatureCleaner

logger = logging.getLogger(__name__)

class FraudPipeline(BaseEstimator, TransformerMixin):
    """
    End-to-end feature transformer for the fraud detection model.

    Applies four transformations in sequence:
    1. V-feature pruning   — drops highly correlated V-columns (threshold 0.90)
                             to reduce noise and training time.
    2. Feature selection   — retains only the columns listed in SELECTED_FEATURES
                             plus the surviving V-columns.
    3. Categorical encoding — maps string categories to integers so LightGBM
                             can use them as categorical splits.
    4. Column alignment    — reindexes every output to the exact column list
                             seen at fit time; columns missing at inference
                             (e.g. V-features absent from API requests) become
                             NaN, which LightGBM handles natively.
    """

    def __init__(self):
        self.v_cleaner = VFeatureCleaner()
        self.cat_encoders = {}

        # Full candidate list of string columns that *could* be categorical.
        # At fit time this is filtered down to only those columns that are
        # both present in the training data AND listed in SELECTED_FEATURES
        # (stored as self.cat_cols).  M2, M3, M8, M9 are listed here as
        # candidates but are not in SELECTED_FEATURES, so they are silently
        # ignored during encoding.
        self.string_cols = [
            'ProductCD', 'card4', 'card6',
            'P_emaildomain', 'R_emaildomain',
            'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
        ]

    def fit(self, X, y=None):
        """
        Learns from training data: fits VFeatureCleaner, builds categorical encoders,
        and records the authoritative ordered feature list used to align inference rows.
        """

        logger.info('--- Fitting Pipeline ---')

        v_cols = [col for col in X.columns if col.startswith('V')]
        if v_cols:
            logger.info(f'Optimizing {len(v_cols)} V variables.')
            self.v_cleaner.fit(X[v_cols])

        logger.info('Encoding categorical variables.')
        self.cat_cols = [col for col in self.string_cols if col in X.columns and col in SELECTED_FEATURES]
        for col in self.cat_cols:
            self.cat_encoders[col] = {k: i for i, k in enumerate(X[col].astype(str).value_counts().index)}

        # Store the authoritative ordered column list so transform() can reindex
        # single-row API requests that are missing V-features or SELECTED_FEATURES.
        base_in_train = [col for col in SELECTED_FEATURES if col in X.columns]
        self.feature_names_ = base_in_train + self.v_cleaner.top_features
        logger.info(
            f'Pipeline fitted — {len(self.feature_names_)} features '
            f'({len(base_in_train)} base + {len(self.v_cleaner.top_features)} V).'
        )

        return self

    def transform(self, X):
        """
        Transforms raw transaction data into a model-ready feature matrix.
        Always returns exactly the columns seen at fit time, in the same order;
        columns absent from the input (e.g. V-features missing from API requests)
        are filled with NaN so LightGBM can apply its learned splits correctly.
        """
        X = X.copy()

        # Drop highly-correlated V-features learned during fit.
        X = self.v_cleaner.transform(X)

        # Build the column set actually present in this input.
        base_features_present = [col for col in SELECTED_FEATURES if col in X.columns]
        remaining_v_cols = [col for col in X.columns if col.startswith('V')]
        X_final = X[base_features_present + remaining_v_cols].copy()

        # Encode categorical columns; unseen categories → -1.
        for col, mapping in self.cat_encoders.items():
            if col in X_final.columns:
                X_final[col] = X_final[col].astype(str).map(mapping).fillna(-1).astype(int)
                X_final[col] = X_final[col].astype('category')

        # Reindex to the authoritative column list.
        # Missing columns (e.g. V-features absent from an API request) become NaN.
        # Surplus columns (e.g. isFraud, TransactionDT) are silently dropped.
        if hasattr(self, 'feature_names_'):
            target_cols = self.feature_names_
        else:
            # Legacy fallback: derive column list from existing attributes.
            target_cols = (
                [c for c in SELECTED_FEATURES]
                + [c for c in self.v_cleaner.top_features if c not in SELECTED_FEATURES]
            )

        X_final = X_final.reindex(columns=target_cols)

        # After reindex, categorical columns may be NaN or still carry their old
        # category dtype.  Converting via object first allows fillna(-1) to work
        # regardless of the current dtype state.
        for col in self.cat_encoders:
            if col in X_final.columns:
                # Go through float to allow fillna(-1) without triggering the
                # pandas FutureWarning about downcasting object arrays, then
                # convert to int→category for LightGBM.
                X_final[col] = (
                    X_final[col]
                    .astype(float)
                    .fillna(-1)
                    .astype(int)
                    .astype('category')
                )

        return X_final
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info('Loading data to test pipeline...')
    try:
        df_test = pd.read_csv('data/raw/train_transaction.csv', nrows=1000)
        test_pipeline = FraudPipeline()
        test_pipeline.fit(df_test)
        X_processed = test_pipeline.transform(df_test)
        logger.info(f'Input shape: {df_test.shape}')
        logger.info(f'Output shape: {X_processed.shape}')
        logger.info(f'Feature count: {len(X_processed.columns)} — {X_processed.columns.tolist()[:5]} ...')
    except FileNotFoundError:
        logger.error('File not found — check training data file path.')