import lightgbm as lgb
import pandas as pd
import numpy as np
import logging
import json
import joblib
import os
from sklearn.metrics import roc_auc_score, average_precision_score

from src.evaluation.metrics import calculate_economic_cost, find_optimal_threshold
from src.config import PARAMS_PATH

logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FraudModel:
    """
    Wrapper for LightGBM model. Automates hyperparameter tuning and finds optimal threshold based on economic cost function.
    """

    def __init__(self, params = None):
        # Using default hyperparameters
        self.model = None
        self.optimal_threshold = 0.5
        self.params = self._load_params(params)

    def _load_params(self, custom_params):
        """
        Retrieves params from best_params.json.
        """
        if custom_params:
            return custom_params
        
        if os.path.exists(PARAMS_PATH):
            logger.info(f'Loading optimized hyperparameters from {PARAMS_PATH}')
            with open(PARAMS_PATH, 'r') as f:
                loaded_params = json.load(f)
                loaded_params['objective'] = 'binary'
                loaded_params['metric'] = 'auc'
                loaded_params['n_jobs'] = -1
                loaded_params['verbose'] = -1
                return loaded_params
        
        logger.warning(f'Params not found at {PARAMS_PATH}. Using default values instead.')
        
        return {
            'objective': 'binary',
            'metric': 'auc',
            'learning_rate': 0.025,
            'num_leaves': 31,
            'n_jobs': -1,
            'verbose': -1
        }
    
    def train(self, X_train, y_train, X_val, y_val):
        logger.info('Training LightGBM model...')

        # Creating datasets
        train_data = lgb.Dataset(X_train, label = y_train)
        val_data = lgb.Dataset(X_val, label = y_val, reference = train_data)

        # Training with early stopping
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round = 3000,
            valid_sets = [train_data, val_data],
            callbacks = [lgb.early_stopping(stopping_rounds = 100), lgb.log_evaluation(100)]
        )

        logger.info('LightGBM training complete.')
        return self.model

    def evaluate(self, X_val, y_val):
        """
        Calculates ROC-AUC, PR-AUC and business cost.
        """

        logger.info('Evaluating business impact of the model...')

        y_pred_proba = self.model.predict(X_val)
        roc_auc = roc_auc_score(y_val, y_pred_proba)
        pr_auc = average_precision_score(y_val, y_pred_proba)
        logger.info(f'Validation ROC-AUC: {roc_auc:.4f}')
        logger.info(f'Validation PR-AUC: {pr_auc}')

        # Finding the threshold which saves the most money based on the economic function.
        optimal_threshold, min_cost = find_optimal_threshold(
            y_val, y_pred_proba, cost_fn = 525, cost_fp = 150
        )

        self.optimal_threshold = optimal_threshold

        logger.info(f'Optimal Decision Threshold: {self.optimal_threshold:.3f}')
        logger.info(f'Minimum Expected Cost per Transaction: {min_cost:.2f}')

        return {
            'auc': roc_auc,
            'pr_auc': pr_auc,
            'optimal_threshold': self.optimal_threshold,
            'cost_per_txn': min_cost
        }
    
    def predict(self, X):
        # Returns only the raw probs
        return self.model.predict(X)
    
    def predict_decision(self, X):
        # Returns 1 (block) or 0 (allow) based on economic threshold
        proba = self.model.predict(X)
        return (proba >= self.optimal_threshold).astype(int)
    
    def save(self, path):
        joblib.dump(self, path)
        logger.info(f'Fraud model has been saved to {path}')

    @staticmethod
    def load(path):
        return joblib.load(path)