import lightgbm as lgb
import pandas as pd
import numpy as np
import logging
from sklearn.metrics import roc_auc_score
from src.evaluation.metrics import calculate_economic_cost, find_optimal_threshold

logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')

class FraudModel:
    """
    Docstring for FraudModel
    """

    def __init__(self, params = None):
        # Using default hyperparameters
        self.params = params if params else {
            'objective': 'binary',
            'metric': 'auc',
            'is_unbalance': False,
            'scale_pos_weight': 9, # This should be tuned
            'num_leaves': 64, 
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'n_jobs': -1
        }
        self.model = None
        self.optimal_threshold = 0.5

    def train(self, X_train, y_train, X_val, y_val):
        logging.info('Training LightGBM model...')

        # Creating datasets
        train_data = lgb.Dataset(X_train, label = y_train)
        val_data = lgb.Dataset(X_val, label = y_val, reference = train_data)

        # Training with early stopping
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round = 1000,
            valid_sets = [train_data, val_data],
            callbacks = [lgb.early_stopping(stopping_rounds = 50), lgb.log_evaluation(50)]
        )

        logging.info('LightGBM training complete.')

    def evaluate(self, X_val, y_val):
        """
        Docstring for evaluate
        
        :param self: Description
        :param X_val: Description
        :param y_val: Description
        """

        logging.info('Evaluating business impact of the model...')

        y_pred_proba = self.model_predict(X_val)
        auc = roc_auc_score(y_val, y_pred_proba)
        logging.info(f'Validation AUC: {auc:.4f}')

        # Finding the threshold which saves the most money based on the economic function
        # built in metrics.py
        optimal_threshold, min_cost = find_optimal_threshold(
            y_val, y_pred_proba, cost_fn = 525, cost_fp = 150
        )

        self.optimal_threshold = optimal_threshold

        logging.info(f'Optimal Decision Threshold: {self.optimal_threshold:.3f}')
        logging.info(f'Minimum Expected Cost per Transaction: {min_cost:.2f}')

        return {
            'auc': auc,
            'optimal_threshold': self.optimal_threshold,
            'cost_per_txn': min_cost
        }
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_decision(self, X):
        # Returns 1 (block) or 0 (allow) based on economic threshold
        proba = self.model.predict(X)
        return (proba >= self.optimal_threshold).astype(int)