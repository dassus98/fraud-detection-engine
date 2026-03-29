import pandas as pd
import numpy as np
import optuna
import lightgbm as lgb
import json
import logging
import os
import gc
from src.pipeline import FraudPipeline
from sklearn.metrics import roc_auc_score
from src.config import DATA_PATH, PARAMS_PATH
from src.utils import reduce_mem_usage
from optuna.integration import LightGBMPruningCallback

logger = logging.getLogger(__name__)

# Defining global var at modular level for safety
global_df = None

def objective(trial):
    """
    Optuna objective function. Optimizes LightGBM hyperparameters.
    """

    # Declaring usage of global var
    global global_df

    param = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_jobs': -1,
        'random_state': 42,
        
        # Tree structure
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        
        # Learning speed
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        
        # Regularization
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
    }

    n_estimators = trial.suggest_int('n_estimators', 500, 3000)

    if global_df is None:
        logger.info('Loading data for tuning...')
        df_temp = pd.read_csv(DATA_PATH)
        df_temp = reduce_mem_usage(df_temp)

        logger.info('Sorting by time for validation split...')
        global_df = df_temp.sort_values('TransactionDT').reset_index(drop=True)
        del df_temp
        gc.collect()

    split_idx = int(len(global_df) * 0.8)
    train_df = global_df.iloc[:split_idx]
    val_df = global_df.iloc[split_idx:]

    pipeline = FraudPipeline()
    X_train = pipeline.fit_transform(train_df)
    X_val = pipeline.transform(val_df)
    y_train = train_df['isFraud']
    y_val = val_df['isFraud']

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        param,
        train_data,
        num_boost_round=n_estimators,
        valid_sets = [val_data],
        callbacks = [
            lgb.early_stopping(stopping_rounds=50), 
            lgb.log_evaluation(period=0)
            ]
    )

    preds = model.predict(X_val)
    auc = roc_auc_score(y_val, preds)

    return auc

def run_tuning():
    logger.info('--- Starting Hyperparameter Tuning ---')

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    logger.info('--- Tuning Complete ---')
    logger.info(f'Best AUC: {study.best_value}')
    logger.info(f'Best Params: {study.best_params}')

    # Saving best params
    os.makedirs(os.path.dirname(PARAMS_PATH), exist_ok=True)
    with open(PARAMS_PATH, 'w') as f:
        json.dump(study.best_params, f, indent=4)
    logger.info(f'Saved best params to: {PARAMS_PATH}')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    run_tuning()