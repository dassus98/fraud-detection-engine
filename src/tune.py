import pandas as pd
import numpy as np
import optuna
import lightgbm as lgb
import json
import os
import gc
from src.pipeline import FraudPipeline
from sklearn.metrics import roc_auc_score
from optuna.integration import LightGBMPruningCallback

RAW_DATA_PATH = 'data/raw/train_transaction.csv'
OUTPUT_PATH = 'models/best_params.json'

SELECTED_FEATURES = [
    'TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
    'addr1', 'addr2', 'dist1', 'dist2', 'P_emaildomain', 'R_emaildomain',
    'C1', 'C2', 'C5', 'C8', 'C9', 'C12', # Choosing Cs with >0.30 correlation with Fraud
    'D1', 'D3', 'D4', 'D5', 'D8', 'D10', 'D11', 'D13', 'D14', 'D15', # Choosing non-collinear D variables
    'M1', 'M4', 'M5', 'M6', 'M7',
]

def objective(trial):
    """
    Docstring for objective
    
    :param trial: Description
    """

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
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
        
        # Regularization
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
    }

    dtrain = lgb.Dataset(X_train, label = y_train)
    dval = lgb.Dataset(X_val, label = y_val, reference = dtrain)
    pruning_callback = LightGBMPruningCallback(trial, 'auc')

    model = lgb.train(
        param,
        dtrain,
        valid_sets = [dval],
        callbacks = [pruning_callback]
    )

    preds = model.predict(X_val)
    auc = roc_auc_score(y_val, preds)

    return auc

if __name__ == '__main__':
    print('--- STARTING MODEL OPTIMIZATION ---')

    path = RAW_DATA_PATH if os.path.exists(RAW_DATA_PATH) else print('Raw path error.')

    df = pd.read_csv(path)

    X = df.drop(columns = ['isFraud', 'TransactionID'])
    y = df['isFraud']
    del df
    gc.collect()

    print('Running pipeline...')
    pipeline = FraudPipeline(selected_features=SELECTED_FEATURES, v_threshold=0.90)
    X_processed = pipeline.fit_transform(X)

    cat_cols = X_processed.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols:
        X_processed[col] = X_processed[col].astype('category')

    split_idx = int(len(X_processed) * 0.8)
    X_train, X_val = X_processed.iloc[:split_idx], X_processed.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f'Training shape: {X_train.shape}')

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    print('\n--- OPTIMIZATION COMPLETE ---')
    print(f'Best AUC: {study.best_value:.4f}')
    print('Best params:')
    for key, value in study.best_params.items():
        print(f'{key}: {value}')

    os.makedirs('models', exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(study.best_params, f, indent=4)
    
    print(f'Best parameters saved to {OUTPUT_PATH}')