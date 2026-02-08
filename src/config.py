import os

DATA_PATH = os.path.join('data', 'raw', 'train_transaction.csv')
MODEL_SAVE_PATH = os.path.join('models', 'fraud_model.pkl')
PIPELINE_SAVE_PATH = os.path.join('models', 'pipeline.pkl')
PARAMS_PATH = os.path.join('models', 'best_params.json')

DEFAULT_LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'n_jobs': -1,
    'learning_rate': 0.02,
    'num_leaves': 31,
    'max_depth': 8,
    'random_state': 42
}

SELECTED_FEATURES = [
    'TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
    'addr1', 'addr2', 'dist1', 'dist2', 'P_emaildomain', 'R_emaildomain',
    'C1', 'C2', 'C5', 'C8', 'C9', 'C12',
    'D1', 'D3', 'D4', 'D5', 'D8', 'D10', 'D11', 'D13', 'D14', 'D15',
    'M1', 'M4', 'M5', 'M6', 'M7'
]

DROP_COLS = ['isFraud', 'TransactionID', 'TransactionDT']