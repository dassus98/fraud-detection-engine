import pandas as pd
import joblib
import logging
import gc
import os
import lightgbm as lgb
from src.features.pipeline import FeaturePipeline
from src.pipeline.pipeline import FraudPipeline
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

RAW_DATA_PATH = 'data/raw/train_transaction.csv'
MODEL_SAVE_PATH = 'models/fraud_model_v1.pkl'
PIPELINE_SAVE_PATH = 'models/pipeline_v1.pkl'

SELECTED_FEATURES = [
    'TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
    'addr1', 'addr2', 'dist1', 'dist2', 'P_emaildomain', 'R_emaildomain',
    'C1', 'C2', 'C5', 'C8', 'C9', 'C12', # Choosing Cs with >0.30 correlation with Fraud
    'D1', 'D3', 'D4', 'D5', 'D8', 'D10', 'D11', 'D13', 'D14', 'D15', # Choosing non-collinear D variables
    'M1', 'M4', 'M5', 'M6', 'M7',
]

def train_production_model():
    print('--- STARTING TRAIN ---')

    path = RAW_DATA_PATH if os.path.exists(RAW_DATA_PATH) else print('Check raw data path.')
    print(f'RETRIEVING DATA FROM {path}...')
    df = pd.read_csv(path)

    X = df.drop(columns = ['isFraud', 'TransactionID'])
    y = df['isFraud']

    del df
    gc.collect()

    print(f'RUNNING PIPELINE WITH {len(SELECTED_FEATURES)} FEATURES')
    pipeline = FraudPipeline(selected_features=SELECTED_FEATURES, v_threshold=0.90)

    X_processed = pipeline.fit_transform(X)
    print(f'Final Shape: {X_processed.shape}')

    split_idx = int(len(X_processed) * 0.80)
    X_train, X_val = X_processed.iloc[:split_idx], X_processed.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    print('TRAINING LGM MODEL')

    clf = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.02,
        num_leaves=30,
        max_depth=10,
        subsample=0.75,
        colsample_bytree=0.75,
        metric='auc',
        n_jobs=-1,
        random_state=42
    )

    cat_cols = X_processed.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        for col in cat_cols:
            X_train[col] = X_train[col].astype('category')
            X_val[col] = X_val[col].astype('category')
        
    clf.fit(
        X_train, y_train,
        eval_set = [(X_val, y_val)],
        callbacks = [
            lgb.early_stopping(100),
            lgb.log_evaluation(100)
        ]
    )

    print('\n--- RESULTS ---')
    val_preds = clf.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, val_preds)
    precision, recall, _ = precision_recall_curve(y_val, val_preds)
    pr_auc = auc(recall, precision)

    print(f'ROC-AUC: {auc_score:.4f}')
    print(f'PR AUC: {pr_auc:.4f}')

    print('SAVING MODEL')
    os.makedirs('models', exist_ok=True)
    joblib.dump(clf, MODEL_SAVE_PATH)
    joblib.dump(pipeline, PIPELINE_SAVE_PATH)
    print(f'MODEL AND PIPELINE SUCCESSFULLY SAVED')

if __name__ == '__main__':
    train_production_model()