import pandas as pd
import joblib
import logging
import gc
import os
from sklearn.metrics import roc_auc_score, average_precision_score

from src.config import DATA_PATH, MODEL_SAVE_PATH, PIPELINE_SAVE_PATH
from src.pipeline import FraudPipeline
from src.models.fraud_model import FraudModel
from src.utils import reduce_mem_usage

# Setting up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_production_model():
    logger.info('--- Starting Production Training Run ---')
    if not os.path.exists(DATA_PATH):
        logger.error(f'Data path not found: {DATA_PATH}')
        return
    
    logger.info(f'Loading data...')
    df = pd.read_csv(DATA_PATH)
    df = reduce_mem_usage(df)

    logger.info('Sorting data by temporal split...')
    
    df = df.sort_values('TransactionDT').reset_index(drop=True)
    split_idx = int(len(df) * 0.8)

    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()
    y_train = train_df['isFraud']
    y_val = val_df['isFraud']

    # Cleaning memory
    del df
    gc.collect()

    logger.info(f'Train set: {train_df.shape}')
    logger.info(f'Val set: {val_df.shape}')
    logger.info(f'y train set: {y_train.shape}')
    logger.info(f'y val set: {y_val.shape}')

    logger.info('Fitting data on Train dataset...')
    
    pipeline = FraudPipeline()
    X_train_processed = pipeline.fit_transform(train_df)
    X_val_processed = pipeline.transform(val_df)

    # Saving pipeline
    os.makedirs(os.path.dirname(PIPELINE_SAVE_PATH), exist_ok=True)
    joblib.dump(pipeline, PIPELINE_SAVE_PATH)
    logger.info(f'Pipeline saved to: {PIPELINE_SAVE_PATH}')

    # Training model
    logger.info('Training LightGBM model with optimized params...')
    model = FraudModel()
    model.train(X_train_processed, y_train, X_val_processed, y_val)

    # Evaluate model
    val_preds = model.predict(X_val_processed)
    roc_auc = roc_auc_score(y_val, val_preds)
    pr_auc = average_precision_score(y_val, val_preds)

    logger.info('--- Final Results ---')
    logger.info(f'ROC-AUC: {roc_auc:.4f}')
    logger.info(f'PR-AUC: {pr_auc:.4f}')

    # Saving model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    logger.info(f'Model saved to {MODEL_SAVE_PATH}')

if __name__ == '__main__':
    train_production_model()