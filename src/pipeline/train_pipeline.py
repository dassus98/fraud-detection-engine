import pandas as pd
import joblib
import logging
import os
from src.features.pipeline import FeaturePipeline
from src.models.lightgbm_model import FraudModel
from src.evaluation.stratified import StratifiedEvaluator

os.makedirs('logs', exist_ok = True)
logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s',
                    handlers = [
                        logging.FileHandler('logs/train_run.log'),
                        logging.StreamHandler()
                    ],
                    force = True)

def run_training_pipeline(data_path, model_save_path):
    """
    Docstring for run_training_pipeline
    
    :param data_path: Description
    :param model_save_path: Description
    """

    logging.info('Loading raw data...')
    
    # Loading csvs right now but ideally would be loading parquet files to save time
    train_transaction = pd.read_csv('data/raw/train_transaction.csv')
    train_identity = pd.read_csv('data/raw/train_identity.csv')
    df = pd.merge(train_transaction, train_identity, on = 'TransactionID', how = 'left')

    # Feature engineering
    logging.info('Running feature engineering pipeline...')
    pipeline = FeaturePipeline()
    df = pipeline.run(df)

    # Splitting data by time (val size = 20%)
    df = df.sort_values('TransactionDT')
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    # Defining targets, dropping columns
    target = 'isFraud'
    drop_cols = [target, 'TransactionID', 'TransactionDT']

    # Defining feature columns
    features = [col for col in df.columns if col not in drop_cols]
    numeric_features = df[features].select_dtypes(include = ['number', 'bool']).columns.tolist()

    X_train = train_df[numeric_features]
    y_train = train_df[target]
    X_val = val_df[numeric_features]
    y_val = val_df[target]

    logging.info(f'Shapes: X_train = {X_train.shape}, y_train = {y_train.shape}, X_val = {X_val.shape}, y_val = {y_val.shape}')

    # Training model
    logging.info('Initializing LightGBM model...')
    model = FraudModel()
    model.train(X_train, y_train, X_val, y_val)

    # Evaluating model
    metrics = model.evaluate(X_val, y_val)
    logging.info(f'Final metrics: {metrics}')
    
    logging.info('Stratified analysis...')
    strat_eval = StratifiedEvaluator(model, threshold = metrics['optimal_threshold'])
    strat_eval.analyze(val_df)

    # Saving artifacts
    logging.info(f'Saving model to {model_save_path}')
    os.makedirs(os.path.dirname(model_save_path), exist_ok = True)
    joblib.dump(model, model_save_path)
    logging.info('Pipeline is complete.')

if __name__ == '__main__':
    if not os.path.exists('data/raw/train_transaction.csv'):
        print('ERROR: Data path not found.')
    else:
        run_training_pipeline(
            data_path = 'data/raw/',
            model_save_path = 'models/lgbm_model_v1.pkl'
        )