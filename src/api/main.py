import pandas as pd
import joblib
import uvicorn
import os
import logging
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import MODEL_SAVE_PATH, PIPELINE_SAVE_PATH
from src.models.fraud_model import FraudModel
from src.pipeline import FraudPipeline

# Initializing API
app = FastAPI(
    title = 'Fraud Detection Engine',
    description='Real-time fraud detection API using LightGBM',
    version = '1.0.0')

logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Loading artifacts
artifacts = {
    'model': None,
    'pipeline': None
}

class TransactionRequest(BaseModel):
    """
    Defining only a few key fields for documentation.
    """
    TransactionID: int = None
    TransactionAmt: float
    ProductCD: str
    card1: int
    card2: float = None

    class Config:
        # Permit extra fields to be added without validation errors
        extra = 'allow'

@app.on.event('startup')
def load_artifacts():
    """
    Loading model and pipeline on startup to prevent re-loading for every request.
    """

    logger.info('Loading artifacts...')

    try:
        if not os.path.exists(MODEL_SAVE_PATH) or not os.path.exists(PIPELINE_SAVE_PATH):
            logger.error("Artifacts not found, paths don't exist.")
            return
        
        artifacts['pipeline'] = joblib.load(PIPELINE_SAVE_PATH)
        artifacts['model'] = joblib.load(MODEL_SAVE_PATH)

        logger.info('--- Artifacts loaded successfully ---')

    except Exception as e:
        logger.error(f'ERROR: Failed to load artifacts. {e}')
        logger.error(traceback.format_exc())

@app.get('/health')
def health_check():
    """
    Check to see if server is up and artifacts are ready.
    """
    if artifacts['model'] is None or artifacts['pipeline'] is None:
        raise HTTPException(status_code=503, detail='Artifacts not loaded.')
    return {'status': 'healthy', 'model_version':'v1'}

@app.post('/predict')
def predict(txn: TransactionRequest):
    """
    Receives JSON file, runs pipeline, returns fraud score & decision.
    """
    if not artifacts['model'] or not artifacts['pipeline']:
        raise HTTPException(status_code=503, detail='Artifacts not loaded')
    try:
        # Convert JSON to df
        input_data = txn.dict()
        df = pd.DataFrame([input_data])

        # Run pipeline
        df_processed = artifacts['pipeline'].transform(df)

        # Predict probs and decision
        prob = artifacts['model'].predict(df_processed)[0]
        decision = artifacts['model'].predict_decision(df_processed)[0]
        threshold = getattr(artifacts['model'], 'optimal_threshold', 0.5)

        return {
            'transaction_id': input_data.get('TransactionID', 'N/A'),
            'fraud_probability': float(prob),
            'decision': "BLOCK" if decision == 1 else 'ALLOW',
            'decision_reason': 'High Risk (> Threshold)' if decision == 1 else 'Low Risk',
            'threshold_used': float(threshold)
        }
    
    except Exception as e:
        logger.error(f'Prediction Error: {e}')
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == '__main__':
    uvicorn.run(app, host = '0.0.0.0', port = 8000)
    