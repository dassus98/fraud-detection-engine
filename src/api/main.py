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
from src.utils.redis_client import RedisClient

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

# Redis feature store — initialized once at module load; degrades gracefully
# if Redis is not running (get_card_features returns {} in that case).
redis_store = RedisClient()

class TransactionRequest(BaseModel):
    """Pydantic schema documenting core fields; extra fields are forwarded as-is."""
    TransactionID: int = None
    TransactionAmt: float
    ProductCD: str
    card1: int
    card2: float = None

    class Config:
        # Permit extra fields to be added without validation errors
        extra = 'allow'

@app.on_event('startup')
def load_artifacts():
    """Load model and pipeline once at startup to avoid per-request disk I/O."""

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
    """Return 200 if the server is up and artifacts are loaded, 503 otherwise."""
    if artifacts['model'] is None or artifacts['pipeline'] is None:
        raise HTTPException(status_code=503, detail='Artifacts not loaded.')
    return {
        'status': 'healthy',
        'model_version': 'v1',
        'redis_available': redis_store.available,
    }

@app.post('/predict')
def predict(txn: TransactionRequest):
    """Run the feature pipeline and return a fraud score with a BLOCK/ALLOW decision."""
    if not artifacts['model'] or not artifacts['pipeline']:
        raise HTTPException(status_code=503, detail='Artifacts not loaded')
    try:
        input_data = txn.dict()
        df = pd.DataFrame([input_data])

        # Enrich with real-time per-card aggregates from Redis.
        # card1 is used as the card identifier; returns {} when Redis is down
        # or the card has no history, so the pipeline always receives at least
        # the base transaction fields.
        card_features = redis_store.get_card_features(input_data.get('card1'))
        if card_features:
            for col, val in card_features.items():
                df[col] = val

        # Run pipeline
        df_processed = artifacts['pipeline'].transform(df)

        # Predict probs and decision
        prob = artifacts['model'].predict(df_processed)[0]
        decision = artifacts['model'].predict_decision(df_processed)[0]
        threshold = getattr(artifacts['model'], 'optimal_threshold', 0.5)

        # Update card aggregates so the next prediction for this card sees
        # the current transaction included in its history.
        redis_store.update_card_features(
            card_id=input_data.get('card1'),
            txn_amount=input_data.get('TransactionAmt', 0.0),
        )

        return {
            'transaction_id': input_data.get('TransactionID', 'N/A'),
            'fraud_probability': float(prob),
            'decision': "BLOCK" if decision == 1 else 'ALLOW',
            'decision_reason': 'High Risk (> Threshold)' if decision == 1 else 'Low Risk',
            'threshold_used': float(threshold),
            'redis_enriched': bool(card_features),
        }

    except Exception as e:
        logger.error(f'Prediction Error: {e}')
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host = '0.0.0.0', port = 8000)
