import os
import logging
import traceback

import pandas as pd
import joblib
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import MODEL_SAVE_PATH, PIPELINE_SAVE_PATH
from src.pipeline import FraudPipeline
from src.models.fraud_model import FraudModel
from src.utils.redis_client import RedisClient

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title='Fraud Detection Engine',
    description='Real-time fraud detection API using LightGBM',
    version='1.0.0',
)

# All ML artifacts live here; populated once at startup.
artifacts: dict = {
    'model': None,
    'pipeline': None,
    'explainer': None,   # shap.TreeExplainer — None if SHAP unavailable or init failed
}

# Redis feature store — degrades gracefully when Redis is not running.
redis_store = RedisClient()


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class TransactionRequest(BaseModel):
    """Core transaction fields; any additional model features can be passed as extras."""
    TransactionID: int = None
    TransactionAmt: float
    ProductCD: str
    card1: int
    card2: float = None

    class Config:
        extra = 'allow'


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event('startup')
def load_artifacts():
    """Load model, pipeline, and SHAP explainer once at startup."""
    logger.info('Loading ML artifacts...')

    for path, label in [(MODEL_SAVE_PATH, 'model'), (PIPELINE_SAVE_PATH, 'pipeline')]:
        if not os.path.exists(path):
            logger.error(f'Artifact not found at {path} — run training before starting the API.')
            return

    try:
        artifacts['pipeline'] = joblib.load(PIPELINE_SAVE_PATH)
        logger.info(f'Pipeline loaded from {PIPELINE_SAVE_PATH}')

        artifacts['model'] = joblib.load(MODEL_SAVE_PATH)
        logger.info(f'Model loaded from {MODEL_SAVE_PATH} | threshold={artifacts["model"].optimal_threshold:.3f}')
    except Exception:
        logger.error(f'Failed to load artifacts:\n{traceback.format_exc()}')
        return

    if _SHAP_AVAILABLE:
        try:
            booster = artifacts['model'].model   # underlying lgb.Booster
            artifacts['explainer'] = shap.TreeExplainer(booster)
            logger.info('SHAP TreeExplainer initialised.')
        except Exception:
            logger.warning(f'SHAP explainer init failed — explanations will be unavailable:\n{traceback.format_exc()}')
    else:
        logger.warning('shap package not installed — explanations will be unavailable.')


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get('/health')
def health_check():
    """Return 200 if artifacts are loaded; 503 otherwise."""
    if artifacts['model'] is None or artifacts['pipeline'] is None:
        raise HTTPException(status_code=503, detail='Artifacts not loaded.')
    return {
        'status': 'healthy',
        'model_loaded': True,
        'shap_available': artifacts['explainer'] is not None,
        'redis_available': redis_store.available,
    }


@app.post('/predict')
def predict(txn: TransactionRequest):
    """Score a transaction and return probability, decision, threshold, and top SHAP drivers."""
    if artifacts['model'] is None or artifacts['pipeline'] is None:
        raise HTTPException(status_code=503, detail='Artifacts not loaded.')

    input_data = txn.dict()
    txn_id = input_data.get('TransactionID', 'N/A')
    logger.info(f'Scoring transaction {txn_id}')

    try:
        df = pd.DataFrame([input_data])

        # --- Redis enrichment -----------------------------------------------
        card_id = input_data.get('card1')
        card_features = redis_store.get_card_features(card_id)
        if card_features:
            for col, val in card_features.items():
                df[col] = val
            logger.info(f'Redis enriched transaction {txn_id} with {list(card_features.keys())}')

        # --- Feature pipeline ------------------------------------------------
        df_processed = artifacts['pipeline'].transform(df)

        # --- Scoring ---------------------------------------------------------
        proba = float(artifacts['model'].predict(df_processed)[0])
        threshold = float(artifacts['model'].optimal_threshold)
        decision = 'BLOCK' if proba >= threshold else 'ALLOW'

        logger.info(f'Transaction {txn_id}: prob={proba:.4f} threshold={threshold:.3f} → {decision}')

        # --- SHAP top-3 contributions ----------------------------------------
        top_features = _compute_top_shap(df_processed, n=3)

        # --- Update Redis with this transaction's data -----------------------
        redis_store.update_card_features(
            card_id=card_id,
            txn_amount=input_data.get('TransactionAmt', 0.0),
        )

        return {
            'transaction_id': txn_id,
            'fraud_probability': round(proba, 6),
            'decision': decision,
            'threshold_used': round(threshold, 4),
            'redis_enriched': bool(card_features),
            'top_shap_features': top_features,
        }

    except Exception:
        logger.error(f'Prediction failed for transaction {txn_id}:\n{traceback.format_exc()}')
        raise HTTPException(status_code=500, detail='Internal scoring error — see server logs.')


# ---------------------------------------------------------------------------
# SHAP helper
# ---------------------------------------------------------------------------

def _compute_top_shap(df_processed: pd.DataFrame, n: int = 3) -> list[dict]:
    """
    Return the top-n features by absolute SHAP value for a single-row DataFrame.
    Returns an empty list if the explainer is unavailable or computation fails.
    """
    explainer = artifacts['explainer']
    if explainer is None:
        return []

    try:
        shap_values = explainer.shap_values(df_processed)

        # TreeExplainer on a binary LightGBM booster returns a single 2-D array;
        # for multi-output it returns a list — take the positive-class slice.
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        row = shap_values[0]                  # shape: (n_features,)
        feature_names = df_processed.columns.tolist()

        ranked = sorted(
            zip(feature_names, row.tolist()),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        return [
            {'feature': feat, 'shap_contribution': round(val, 6)}
            for feat, val in ranked[:n]
        ]
    except Exception:
        logger.warning(f'SHAP computation failed:\n{traceback.format_exc()}')
        return []


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
