import pandas as pd
import joblib
import logging
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.features.pipeline import FeaturePipeline

# Initializing API
app = FastAPI(title = 'Fraud Detection Engine', version = '1.0.0')
logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Loading artifacts
try:
    model = joblib.load('models/lgbm_model_v1.pkl')
    pipeline = FeaturePipeline()
    logging.info('Model and Pipeline loaded succesfully.')
except Exception as e:
    logging.error(f'Failed to load artifacts. {e}')
    model = None
    pipeline = None

# Defining input schema
class TransactionRequest(BaseModel):
    TransactionID: int
    TransactionDT: int
    TransactionAmt: float
    card1: int
    P_emaildomain: str = None
    R_emaildomain: str = None
    DeviceInfo: str = None
    id_01: float = None

# Defining endpoints
@app.get('/health')
def health_check():
    """Checks to see if server is alive"""
    if model is None:
        raise HTTPException(status_code = 503, detail = 'Model load failed.')
    return {'status': 'healthy', 'model_version': 'v1'}

@app.post('/predict')
def predict(txn: TransactionRequest):
    """
    Receives JSON file, converts to df, runs pipeline.
    """
    if not model:
        raise HTTPException(status_code = 503, detail = 'Model load failed')
    
    try:
        # Convert JSON to DataFrame
        data_dict = {k: [v] for k, v in txn.dict().items()}
        df = pd.DataFrame(data_dict)

        # Run pipeline
        df_processed = pipeline.run(df)

        try:
            model_features = model.model.feature_name()
            logging.info(f"Model expects {len(model_features)} features.")
            
            # Add missing columns with 0
            for col in model_features:
                if col not in df_processed.columns:
                    df_processed[col] = 0
            
            # Reorder exactly to match model
            X_input = df_processed[model_features]
        except Exception as e:
            logging.warning(f'Could not retrieve feature names from model ({e}). Fallback to numerics.')
            X_input = df_processed.select_dtypes(include=['number', 'bool'])
        
        prob = model.predict(X_input)[0]
        decision = model.predict_decision(X_input)[0]

        return {
            'transaction_id': txn.TransactionID,
            'fraud_probability': float(prob),
            'decision': "BLOCK" if decision == 1 else 'ALLOW',
            'decision_reason': 'High Risk (> Threshold)' if decision == 1 else 'Low Risk'
        }
    except Exception as e:
        logging.error(f'Prediction Error: {e}')
        logging.error(traceback.format_exc())
        raise HTTPException(status_code = 500, detail = str(e))
    
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host = '0.0.0.0', port = 8000)
    