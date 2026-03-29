import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_feature_importance(model_path = 'models/lgbm_model_v1.pkl'):
    logger.info(f'Loading model from {model_path}...')
    model_wrapper = joblib.load(model_path)
    model = model_wrapper.model

    importances = model.feature_importance(importance_type = 'gain')
    feature_names = model.feature_name()

    df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending = False)

    logger.info('Top features by gain:')
    logger.info(df.head(10))

    graph_features = [feat for feat in feature_names if 'graph' in feat]
    logger.info('Graph feature ranking:')
    for feat in graph_features:
        rank = df[df['Feature'] == feat].index[0]
        score = df[df['Feature'] == feat]['Importance'].values[0]
        logger.info(f' - {feat}: Rank {rank} (Score: {score:.2f})')

if __name__ == '__main__':
    plot_feature_importance()