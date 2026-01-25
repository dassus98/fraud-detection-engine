import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import joblib

logging.basicConfig(level = logging.INFO, format = '%(message)s')

def plot_feature_importance(model_path = 'models/lgbm_model_v1.pkl'):
    logging.info(f'Loading model from {model_path}...')
    model_wrapper = joblib.load(model_path)
    model = model_wrapper.model

    importances = model.feature_importance(importance_type = 'gain')
    feature_names = model.feature_name()

    df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending = False)

    logging.info('Top features by gain:')
    logging.info(df.head(10))

    graph_features = [feat for feat in feature_names if 'graph' in feat]
    logging.info('Graph feature ranking:')
    for feat in graph_features:
        rank = df[df['Feature'] == feat].index[0]
        score = df[df['Feature'] == feat]['Importance'].values[0]
        logging.info(f' - {feat}: Rank {rank} (Score: {score:.2f})')

if __name__ == '__main__':
    plot_feature_importance()