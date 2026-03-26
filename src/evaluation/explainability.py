import os
import logging
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import shap

from src.config import DATA_PATH, MODEL_SAVE_PATH, PIPELINE_SAVE_PATH
from src.models.fraud_model import FraudModel
from src.utils import reduce_mem_usage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REPORTS_DIR = 'reports'
PLOT_PATH = os.path.join(REPORTS_DIR, 'shap_summary.png')
RANKINGS_PATH = os.path.join(REPORTS_DIR, 'feature_rankings.csv')
TOP_N = 20


def _load_validation_set(pipeline):
    """Loads data, sorts temporally, takes the last 20%, and runs it through the pipeline."""
    logger.info(f'Loading data from {DATA_PATH}...')
    df = pd.read_csv(DATA_PATH)
    df = reduce_mem_usage(df)
    df = df.sort_values('TransactionDT').reset_index(drop=True)

    split_idx = int(len(df) * 0.8)
    val_df = df.iloc[split_idx:].copy()
    del df

    logger.info(f'Validation rows: {len(val_df)}')
    X_val = pipeline.transform(val_df)
    return X_val


def _plot_to_buffer(shap_values, X, plot_type):
    """Renders a SHAP summary plot to an in-memory PNG buffer."""
    shap.summary_plot(shap_values, X, plot_type=plot_type, max_display=TOP_N, show=False)
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close('all')
    buf.seek(0)
    return buf


def run_explainability():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Load artifacts
    logger.info(f'Loading model from {MODEL_SAVE_PATH}...')
    fraud_model = FraudModel.load(MODEL_SAVE_PATH)

    logger.info(f'Loading pipeline from {PIPELINE_SAVE_PATH}...')
    pipeline = joblib.load(PIPELINE_SAVE_PATH)

    X_val = _load_validation_set(pipeline)
    feature_names = X_val.columns.tolist()

    # Compute SHAP values against the native lgb.Booster
    logger.info('Computing SHAP values with TreeExplainer...')
    explainer = shap.TreeExplainer(fraud_model.model)
    shap_values = explainer.shap_values(X_val)  # shape: (n_samples, n_features)

    # Rank all features by mean absolute SHAP value
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    rankings = (
        pd.DataFrame({'feature': feature_names, 'mean_abs_shap': mean_abs_shap})
        .sort_values('mean_abs_shap', ascending=False)
        .reset_index(drop=True)
    )
    rankings.index += 1
    rankings.index.name = 'rank'
    rankings.to_csv(RANKINGS_PATH)
    logger.info(f'Feature rankings saved to {RANKINGS_PATH}')

    # Slice down to top 20 for plots
    top_features = rankings['feature'].head(TOP_N).tolist()
    top_idx = [feature_names.index(f) for f in top_features]
    shap_top = shap_values[:, top_idx]
    X_val_top = X_val[top_features]

    # Generate each plot into its own buffer (SHAP creates its own figure internally)
    logger.info('Generating SHAP plots...')
    buf_beeswarm = _plot_to_buffer(shap_top, X_val_top, plot_type='dot')
    buf_bar = _plot_to_buffer(shap_top, X_val_top, plot_type='bar')

    # Stitch the two plots side-by-side into a single output image
    img_beeswarm = mpimg.imread(buf_beeswarm)
    img_bar = mpimg.imread(buf_bar)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
    fig.suptitle(f'SHAP Feature Importance — Top {TOP_N} Features', fontsize=14, fontweight='bold')

    ax1.imshow(img_beeswarm)
    ax1.set_title('Summary Plot (Beeswarm)', fontsize=12)
    ax1.axis('off')

    ax2.imshow(img_bar)
    ax2.set_title('Mean |SHAP Value| (Bar)', fontsize=12)
    ax2.axis('off')

    plt.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'SHAP plots saved to {PLOT_PATH}')

    logger.info(f'--- Top {TOP_N} Features by Mean |SHAP| ---')
    logger.info(f'\n{rankings.head(TOP_N).to_string()}')


if __name__ == '__main__':
    run_explainability()
