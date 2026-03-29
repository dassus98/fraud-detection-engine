"""
Feature importance report using LightGBM's built-in gain metric.

Run as a standalone script after training:
    python -m src.evaluation.importance

Output is logged to stdout; the top-10 features by gain are printed in
descending order.  'Gain' measures how much each feature reduces prediction
error across all the trees that use it — a higher score means the feature
carries more predictive weight.
"""

import logging

import joblib
import pandas as pd

from src.config import MODEL_SAVE_PATH

# Module-level logger; basicConfig is set by the entry point (__main__ block
# below), not here, so importing this module never reconfigures the root logger.
logger = logging.getLogger(__name__)


def plot_feature_importance(model_path: str = MODEL_SAVE_PATH) -> pd.DataFrame:
    """
    Load a trained FraudModel and log its top features by LightGBM gain.

    Args:
        model_path: Path to a joblib-serialised FraudModel.  Defaults to the
                    production model path defined in src/config.py.

    Returns:
        DataFrame with columns ['Feature', 'Importance'] sorted descending.
    """
    logger.info(f'Loading model from {model_path}...')
    model_wrapper = joblib.load(model_path)

    # model_wrapper is a FraudModel instance; .model is the underlying lgb.Booster.
    booster = model_wrapper.model

    importances  = booster.feature_importance(importance_type='gain')
    feature_names = booster.feature_name()

    df = (
        pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        .sort_values('Importance', ascending=False)
        .reset_index(drop=True)
    )

    logger.info('Top 10 features by gain:')
    logger.info('\n' + df.head(10).to_string(index=False))

    return df


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    plot_feature_importance()
