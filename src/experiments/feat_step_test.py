import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import gc

def get_feature_groups():
    """
    Docstring for get_feature_groups
    """

    # Setting a baseline for the most basic of features
    base_features = ['ProductCD', 'addr1', 'addr2', 'dist1', 'dist2', 'P_emaildomain', 'R_emaildomain'] + \
        [f'card{i}' for i in range(1,7)]
    
    c_cols = [f'C{i}' for i in range(1,15)]
    d_cols = [f'D{i}' for i in range(1,16)]
    m_cols = [f'M{i}' for i in range(1,10)]
    v_cols = [f'V{i}' for i in range(1,340)]

    return {
        'Base Features': base_features,
        'Counting Features': c_cols,
        'Timedelta Features': d_cols,
        'Match Features': m_cols,
        'Vesta Masked Features': v_cols
    }

def train_and_evaluate(df, features, stage_name):
    """
    Docstring for train_and_evaluate
    
    :param df: Description
    :param features: Description
    :param stage_name: Description
    """

    print(f'\n--- Training Stage: {stage_name} ---')
    print(f'Feature Count: {len(features)}')

    X = df[features].copy()
    y = df['isFraud']

    # Encoding categorical labels
    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Taking 80% of the data for training, remaining 20% for validation
    split_idx = int(len(X) * 0.80)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    # Training model
    model = lgb.LGBMClassifier(
        objective = 'binary',
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=30,
        n_jobs=-1,
        random_state=42
    )

    model.fit(
        X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='auc', callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )

    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    print(f'Stage AUC: {auc:.4f}')
    
    return auc

if __name__ == '__main__':
    # By running this, we should be able to see that the model improves as we add features
    # but the V-features add a lot of noise. It's best to clean up the v-features and remove
    # the ones a high collinearity.

    print('Loading data...')

    try:
        df = pd.read_csv('data/raw/train_transaction.csv')
    except FileNotFoundError:
        print('File has not been found. Check path.')

    groups = get_feature_groups()
    current_features = []
    results = {}

    for name, new_feats in groups.items():
        current_features += new_feats

        valid_feats = [feat for feat in current_features if feat in df.columns]

        score = train_and_evaluate(df, valid_feats, name)
        results[name] = score

        gc.collect()

    print('\n--- Final Results ---')
    print(f'{'Stage':<40} | {'AUC': <10} | {'Lift'}')
    print('-' * 65)

    prev_score = 0
    for name, score in results.items():
        lift = score - prev_score if prev_score > 0 else 0
        print(f'{name:<40} | {score:.4f} | {lift:+.4f}')
        prev_score = score
