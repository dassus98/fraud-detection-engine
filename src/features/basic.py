import pandas as pd
import numpy as np

class BasicFeatureEngineer:
    def __init__(self):
        pass
    
    def fit_transform(self, df):
        """
        Applies Tier 1 (Basic) and Tier 2 (Aggregations) feature engineering.
        Splitting features into tiers (stateless & stateful) for latency and ease of debugging.
        """
        df = df.copy()
        
        # Tier 1: Basic & stateless transformations
        print("Creating Tier 1 Features...")
        
        # Transforming TransactionAmt
        # Adding 1 to avoid log(0) errors
        df['TransactionAmt_Log'] = np.log1p(df['TransactionAmt'])
        
        # Extracting email domains
        # P_emaildomain = Purchaser, R_emaildomain = Recipient
        # Splitting 'gmail.com' -> 'gmail' and 'com'
        for col in ['P_emaildomain', 'R_emaildomain']:
            df[col + '_prefix'] = df[col].astype(str).str.split('.').str[0]
            df[col + '_suffix'] = df[col].astype(str).str.split('.').str[-1]
            
        # Flagging missing values
        # Important for trees to know if a value was explicitly missing
        df['no_identity_info'] = df['id_01'].isnull().astype('int8')
        
        # Tier 2: Aggregations (stateful features)
        print("Creating Tier 2 Features...")
        
        # Velocity features
        # How many times has this card been seen?
        # card1 = Card issuer identification number (rough proxy for unique card)
        df['card1_count_full'] = df.groupby('card1')['TransactionID'].transform('count')
        
        # Historical spending balances
        # What is the average spend for this card?
        df['card1_amt_mean'] = df.groupby('card1')['TransactionAmt'].transform('mean')
        df['card1_amt_std'] = df.groupby('card1')['TransactionAmt'].transform('std')
        
        # Deviation from mean
        # Is this specific transaction larger than usual for this card?
        df['TransactionAmt_to_mean_card1'] = df['TransactionAmt'] / df['card1_amt_mean']
        
        return df

if __name__ == "__main__":
    # Test block to verify it works
    # Create a dummy dataframe
    data = {
        'TransactionID': [1, 2, 3, 4],
        'TransactionAmt': [100.0, 50.0, 100.0, 5000.0],
        'card1': [1000, 1000, 1000, 1000],
        'P_emaildomain': ['gmail.com', 'yahoo.com', 'gmail.com', None],
        'R_emaildomain': [None, None, 'hotmail.com', 'gmail.com'],
        'id_01': [-5.0, -5.0, None, -10.0]
    }
    df_test = pd.DataFrame(data)
    
    engineer = BasicFeatureEngineer()
    df_transformed = engineer.fit_transform(df_test)
    
    print("\nTransformed Columns:")
    print(df_transformed.columns.tolist())
    print("\nExample Feature (TransactionAmt_to_mean_card1):")
    print(df_transformed[['TransactionAmt', 'card1_amt_mean', 'TransactionAmt_to_mean_card1']])