import pandas as pd
import logging
from src.features.basic import BasicFeatureEngineer
from src.features.behavioral import BehavioralFeatureEngineer
from src.features.velocity import VelocityFeatureEngineer

# Setting up logging
logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')

class FeaturePipeline:
    """
    Docstring for FeaturePipeline
    """

    def __init__(self):
        self.basic_engineer = BasicFeatureEngineer()
        self.behavioral_engineer = BehavioralFeatureEngineer(key_entity = 'card1')
        self.velocity_engineer = VelocityFeatureEngineer(time_col = 'TransactionDT', key_col = 'card1')

    def run(self, df):
        """
        Docstring for run
        
        :param self: Description
        :param df: Description
        """

        logging.info(f'Starting feature pipeline on {len(df)} records...')

        # Tier 1 & 2 calculations
        df = self.basic_engineer.fit_transform(df)
        logging.info('Tier 1 & 2 calculations are complete.')

        # Tier 3 calculations
        df = self.behavioral_engineer.fit_transform(df)
        logging.info('Tier 3 calculations are complete.')

        # Tier 4 calculations
        df = self.velocity_engineer.fit_transform(df)
        logging.info('Tier 4 calculations are complete.')

        # Tier 5 calculations
        # GRAPH CALCULATIONS, ADD LATER

        logging.info(f'Pipeline complete. Final shale: {df.shape}')
        return df
    
if __name__ == '__main__':
    # Testing if the pipeline works.
    # Creating dummy dataset to verify the correct flow of data within the pipeline.
    dummy_data = {
        'TransactionID': range(5),
        'TransactionDT': [100, 200, 250, 80000, 80100], # Seconds
        'TransactionAmt': [50.0, 50.0, 500.0, 20.0, 20.0],
        'card1': [1000, 1000, 1000, 2000, 2000],
        'P_emaildomain': ['gmail.com', 'gmail.com', 'yahoo.com', 'hotmail.com', 'hotmail.com'],
        'R_emaildomain': ['yahoo.com', None, None, 'gmail.com', None],
        'id_01': [-5.0, -5.0, -10.0, None, None]
    }

    df_test = pd.DataFrame(dummy_data)

    pipeline = FeaturePipeline()
    df_final = pipeline.run(df_test)

    print('\nFinal Columns: ', df_final.columns.tolist())
    print('\nVelocity: \n', df_final[['TransactionDT', 'card1', 'time_since_last_txn']])