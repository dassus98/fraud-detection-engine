import pandas as pd

class AggregationFeatureEngineer:
    """
    Docstring for AggregationFeatureEngineer
    """

    def __init__(self):
        pass

    def transform(self, df):
        """
        Docstring for transform
        
        :param self: Description
        :param df: Description
        """

        df = df.copy()

        for col in ['card1', 'addr1', 'P_emaildomain']