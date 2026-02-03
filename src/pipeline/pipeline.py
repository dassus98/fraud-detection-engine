import pandas as pd
import gc
from src.features.v_features import VFeatureCleaner

class FraudPipeline:
    def __init__(self, selected_features = None, v_threshold = 0.90):
        """
        Docstring for __init__
        
        :param self: Description
        :param selected_features: Description
        :param v_threshold: Description
        """

        self.manual_selection = selected_features or [
            'C1', 'C2', 'C5', 'C8', 'C9', 'C12', # Choosing Cs with >0.30 correlation with Fraud
            'D1', 'D3', 'D4', 'D5', 'D8', 'D10', 'D11', 'D13', 'D14', 'D15', # Choosing non-collinear D variables
            'M1', 'M4', 'M5', 'M6', 'M7',
            

        ]