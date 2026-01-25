import pandas as pd
import numpy as np
import logging
import os
# Adjust this import based on where you put the class in Sprint 3
from src.features.graph import GraphFeatureEngineer 

logging.basicConfig(level=logging.INFO, format='%(message)s')

def debug_graph_features():
    logging.info("Loading 50k rows for graph debugging...")
    
    # Load a subset (enough to find connections)
    df = pd.read_csv('data/raw/train_transaction.csv', nrows=50000)
    
    # Run the Graph Engineer
    logging.info("Building Graph...")
    graph_engine = GraphFeatureEngineer()
    df_graph = graph_engine.fit_transform(df)
    
    # 1. CHECK VARIANCE: Are all values the same?
    cols = [c for c in df_graph.columns if 'graph' in c]
    
    for col in cols:
        logging.info(f"\n--- Analysis of {col} ---")
        logging.info(f"Unique Values: {df_graph[col].nunique()}")
        logging.info("Top 10 Value Counts:")
        logging.info(df_graph[col].value_counts().head(10))
        
        if df_graph[col].nunique() <= 1:
            logging.warning(f"CRITICAL: {col} has Zero Variance! It is effectively a constant.")
            
    # 2. CHECK CORRELATION: Is it just a copy of card1_count?
    # We need to quickly calculate card1 count to compare
    df_graph['simple_count'] = df_graph.groupby('card1')['TransactionID'].transform('count')
    
    if 'uid_graph_component_size' in df_graph.columns:
        corr = df_graph['uid_graph_component_size'].corr(df_graph['simple_count'])
        logging.info(f"\nCorrelation between Graph Size and Card1 Count: {corr:.4f}")
        
        if corr > 0.95:
            logging.warning("INSIGHT: Graph feature is >95% correlated with simple count. The model prefers the simple one.")

if __name__ == "__main__":
    debug_graph_features()