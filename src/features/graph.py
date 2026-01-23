import pandas as pd
import networkx as nx
import logging

class GraphFeatureEngineer:
    """
    Tier 5: Graph-based features

    Business justification:
    - Each transaction can be mapped into interrelated nodes. If we have a credit card which is
      associated with John Das and is accessed from John Das' phone and John Das' computer, then 
      we know that the nodes are consistent with one another. But if John Das' card is accessed
      from the Netherlands or from an unknown laptop, and if that node also accessed 50 other 
      credit cards belonging to different people, then we can say with a high confidence that
      we have an instance of fraud.

    - By mapping each TransactionAmt to TrasactionID, we can create a graph where we attempt
      to detect fraud by creating nodes and detecting group activity from suspicious accounts or
      locations.
    """

    def __init__(self, cols_to_link = ['card1', 'P_emaildomain', 'DeviceInfo']):
        self.cols_to_link = cols_to_link

    def fit_transform(self, df):
        # Right now we are using networkx - in a production setting, a graph database like Neo4j 
        # would be necessary.

        logging.info('Building graph...')
        df = df.copy()

        G = nx.Graph()

        # Converting columns to strings to avoid mismatches
        for col in self.cols_to_link:
            if col in df.columns:
                # Creating edges
                df[col] = df[col].fillna('Unknown')
                edges = list(zip(df['TransactionID'], df[col].apply(lambda x: f'{col}_{x}')))
                G.add_edges_from(edges)

        logging.info(f'Graph has been built: {G.number_of_nodes} nodes, {G.number_of_edges} edges.')

        # Finding size of each component (i.e. fraud ring)
        logging.info('Calculating components...')

        components = list(nx.connected_components(G))
        node_component_size = {}
        for comp in components:
            size = len(comp)
            for node in comp:
                node_component_size[node] = size

        # Finding popularity of each node of the TransactionID
        # Can explain how frequently this source is used for transactions
        degree_dict = dict(G.degree(df['TransactionID']))
        logging.info('Mapping graph features...')

        df['uid_graph_degree'] = df['TransactionID'].map(degree_dict).fillna(0)
        df['uid_graph_component_size'] = df['TransactionID'].map(node_component_size).fillna(1)

        return df
    
if __name__ == '__main__':
    # Testing on dummy data. Transaction 5 is the odd one out.
    dummy_data = {
        'TransactionID': [1, 2, 3, 4, 5],
        'card1': [100, 100, 200, 200, 300],
        'P_emaildomain': ['A', 'A', 'A', 'B', 'C'],
        'DeviceInfo': ['D1', 'D2', 'D1', 'D2', 'D3']
    }

    df_test = pd.DataFrame(dummy_data)
    graph_engineer = GraphFeatureEngineer(cols_to_link = ['card1', 'P_emaildomain', 'DeviceInfo'])
    df_out = graph_engineer.fit_transform(df_test)

    print(df_out[['TransactionID', 'uid_graph_component_size']])