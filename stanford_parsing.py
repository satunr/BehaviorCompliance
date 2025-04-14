import pandas as pd
import networkx as nx
import os
from pathlib import Path
import matplotlib.pyplot as plt

extract_file = True

def load_network_to_dict(network_file):
    """
    Parse a network[ID].csv file and return a dictionary {time: DiGraph}
    containing only human-to-human interactions (ignoring laptop).
    
    Args:
        network_file (str): Path to the network[ID].csv file.
    
    Returns:
        dict: Dictionary mapping timestamp (float) to NetworkX DiGraph.
    """
    # Read the CSV file
    df = pd.read_csv(network_file)
    
    # Get the number of participants by checking columns like P1_TO_P*
    # Columns are TIME, P1_TO_LAPTOP, P1_TO_P1, ..., P1_TO_Pn, P2_TO_LAPTOP, ...
    # Find unique participant IDs by looking at columns
    all_columns = df.columns.tolist()
    participants = set()
    for col in all_columns:
        if col.startswith("P") and "_TO_P" in col:
            # Extract participant number from Px_TO_Py
            parts = col.split("_TO_P")
            participants.add(parts[0])  # e.g., P1
            participants.add("P" + parts[1])  # e.g., P2
    
    participants = sorted(list(participants))  # e.g., ['P1', 'P2', ..., 'Pn']
    
    # Filter out laptop columns (e.g., P1_TO_LAPTOP)
    human_columns = [col for col in all_columns if "_TO_LAPTOP" not in col]
    df_human = df[human_columns]
    
    # Initialize the dictionary
    time_to_graph = {}
    
    # Process each row (timestamp)
    for _, row in df_human.iterrows():
        time = row["TIME"]
        G = nx.DiGraph()
        
        # Add all participants as nodes
        for p in participants:
            G.add_node(p)
        
        # Add directed edges based on binary values
        for col in human_columns[1:]:  # Skip TIME column
            if row[col] == 1:
                # Column name is like Px_TO_Py
                source, target = col.split("_TO_")
                G.add_edge(source, target)
        
        time_to_graph[time] = G
    
    return time_to_graph

if extract_file == True:
    time_to_graph = load_network_to_dict("network0.csv")

    G = time_to_graph[10.0]
    nx.draw(G, with_labels=True)
    plt.show()

    