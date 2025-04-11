import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def create_correlated_digraph(base_graph, correlation_factor, base_probability=0.01, num_nodes=100):
    """
    Creates a directed graph A where edge existence correlates with 
    an undirected base graph B, with better sparsity control
    correlation_factor: float between 0 and 1, controls correlation strength
    base_probability: float between 0 and 1, controls overall density
    """
    # Create empty directed graph A
    A = nx.DiGraph()
    A.add_nodes_from(range(num_nodes))
    
    # Get edges from base graph (undirected)
    b_edges = set(base_graph.edges())
    
    # Consider all possible directed edges
    possible_edges = [(i, j) for i in range(num_nodes) 
                     for j in range(num_nodes) if i != j]
    
    # For each possible directed edge in A
    for edge in possible_edges:
        edge_undirected = tuple(sorted(edge))
        
        # Base probability modified by correlation
        if edge_undirected in b_edges:
            # Edges present in B get a boost
            prob = base_probability * (1 + correlation_factor)
        else:
            # Edges not in B get reduced probability
            prob = base_probability * (1 - correlation_factor)
            
        # Cap probability at 1
        prob = min(prob, 1.0)
        
        # Add directed edge probabilistically
        if np.random.random() < prob:
            A.add_edge(*edge)
    
    return A