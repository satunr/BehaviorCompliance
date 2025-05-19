import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from copy import deepcopy

k_hop_simulation = False

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

# Modified function to accept a k parameter for hop distance
def create_w_k_hop_correlation(base_graph, k):
    # Initialize an empty unweighted graph
    correlated_graph = nx.DiGraph()
    
    # Add all nodes from the base graph to the correlated graph
    correlated_graph.add_nodes_from(base_graph.nodes())
    
    # Initialize the similarity matrix (dictionary)
    similarity_matrix = {}
    
    # Get nodes from the base graph
    nodes = list(base_graph.nodes())
    
    # Compute Jaccard similarity for each pair of nodes based on k-hop neighborhoods
    for idx_i, i in enumerate(nodes):
        # Get the k-hop neighborhood of node i (ego graph with radius k)
        i_neighbors = set(nx.ego_graph(base_graph, i, radius=k).nodes())
        for j in nodes[idx_i + 1:]:  # Avoid duplicate pairs (i,j) and (j,i)
            # Get the k-hop neighborhood of node j
            j_neighbors = set(nx.ego_graph(base_graph, j, radius=k).nodes())
            
            # Compute intersection and union
            intersection = len(i_neighbors & j_neighbors)
            union = len(i_neighbors | j_neighbors)
            
            # Handle division by zero
            if union == 0:
                similarity = 0.0
            else:
                similarity = intersection / union
            
            # Store the similarity in the matrix
            similarity_matrix[(i, j)] = similarity
            
            # Add an edge with probability equal to the Jaccard similarity
            if random.random() < similarity:
                correlated_graph.add_edge(i, j)  # Unweighted edge
    
    return correlated_graph, similarity_matrix

# Create a larger base graph with ~100 nodes
def create_sample_graph():
    # Create an Erdős-Rényi random graph with 100 nodes and edge probability 0.05
    num_nodes = 100
    edge_prob = 0.05  # Adjust this probability to control density
    G = nx.erdos_renyi_graph(num_nodes, edge_prob, seed=42)  # Seed for reproducibility
    return G

if k_hop_simulation == True:
    # Driver code to create, compute, and visualize the graphs
    # Set a random seed for reproducibility of the correlated graphs

    random.seed(42)

    # Create the base graph
    base_graph = create_sample_graph()

    # Print basic info about the base graph
    print("Base Graph Nodes:", len(base_graph.nodes()))
    print("Base Graph Edges:", len(base_graph.edges()))

    # Compute the correlated graph with k=1 (1-hop)
    correlated_graph_1hop, similarity_matrix_1hop = create_w_k_hop_correlation(base_graph, k=1)

    # Compute the correlated graph with k=2 (2-hop)
    correlated_graph_2hop, similarity_matrix_2hop = create_w_k_hop_correlation(base_graph, k=2)

    # Print basic info about the correlated graphs
    print("\nCorrelated Graph (1-hop) Nodes:", len(correlated_graph_1hop.nodes()))
    print("Correlated Graph (1-hop) Edges:", len(correlated_graph_1hop.edges()))
    print("Correlated Graph (2-hop) Nodes:", len(correlated_graph_2hop.nodes()))
    print("Correlated Graph (2-hop) Edges:", len(correlated_graph_2hop.edges()))

    # Print a sample of the similarity matrices (to avoid flooding the output)
    print("\nSimilarity Matrix (1-hop, Sample):")
    sample_pairs_1hop = list(similarity_matrix_1hop.items())[:5]  # Show first 5 pairs
    for (i, j), similarity in sample_pairs_1hop:
        print(f"Nodes ({i}, {j}): Jaccard Similarity = {similarity:.3f}")

    print("\nSimilarity Matrix (2-hop, Sample):")
    sample_pairs_2hop = list(similarity_matrix_2hop.items())[:5]  # Show first 5 pairs
    for (i, j), similarity in sample_pairs_2hop:
        print(f"Nodes ({i}, {j}): Jaccard Similarity = {similarity:.3f}")

    # Compute positions for nodes using a spring layout (same for all graphs for consistency)
    pos = nx.spring_layout(base_graph, seed=42)  # Seed for reproducibility

    # First comparison: Base Graph vs. 1-hop Correlated Graph
    plt.figure(figsize=(12, 5))

    # Plot the base graph
    plt.subplot(121)
    nx.draw(base_graph, pos, with_labels=False, node_size=50, node_color='lightblue', edge_color='gray', width=1)
    plt.title("Base Graph")

    # Plot the 1-hop correlated graph
    plt.subplot(122)
    nx.draw(correlated_graph_1hop, pos, with_labels=False, node_size=50, node_color='salmon', edge_color='gray', width=1)
    plt.title("Correlated Graph (1-hop)")

    plt.tight_layout()
    plt.show()

    # Second comparison: Base Graph vs. 2-hop Correlated Graph
    plt.figure(figsize=(12, 5))

    # Plot the base graph again
    plt.subplot(121)
    nx.draw(base_graph, pos, with_labels=False, node_size=50, node_color='lightblue', edge_color='gray', width=1)
    plt.title("Base Graph")

    # Plot the 2-hop correlated graph
    plt.subplot(122)
    nx.draw(correlated_graph_2hop, pos, with_labels=False, node_size=50, node_color='lightgreen', edge_color='gray', width=1)
    plt.title("Correlated Graph (2-hop)")

    plt.tight_layout()
    plt.show()

# prob vector: 1 x n; probability of each node being informed
    # vector because it represents network as last time step
def generate_from_prob_matrix(network, prob_vector):
    """
    Generates a directed graph from a given probability matrix, and assigns edge weight attributes as probabilities.
    """
    G = deepcopy(network)

    # Assign weights to edges based on the probability matrix
    for i in range(0, len(G.nodes())):
        for j in G.successors(i):
            # G[i][j]['weight'] = abs(1 - prob_vector[i])  # 0 prob of informed -> 1 prob of having edges to neighbors
            nx.set_edge_attributes(G, {(i, j): {'weight': abs(1 - prob_vector[i])}})
    
    return G

