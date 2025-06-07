import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from copy import deepcopy

k_hop_simulation = False

def create_correlated_digraph(base_graph, correlation_factor, base_probability=0.01):
    """
    Creates a directed graph A where edge existence correlates with 
    an undirected base graph B, with better sparsity control
    correlation_factor: float between 0 and 1, controls correlation strength
    base_probability: float between 0 and 1, controls overall density
    """
    num_nodes = len(base_graph.nodes()) 

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

def create_w_k_hop_correlation(base_graph, k):
    x_vals, y_vals = [], []

    # Initialize an empty unweighted graph
    correlated_graph = nx.DiGraph()
    
    # Add all nodes from the base graph to the correlated graph
    correlated_graph.add_nodes_from(base_graph.nodes())
    nodes = list(base_graph.nodes()) # Node list we will be working with
    
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
            
            # Add an edge with probability equal to the Jaccard similarity
            if random.random() < similarity:
                correlated_graph.add_edge(i, j)  # Unweighted edge

            x_vals.append(similarity)
            y_vals.append(1 if correlated_graph.has_edge(i, j) else 0)

    return correlated_graph, (x_vals, y_vals)  # Return the graph, similarity matrix, and x,y values for plotting

def jaccard_similarity(set1, set2):
    set1 = set(set1)
    set2 = set(set2)

    # intersection of two sets
    intersection = len(set1.intersection(set2))
    # Unions of two sets
    union = len(set1.union(set2))

    return intersection / union

# H: Base graph, undirected
def Jaccard_similarity_plot(H):

    # Ego network of K-hops around each node in H and Jaccard similarity
    K = 2
    N = list(sorted(H.nodes()))

    # Create ego network for each node
    Ego = {u: list(nx.ego_graph(H, u, radius=K).nodes()) for u in H.nodes()}

    # Compute Jaccard similarity for each pair of nodes in Ego
    sim = {(N[i], N[j]): jaccard_similarity(Ego[N[i]], Ego[N[j]])
        for i in range(len(N) - 1) for j in range(i + 1, len(N))}

    # Create social graph G with density q
    G = nx.DiGraph()
    nE = 450

    # Sample 'nE' edge pairs with high Jaccard similarity
    A = [(N[i], N[j]) for i in range(len(N) - 1) for j in range(i + 1, len(N))]
    prob = [np.exp(sim[(N[i], N[j])]) for i in range(len(N) - 1) for j in range(i + 1, len(N))]
    prob = [val / sum(prob) for val in prob]

    E = np.random.choice(a=[i for i in range(len(A))], p=prob, size=nE, replace=False).tolist()
    E = [A[index] for index in E]

    # For each sampled edge (u, v),
    # randomly choose whether to add edge from (u, v) or (v, u) in G
    for (u, v) in E:
        if random.choice([0, 1]) == 0:
            G.add_edge(u, v)
        else:
            G.add_edge(v, u)

    # Plot correlation between similarity and edge existence in undirected version of G
    I = G.to_undirected()
    plt.scatter([sim[(N[i], N[j])] for i in range(len(N) - 1) for j in range(i + 1, len(N))],
        [int(I.has_edge(N[i], N[j])) for i in range(len(N) - 1) for j in range(i + 1, len(N))],
                s=10, alpha=0.1)
    plt.show()
    
    return G  # Return the created graph for further use if needed

