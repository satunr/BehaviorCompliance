import networkx as nx
import SIR
import parse
import correlated_graphs
import matplotlib.pyplot as plt
from copy import deepcopy
import pickle

n = 100
T = 60
Repeat = 1

beta = 0.20  #infection rate
gamma = 0.03  # recovery rate
mu = 0.10   # immunity loss
init = 0.1

# Parameters for misinformation
misinformation = 0.2

# Specify the filename
filename = 'contact_network_text.txt'
# Create the graph from the file
contact_graph = parse.parse(filename)

# Relabel nodes in parsed graph to avoid off by 1 errors in SIR.py
# Create a mapping from old node to new node: i -> i - 1
mapping = {node: node - 1 for node in contact_graph.nodes()}

# Relabel the nodes
contact_network = nx.relabel_nodes(contact_graph, mapping)
social_network = correlated_graphs.create_w_k_hop_correlation(contact_network, k=2)[0]  # We just want the graph part of this output

# Figure 1
# Function to compare SIR runs with and without informed individuals. Quarantines are permanent
def informed_vs_noninformed():
    #----------
    #
    #  Run SIR with informed individuals
    #
    #----------

    data1 = SIR.Simulate_SIR(contact_network=deepcopy(contact_network),social_network=deepcopy(social_network),T=T,Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init,average_data=False,q=True,allow_restoration=False)[2]

    # Extract x and y data from arrays
    x1, y1 = data1[0], data1[1]

    #----------
    #
    #  Run SIR without informed individuals
    #
    #----------

    data2 = SIR.Simulate_SIR(contact_network=deepcopy(contact_network),social_network=deepcopy(social_network),T=T,Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init,average_data=False,q=False,allow_restoration=False)[2]

    # Extract x and y data from arrays
    x2, y2 = data2[0], data2[1]

    # Create the plot
    plt.plot(x1, y1, label='With Informed', color='blue', marker='o')
    plt.plot(x2, y2, label='Without Informed', color='red', marker='o')

    # Customize the plot
    plt.xlabel('Time')
    plt.ylabel('# of Infected')
    plt.title('Informed vs Non-Informed (Permanent Quarantine)')
    plt.legend()  # Add legend to distinguish the lines
    plt.grid(True)

    # Show the plot
    plt.show()

    return data1, data2

# Figure 2
# Same as above, but with temporary quarantines
def const_quarantines():
    #----------
    #
    #  Run SIR with a constant quarantine for informed individuals
    #
    #----------

    quarantine_constant = 40

    data1 = SIR.Simulate_SIR(contact_network=deepcopy(contact_network),social_network=deepcopy(social_network),T=T,Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init,average_data=False,q=quarantine_constant,allow_restoration=True)[2]

    # Extract x and y data from arrays
    x1, y1 = data1[0], data1[1]

    #----------
    #
    #  Run SIR without quarantines
    #
    #----------

    data2 = SIR.Simulate_SIR(contact_network=deepcopy(contact_network),social_network=deepcopy(social_network),T=T,Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init,average_data=False,q=False,allow_restoration=False)[2]

    # Extract x and y data from arrays
    x2, y2 = data2[0], data2[1]

    # Create the plot
    plt.plot(x1, y1, label=f'With quarantine (constant value of {quarantine_constant})', color='blue', marker='o')
    plt.plot(x2, y2, label='Without quarantine', color='red', marker='o')

    # Customize the plot
    plt.xlabel('Time')
    plt.ylabel('# of Infected')
    plt.title('Informed vs Non-Informed (Temporary Quarantine)')
    plt.legend()  # Add legend to distinguish the lines
    plt.grid(True)

    # Show the plot
    plt.show()

    return data1, data2

# Figure 3
def normal_dist_quarantines():
    #----------
    #
    #  Run SIR with a normal distribution for quarantine times
    #
    #----------

    data1 = SIR.Simulate_SIR(contact_network=deepcopy(contact_network),social_network=deepcopy(social_network),T=T,Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init,average_data=False,q=True,allow_restoration=True)[2]

    # Extract x and y data from arrays
    x1, y1 = data1[0], data1[1]

    #----------
    #
    #  Run SIR without quarantines
    #
    #----------

    data2 = SIR.Simulate_SIR(contact_network=deepcopy(contact_network),social_network=deepcopy(social_network),T=T,Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init,average_data=False,q=False,allow_restoration=False)[2]

    # Extract x and y data from arrays
    x2, y2 = data2[0], data2[1]

    # Create the plot
    plt.plot(x1, y1, label='With quarantine', color='blue', marker='o')
    plt.plot(x2, y2, label='Without quarantine', color='red', marker='o')

    # Customize the plot
    plt.xlabel('Time')
    plt.ylabel('# of Infected')
    plt.title('Informed vs Non-Informed (Normal Dist. Quarantine)')
    plt.legend()  # Add legend to distinguish the lines
    plt.grid(True)

    # Show the plot
    plt.show()

    return data1, data2

# Function to plot Jaccard similarity between contact and social networks (2-hop creation) as X, and existence of edge (0 or 1) as Y
def plot_jaccard_similarity():
    new_social = correlated_graphs.Jaccard_similarity_plot(contact_network)
    return new_social

# Pickle results from the functions
def SIR_pickle_dump(filename='pickles.pkl'):
    # We will pickle these parameters along with the results for later reference
    presets = {'T': T, 'Repeat': Repeat, 'beta': beta, 'gamma': gamma, 'mu': mu, 'init': init}

    # data1, data2 = informed_vs_noninformed()
    data3, data4 = const_quarantines()
    # data5, data6 = normal_dist_quarantines()
    # data7 = plot_jaccard_similarity()

    with open(filename, 'wb') as f:
        # Clear the file before writing
        f.truncate(0)

        pickle.dump({'presets': presets}, f)
        # pickle.dump({'data1': data1, 'data2': data2}, f)
        pickle.dump({'data3': data3, 'data4': data4}, f)
        # pickle.dump({'data5': data5, 'data6': data6}, f)
        # pickle.dump({'data7': data7}, f)
    print("Data has been pickled successfully.")

SIR_pickle_dump()

def pickle_load(filename='pickles.pkl'):
    # Open the file in binary read mode
    with open(filename, 'rb') as file:
        data = pickle.load(file)

    # Now `data` holds the deserialized object
    print(data)

# def misinformation_comp():
#     #----------
#     #
#     #  Run SIR with misinformation
#     #
#     #----------

#     social_network = correlated_graphs.create_w_k_hop_correlation(contact_network,k=2)[0]   # We just want the graph part of this output

#     # NOTE: This data is derived from 2 separate runs of Simulate_SIR, and is therefore only an approx. comparison
#     data1 = SIR.Simulate_SIR(contact_network=deepcopy(contact_network),social_network=deepcopy(social_network),T=T,Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init,
#                              average_data=False,q=True,allow_restoration=True, misinformation_prob=None)[2]
#     data2 = SIR.Simulate_SIR(contact_network=deepcopy(contact_network),social_network=deepcopy(social_network),T=T,Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init,
#                              average_data=False,q=True,allow_restoration=False, misinformation_prob=misinformation)[2]

#     # Extract x and y data from arrays
#     x1, y1 = data1[0], data1[1]
#     x2, y2 = data2[0], data2[1]

#     # Create the plot
#     plt.plot(x1, y1, label='Without Misinformation', color='blue', marker='o')
#     plt.plot(x2, y2, label='With Misinformation', color='red', marker='o')

#     # Customize the plot
#     plt.xlabel('Time')
#     plt.ylabel('# of Infected')
#     plt.title('Misinformation Comparison')
#     plt.legend()  # Add legend to distinguish the lines
#     plt.grid(True)

#     # Show the plot
#     plt.show()

# misinformation_comp()

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
    nE = 550

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

