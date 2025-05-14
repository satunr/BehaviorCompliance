import networkx as nx
import numpy as np
import IM
import matplotlib.pyplot as plt

# Probabilistic way of finding seed nodes, given an initial set
#   Governed by: P_i(seed) = (d_i)^k / sum_j ((d_j)^k)
# n is the number of seeds we want
# exponent to control degree weighting
def find_seed_set(g, num_seeds, exponent=1):
    # Get degrees of all nodes
    degrees = dict(g.degree())  # {node: degree}

    # Calculate probabilities
    nodes = list(degrees.keys())
    epsilon = 1e-6  # tiny bump to prevent 0 probabilities
    probs = [(degrees[node] ** exponent) + epsilon for node in nodes]
    total = sum(probs)
    probs = [p / total for p in probs]

    if np.count_nonzero(probs) < num_seeds:
        raise ValueError(f"Cannot choose {num_seeds} seeds — only {np.count_nonzero(probs)} nodes have non-zero probability.")

    # Create the initial seed set
    seed_set = list(np.random.choice(nodes, size=num_seeds, replace=False, p=probs))

    for node in seed_set:
        nx.set_node_attributes(g, {node: {'Seed?': 'Seed'}})

    return seed_set

# lt == True: linear threshold model. lt == False: independent cascade model
def initialize_social_IM(social_network,k=None,p=0.3,num_seeds=1,lt_threshold = None):
    seeds = find_seed_set(social_network, num_seeds, exponent=1)
    if lt_threshold == None:
        result = IM.greedy(social_network, k, seeds, p)
    else:
        result = IM.greedy_for_lt(social_network, seeds, k, lt_threshold)
    max_influence = result[0]
    spread = result[1]

    return max_influence

# Create a sample social network (using a Barabasi-Albert graph as an example)
# social_network = nx.DiGraph()
# social_network.add_nodes_from([1,2,3,4,5])
# social_network.add_edges_from([(1,2), (3,4), (5,1)])

# # Parameters for the Independent Cascade Model
# k = 3       # Number of attempts to activate a node
# p = 0.1      # Probability of activation

# # Number of seed nodes to select
# num_seeds = 5

# # Run the social network influence maximization
# max_influence = initialize_social_IM(social_network, k, p, num_seeds)

# print(f"Selected {num_seeds} seed nodes")
# print(f"Maximum influence spread: {max_influence}")

# # Visualize the network with seed nodes highlighted
# pos = nx.spring_layout(social_network)
# node_colors = ['red' if social_network.nodes[node].get('Seed?') == 'Seed' else 'skyblue' 
#             for node in social_network.nodes()]

# plt.figure(figsize=(10, 8))
# nx.draw(social_network, pos, node_color=node_colors, with_labels=True, node_size=300)
# plt.title("Social Network with Seed Nodes Highlighted")
# plt.show()
