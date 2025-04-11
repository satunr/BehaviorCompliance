import networkx as nx
import numpy as np
import IM
import correlated_graphs

# Probabilistic way of finding seed nodes, given an initial set
#   Governed by: P_i(seed) = (d_i)^k / sum_j ((d_j)^k)
# n is the number of seeds we want
# exponent to control degree weighting
def find_seed_set(g, num_seeds, exponent=1):
    # Get degrees of all nodes
    degrees = dict(g.degree())  # {node: degree}

    # Calculate probabilities
    nodes = list(degrees.keys())
    probs = [(degrees[node] ** exponent) for node in nodes]
    total = sum(probs)
    probs = [p / total for p in probs]

    # Create the initial seed set
    seed_set = list(np.random.choice(nodes, size=num_seeds, replace=False, p=probs))

    for node in seed_set:
        nx.set_node_attributes(g, {node: {'Seed?': 'Seed'}})

    return seed_set

# Creates a social network, creates initial seeds, finds final seeds
# G: base graph
def initialize_social_IM(G, k, p, num_seeds):
    # NOTE: Creates a graph for now. May want to pass in a graph instead
    # S = nx.erdos_renyi_graph(network_size, p, directed=True)
    S = correlated_graphs.create_correlated_digraph(G, 0.3, 0.02, len(G.nodes()))
    seeds = find_seed_set(S, num_seeds, exponent=1)
    result = IM.greedy(S, k, seeds, p)
    max_influence = result[0]
    # spread = result[1]

    return max_influence
