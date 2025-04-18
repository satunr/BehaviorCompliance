import numpy as np
import random
import networkx as nx


def IC(g, S, p, mc=1000):
    spread = []
    for _ in range(mc):
        new_active, A = S[:], S[:]
        while new_active:
            new_ones = []
            for node in new_active:
                out_neighbors = list(g.successors(node))
                success = [u for u in out_neighbors if random.uniform(0, 1) < p]
                new_ones += success
            new_active = list(set(new_ones) - set(A))
            A += new_active
        A = list(set(A))
        spread.append(len(A))
    return np.mean(spread)

def greedy(g, k, seeds, p=0.1, mc=1000):
    S = []
    spread = []
    for _ in range(k):
        best_spread = 0
        for j in seeds:  # Only consider nodes in seeds
            if j in S:
                continue
            s = IC(g, S + [j], p, mc)
            if s > best_spread:
                best_spread, node = s, j
        S.append(node)
        spread.append(best_spread)

    return S, spread

def LT(g, initial_active: set = None):
    # Initialize thresholds for each node
    thresholds = np.random.uniform(0, 1, len(g))

    # Assign fixed influence weights to edges
    edge_weights = {}
    for node in g.nodes():
        predecessors = list(g.predecessors(node))
        if predecessors:
            # Generate random weights and normalize to sum <= 1
            weights = np.random.uniform(0, 1, len(predecessors))
            total = sum(weights)
            if total > 0:  # Avoid division by zero
                weights = weights * (np.random.uniform(0, 1) / total)
            else:
                weights = np.zeros(len(predecessors))
            # Assign weights to edges
            for pred, weight in zip(predecessors, weights):
                edge_weights[(pred, node)] = weight

    # Initialize active nodes
    active = set(initial_active) if initial_active else set()
    influence_result = set(active)  # Track all influenced nodes
    new_active = True

    # Iterative activation process
    while new_active:
        new_active = False
        for node in g.nodes():
            if node not in influence_result:  # Not yet influenced
                # Sum weights from active predecessors
                total_influence = sum(
                    edge_weights.get((pred, node), 0)
                    for pred in g.predecessors(node)
                    if pred in influence_result
                )
                # Check if threshold is exceeded
                if total_influence >= thresholds[node]:
                    influence_result.add(node)
                    new_active = True

    return influence_result
        
