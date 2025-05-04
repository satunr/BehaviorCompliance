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

# k: Number of nodes that are allowed to be informed
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

# k: # of seeds nodes that can be chosen to inform
def greedy_for_lt(g, seed_candidates, k, threshold):
    selected_seeds = set()
    current_spread = 0

    for _ in range(k):
        best_node = None
        best_spread = -1

        # Evaluate each candidate node
        for node in set(seed_candidates) - selected_seeds:
            # Temporarily add node to selected seeds
            temp_seeds = selected_seeds | {node}
            # Compute influence spread with temporary seed set
            spread = len(LT(g, threshold, temp_seeds))
            
            # Update best node if spread is larger
            if spread > best_spread:
                best_spread = spread
                best_node = node

        # If a best node is found, add it to selected seeds
        if best_node is not None:
            selected_seeds.add(best_node)
            current_spread = best_spread
        else:
            break

    return selected_seeds, current_spread

def LT(g, threshold, initial_active: set = None):
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
                total_influence = len(
                    [pred for pred in g.predecessors(node) if pred in influence_result]
                )
                # Check if threshold is exceeded
                if total_influence >= threshold:
                    influence_result.add(node)
                    new_active = True

    return influence_result
