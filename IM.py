import numpy as np
import random
import networkx as nx

# Returns probability matrix of node i being informed
# S: Nodes that are already informed
# p: Probability of activation
# mc: Number of Monte Carlo simulations
# quarantining: List of nodes currently quarantining
def IC_prob_matrix(g, S, p, mc=5000, quarantining=None):
    if S == []: raise ValueError("S cannot be empty")

    quarantine_list = []
    for _ in range(mc):
        A = S[:]
        new_ones = []
        for node in S:
            out_neighbors = list(g.successors(node))
            success = [u for u in out_neighbors if random.uniform(0, 1) < p]
            new_ones += success
        new_active = list(set(new_ones) - set(A))
        A += new_active
        A = list(set(A))

        # Create informed matrix of size len(g.nodes()) x 1
        one_run = np.zeros((1, len(g.nodes())))
        for node in A:
            # if node in quarantining:
            one_run[0][node] = 1
        quarantine_list.append(one_run)

    quarantine_matrix = np.array(quarantine_list)
    quarantine_matrix = np.mean(quarantine_matrix, axis=0)  # Average over all Monte Carlo simulations

    return quarantine_matrix, A  # Return the average probability matrix and the final set of informed nodes

def IC(g, S, p, mc=10):
    spread = []
    for _ in range(mc):
        A = S[:]
        while new_ones != []:  # While there are new nodes to be activated
            new_ones = []
            for node in S:
                out_neighbors = list(g.successors(node))  # Get successors of current node
                success = [u for u in out_neighbors if random.uniform(0, 1) < p]  # Check if activation succeeds
                new_ones += success  # Add successful activations to new_ones
            new_active = list(set(new_ones) - set(A))
            A += new_active
            A = list(set(A))
            spread.append(len(A))

    return np.mean(spread), A

# k: Number of nodes that are allowed to be informed
def greedy(g,k,p=0.1,mc=10,S=None):
    if S == None: S = []
    spread = []

    for _ in range(k):  # k nodes of maximimum influence
        best_spread = 0
        for j in g.nodes():  # Look at nodes not yet in the set
            if j in S:
                continue
            s = IC(g, S + [j], p, mc)[0]
            if s > best_spread:
                best_spread, node = s, j
        S.append(node)
        spread.append(best_spread)

    return S, spread

def lt_prob_matrix(g, S, threshold=1, quarantining=None):
    if S == []: raise ValueError("S cannot be empty")

    A = S[:]
    quarantine_matrix = np.zeros((1, len(g.nodes())))

    for node in g.nodes():
        if node not in A:  # Not yet informed
            total_influence = len(
                [pred for pred in g.predecessors(node) if pred in A]
            )
            if total_influence >= threshold:
                A.append(node)

    for node in A:
        # if node in quarantining:
        quarantine_matrix[0][node] = 1

    return quarantine_matrix, A  # Return the average probability matrix and the final set of informed nodes

# k: # of seeds nodes that can be chosen to inform
def greedy_for_lt(g, seed_candidates, k=3, threshold=1):
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
    new_ones = True  # Flag to track new activations

    while new_ones == True:
        new_ones = False
        for node in g.nodes():
            if node not in influence_result:  # Not yet influenced
                # Sum weights from active predecessors
                total_influence = len(
                    [pred for pred in g.predecessors(node) if pred in influence_result]
                )
                # Check if threshold is exceeded
                if total_influence >= threshold:
                    influence_result.add(node)
                    new_ones = True

    return list(influence_result)
