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