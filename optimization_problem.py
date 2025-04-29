import networkx as nx
import matplotlib.pyplot as plt
import IM
import parse
import find_seeds
import correlated_graphs

run_maximization = True

def find_best_integer(g,seeds:set,A,X_min,X_max):
    """
    Find integer X >= 2 such that f(X) ≈ A, where f: Z -> Z is monotone decreasing.
    
    Args:
        f: Function mapping integers to integers, monotone decreasing.
        A: Target real number (float).
        X_min: Minimum integer X.
        X_max: Maximum integer X to search.
    
    Returns:
        Integer X that minimizes |f(X) - A|.
    """

    # Initialize bounds
    low = X_min
    high = X_max
    
    # If A is outside the range of f, handle edge cases
    if len(IM.LT(g,X_min,seeds)) < A:
        return X_min  # Smallest X gives largest f(X), still too small
    if len(IM.LT(g,X_max,seeds)) > A:
        return X_max  # Largest X gives smallest f(X), still too large
    
    # Binary search
    best_X = None
    min_diff = float('inf')
    
    while low <= high:
        mid = (low + high) // 2
        f_mid = len(IM.LT(mid))
        
        # Evaluate difference
        diff = abs(f_mid - A)
        if diff < min_diff:
            min_diff = diff
            best_X = mid
        
        # Adjust search based on monotonicity
        if f_mid > A:
            low = mid + 1  # Need smaller f(X), so increase X
        elif f_mid < A:
            high = mid - 1  # Need larger f(X), so decrease X
        else:
            return mid  # Exact match (unlikely since A is float)
    
    # Check consecutive integers around the best X
    candidates = [best_X]
    if best_X > X_min:
        candidates.append(best_X - 1)
    if best_X < X_max:
        candidates.append(best_X + 1)
    
    # Return X with smallest |f(X) - A|
    return min(candidates, key=lambda x: abs(len(IM.LT(g,x,seeds)) - A))

# We will use this to make graph representation of the optimization problem cleaner
def truncate_at_value(lst, val, include_val=True):
    try:
        idx = lst.index(val)
        return lst[:idx + 1] if include_val else lst[:idx]
    except ValueError:
        return lst[:]  # Return copy of original list if val not found

if run_maximization == True:
    # Specify the filename
    filename = 'contact_network_text.txt'
    # Create the graph from the file
    contact_graph = parse.parse(filename)  # Requires a lot of computational power
    seeds = find_seeds.find_seed_set(contact_graph, 20)

    social_graph = correlated_graphs.create_w_k_hop_correlation(contact_graph, 2)[0]  # Just want the graph here

    maximization_result = IM.greedy(social_graph, 11, seeds)
    max_influence = len(maximization_result[0])

    thresholds = [i for i in range(2,6)]
    lt_result = []
    for threshold in thresholds:
        lt_result.append(len(IM.LT(social_graph, threshold, seeds)))

    # Create bar graph
    # plt.figure(figsize=(8, 6))
    # plt.bar(thresholds, lt_result, color='skyblue', edgecolor='black')
    # plt.axhline(y=max_influence, color='red', linestyle=':', linewidth=2, label='True Influence')
    # plt.xlabel('Threshold')
    # plt.ylabel('Number of Influenced Nodes')
    # plt.title('Influence Spread vs. Threshold')
    # plt.xticks(thresholds)
    # plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # # Save the plot
    # # plt.savefig('lt_influence_bar_graph.png')
    # plt.show()

    
    #-----------
    #
    #  Fitting problem (using binary search)
    #
    #-----------

    X_fitted = find_best_integer(social_graph,seeds,maximization_result,2,10)
    thresholds = truncate_at_value(thresholds,X_fitted)

    # Create bar graph
    plt.figure(figsize=(8, 6))
    plt.bar(thresholds, lt_result, color='skyblue', edgecolor='black')
    plt.axhline(y=max_influence, color='red', linestyle=':', linewidth=2, label='True Influence')
    plt.xlabel('Threshold')
    plt.ylabel('Number of Influenced Nodes')
    plt.title('Influence Spread vs. Threshold')
    plt.xticks(thresholds)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Save the plot
    # plt.savefig('lt_influence_bar_graph.png')
    plt.show()