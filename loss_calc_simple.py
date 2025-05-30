import networkx as nx
import matplotlib.pyplot as plt
import parse
import correlated_graphs
import IM
import find_seeds
import numpy as np
from copy import deepcopy
from sklearn.metrics import mean_squared_error
import random
import py4cytoscape as p4c

ping_cytoscape = False

# Set NumPy to print full arrays without truncation
# np.set_printoptions(threshold=np.inf)

def random_walk_subgraph(G, si):
    H = G.to_undirected()
    dsum = float(sum(H.degree(u) for u in H.nodes()))
    g = nx.DiGraph()

    # Choose a random node based on degree distribution
    r = np.random.choice(G.nodes(), 1, p=[H.degree(u)/dsum for u in G.nodes()])
    g.add_node(int(r[0]))

    # We want a subgraph of fixed size si
    while len(g) < si:
        nset = [v for u in g.nodes() for v in H.neighbors(u) if v not in g.nodes()]
        if not nset:
            return g
        r = random.choice(nset)
        new_edges = [(u, r) for u in g.nodes() if G.has_edge(u, r)]
        new_edges += [(r, u) for u in g.nodes() if G.has_edge(r, u)]
        g.add_edges_from(new_edges)
    return g

def relabel_nodes_sequential(graph):
    # Get list of nodes
    nodes = list(graph.nodes())
    
    # Create mapping from old node IDs to new sequential IDs (0, 1, 2, ...)
    node_mapping = {old_node: idx for idx, old_node in enumerate(nodes)}
    
    # Create a new graph with relabeled nodes
    relabeled_graph = nx.relabel_nodes(graph, node_mapping, copy=True)
    
    return relabeled_graph, node_mapping

def remove_random_nodes(graph, num_nodes):
    if num_nodes < 0 or num_nodes > len(graph.nodes()):
        raise ValueError("Number of nodes to remove must be between 0 and the total number of nodes in the graph")
    
    # Create a copy of the graph to avoid modifying the original
    new_graph = graph.copy()
    
    # Randomly select nodes to remove
    nodes_to_remove = random.sample(list(new_graph.nodes()), num_nodes)
    
    # Remove selected nodes
    new_graph.remove_nodes_from(nodes_to_remove)
    
    return new_graph

#--------
#
#  Observed data (from I.C.)
#
#--------

def calculate_loss(social_network, plot=False):
    cyto_social = None

    # Send the NetworkX graph to Cytoscape
    # p4c.create_network_from_networkx(social_network, collection="My NetworkX Graph", title="Social Subgraph")

    ic_results = None
    # Factor to not have too many seeds (experimentally determined)
    num_seeds = round(0.15 * len(social_network.nodes()))

    T = 50
    informed = find_seeds.find_seed_set(social_network, num_seeds=num_seeds, exponent=0.5)
    print("# of initial informed nodes: ", len(informed))

    ic_matrix = np.zeros((T, len(social_network.nodes())))

    # Probabilities are weighted to be small, as the networks are highyly connected
    activation_probabilities = np.random.normal(loc=0.03, scale=0.02, size=(1, len(social_network.nodes())))
    # activation_probabilities = 0.03

    matrices = []

    for i in range(0, T):
        print("T for I.C.: ", i)
        print("# of informed nodes: ", len(informed))

        # NOTE: Make sure mc is not too low for real testing
        cur_ic_results = IM.IC_prob_matrix(g=social_network, S=informed, p=activation_probabilities, mc=1000)  # No changes made to network; no deepcopy needed
        ic_matrix[i] = cur_ic_results[0] # Fill out the quarantine probability matrix
        informed = informed + cur_ic_results[1]  # Update the informed set
        informed = list(set(informed)) # Remove duplicates

    for node in informed:
        nx.set_node_attributes(social_network, {node: 'Informed'}, 'Informed?')

    matrices.append(ic_matrix)


    #--------
    #
    #  Inferred data using L.T.
    #
    #--------

    seeds = find_seeds.find_seed_set(social_network, num_seeds=num_seeds, exponent=1)

    losses = []
    for i in range(0,11):  # Vary threshold parameter
        print("# of initial informed nodes for L.T.: ", len(seeds))

        lt_matrix = np.zeros((T, len(social_network.nodes())))  # Only for printing purposes

        informed = seeds[:]  # Reset informed set for each threshold
        # loss_matrix = np.zeros((T, 1))

        for j in range(0, T):
            print("T for L.T.: ", i)
            print("# of informed nodes: ", len(informed))

            # Inferred data using L.T.
            cur_lt_results = IM.lt_prob_matrix(g=social_network, S=informed, threshold=i)
            informed = informed + cur_lt_results[1]  # Update the informed set
            informed = list(set(informed))  # Remove duplicates
            lt_matrix[j] = cur_lt_results[0]  # Fill out the quarantine probability matrix

        losses.append(mean_squared_error(ic_matrix[T-1], cur_lt_results[0])) # Calculate loss of final results of I.C., L.T.

        for node in informed:
            nx.set_node_attributes(social_network, {node: 'Informed'}, 'Informed?')

        # if ping_cytoscape == True:
        #     cyto_social_lt = correlated_graphs.generate_from_prob_matrix(social_network, lt_matrix[T-1]) # Adds probabilistic edge weight attributes

        #     # Verify connection to Cytoscape
        #     print(p4c.cytoscape_ping())

        #     # Export the NetworkX graph to Cytoscape
        #     network1 = p4c.create_network_from_networkx(cyto_social_lt, collection="My Network Collection", title=f"Social after L.T. (Tau = {i})")
            
        #     # Apply a layout (e.g., force-directed)
        #     p4c.layout_network("force-directed")

        #     # Apply a default visual style
        #     p4c.set_visual_style("default")
        
    # print("Losses: ", losses)

        matrices.append(lt_matrix)

    if plot == True:
        x_vals = [i for i in list(range(len(losses)))]

        # Create the bar graph
        plt.figure(figsize=(12, 6))
        plt.bar(x_vals, losses, color='skyblue', edgecolor='black')
        plt.xlabel('Threshold')
        plt.ylabel('Loss wrt # of informed')
        plt.title('Loss between average observed and inferred quarantine data')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    return losses, matrices

#--------
#
#  Loss on many networks
#
#--------

# Si: Size of subgraph to generate
# num_networks: Number of subgraphs to generate
def calculate_loss_on_many_networks(social_graph, num_networks=10, si=70):
    # Generate some random graphs (remove just 1 random node to keep them similar)
    data = [random_walk_subgraph(social_graph, si=si) for _ in range(num_networks)]
    data = [relabel_nodes_sequential(graph)[0] for graph in data]  # Relabel nodes to sequential integers
    # data = [remove_random_nodes(social_graph, num_nodes=random.randint(2,10)) for _ in range(num_networks)]  # Remove 1 random node from each graph
    # data = [relabel_nodes_sequential(graph)[0] for graph in data]  # Relabel nodes to sequential integers
    # Pack the data as a list of tuples of (losses, connectivity of the graph, probability matrices)
    results = [(loss[0], round(np.mean([graph.out_degree(node) for node in graph]), 1), loss[1]) 
               for graph, loss in [(g, calculate_loss(g)) for g in data]]
    matrices = [item[2] for item in results]  # Extract the probability matrices from the results

    # Print the results
    for item in results:
        print(f"Graph Data:\nConnectivity: {item[1]}\nLosses: {item[0]}\n")

    # Create the bar graph
    x_vals = [i for i in list(range(len(results[0][0])))] # Length of losses
    z_vals = [item[1] for item in results] # list of floats
    y_vals = [item[0] for item in results] # list of lists

    # Create figure and 3D axis
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each 2D bar graph at its z-level
    for i, (y, z) in enumerate(zip(y_vals, z_vals)):
        ax.bar(x_vals, y, zs=z, zdir='z', label=f'Slice {i+1}', alpha=0.8)

    # Customize the plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y (Bar Height)')
    ax.set_zlabel('Z (Slice Level)')
    ax.set_title('Stacked 2D Bar Graphs in 3D')
    ax.set_xticks(x_vals)  # Set x-ticks to match x_vals
    ax.set_zticks(z_vals)  # Set z-ticks to match z-levels
    ax.legend()

    # Show plot
    plt.show()

    return matrices

