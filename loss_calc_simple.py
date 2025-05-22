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

ping_cytoscape = False

# Real-world network data
# Specify the filename
filename = 'contact_network_text.txt'
# Create the graph from the file
contact_graph = parse.parse(filename)

# Relabel nodes in parsed graph to avoid off by 1 errors in SIR.py
# Create a mapping from old node to new node: i -> i - 1
mapping = {node: node - 1 for node in contact_graph.nodes()}

# Relabel the nodes
contact_network = nx.relabel_nodes(contact_graph, mapping)
social_graph = correlated_graphs.create_w_k_hop_correlation(contact_network,k=2)[0]   # We just want the graph part of this output

# print mean degree of the graph
# mean_degree = np.mean([social_graph.out_degree(node) for node in social_graph])
# print(f"Mean degree of the graph: {mean_degree:.2f}")

#--------
#
#  Observed data (from I.C.)
#
#--------

def calculate_loss(social_network, plot=False):
    cyto_contact = None
    cyto_social = None
    ic_results = None
    # Factor of 0.15 to not have too many seeds
    num_seeds = round(0.15 * len(social_network.nodes()))

    T = 100
    informed = find_seeds.find_seed_set(social_network, num_seeds=num_seeds, exponent=1)
    ic_matrix = np.zeros((T, len(social_network.nodes())))

    for i in range(0, T):
        cur_ic_results = IM.IC_prob_matrix(g=social_network, S=informed, p=0.03, mc=10)  # No changes made to network; no deepcopy needed
        ic_matrix[i] = cur_ic_results[0] # Fill out the quarantine probability matrix
        informed = informed + cur_ic_results[1]  # Update the informed set

    for node in informed:
        nx.set_node_attributes(social_network, {node: 'Informed'}, 'Informed?')

    # if ping_cytoscape == True:
    #     cyto_social = correlated_graphs.generate_from_prob_matrix(social_network, ic_matrix[T-1]) # Adds probabilistic edge weight attributes

    #     # Verify connection to Cytoscape
    #     print(p4c.cytoscape_ping())

    #     # Export the NetworkX graph to Cytoscape
    #     network1 = p4c.create_network_from_networkx(cyto_social, collection="My Network Collection", title=f"Social after I.C.")

    #     # Apply a layout (e.g., force-directed)
    #     p4c.layout_network("force-directed")

    #     # Apply a default visual style
    #     p4c.set_visual_style("default")


    #--------
    #
    #  Inferred data using L.T.
    #
    #--------

    seeds = find_seeds.find_seed_set(social_network, num_seeds=num_seeds, exponent=1)
    losses = []
    for i in range(0,11):  # Vary threshold parameter
        informed = seeds[:]  # Reset informed set for each threshold
        # loss_matrix = np.zeros((T, 1))

        for _ in range(0, T):
            # Inferred data using L.T.
            cur_lt_results = IM.lt_prob_matrix(g=social_network, S=informed, threshold=i)
            informed = list(set(informed + cur_lt_results[1]))  # Update the informed set

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
        
    print("Losses: ", losses)

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

    return losses

# Generate some random graphs (remove just 1 random node to keep them similar)
data = [correlated_graphs.create_w_k_hop_correlation(social_graph, k=1)[0] for _ in range(10)]
# Process each graph: remove a node and relabel nodes for continuity
for item in data:
    for _ in range(0, random.randint(1, 11)):
        # Get the list of nodes
        nodes = list(item.nodes())
        # Choose a random node to remove
        node_to_remove = random.choice(nodes)
        # Remove the node
        item.remove_node(node_to_remove)
        # Create a mapping to relabel nodes
        mapping = {}
        for node in item.nodes():
            if node > node_to_remove:
                mapping[node] = node - 1  # Decrement labels above the removed node
            else:
                mapping[node] = node  # Keep labels below unchanged
        # Relabel the nodes
        nx.relabel_nodes(item, mapping, copy=False)  # In-place relabeling
# Pack the data as a tuple of (losses, connectivity of the graph)
results = [(calculate_loss(graph), round(np.mean([graph.out_degree(node) for node in graph]), 1)) for graph in data]
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

# calculate_loss(social_graph, plot=True)