import networkx as nx
import matplotlib.pyplot as plt
import parse
import correlated_graphs
import IM
import find_seeds
import numpy as np
import py4cytoscape as p4c
from copy import deepcopy

ping_cytoscape = False

# Real-world network data
# Specify the filename
filename = 'contact_network_text.txt'
# # Create the graph from the file
contact_graph = parse.parse(filename)

# Relabel nodes in parsed graph to avoid off by 1 errors in SIR.py
# Create a mapping from old node to new node: i -> i - 1
mapping = {node: node - 1 for node in contact_graph.nodes()}

# Relabel the nodes
contact_network = nx.relabel_nodes(contact_graph, mapping)
social_network = correlated_graphs.create_w_k_hop_correlation(contact_network,k=2)[0]   # We just want the graph part of this output
copy_social_network = deepcopy(social_network)

# Set NumPy print options to show all rows and columns data
np.set_printoptions(threshold=np.inf)

#--------
#
#  Observed data (from I.C.)
#
#--------

cyto_contact = None
cyto_social = None
ic_results = None

T = 3
informed = find_seeds.find_seed_set(social_network, num_seeds=10, exponent=1)
ic_matrix = np.zeros((T, len(social_network.nodes())))

for i in range(0, T):
    cur_ic_results = IM.IC_prob_matrix(g=social_network, S=informed, p=0.03)
    ic_matrix[i] = cur_ic_results[0] # Fill out the quarantine probability matrix
    informed = informed + cur_ic_results[1]  # Update the informed set

if ping_cytoscape == True:
    cyto_social = correlated_graphs.generate_from_prob_matrix(ic_matrix) # Adds probabilistic edge weight attributes

    # Verify connection to Cytoscape
    print(p4c.cytoscape_ping())

    # Export the NetworkX graph to Cytoscape
    network1 = p4c.create_network_from_networkx(cyto_contact, collection="My Network Collection", title=f"Contact after I.C.")

    # Apply a layout (e.g., force-directed)
    p4c.layout_network("force-directed")

    # Apply a default visual style
    p4c.set_visual_style("default")


#--------
#
#  Inferred data using L.T.
#
#--------

losses = []
for i in range(0,11):
    social_network = copy_social_network
    lt_matrix = np.zeros((T, len(social_network.nodes())))

    for j in range(0, T):
        # Inferred data using L.T.
        cur_lt_results = IM.lt_prob_matrix(g=deepcopy(social_network), S=informed, threshold=i)
        lt_matrix[j] = cur_lt_results[0]  # Fill out the quarantine probability matrix
        informed = informed + cur_lt_results[1]  # Update the informed set

    # Calculate loss between observed, inferred
    loss = np.abs(ic_matrix - lt_matrix)
    loss = np.sum(loss) # Sum over times and nodes
    loss = np.round(loss, 2)
    losses.append(loss)

    print(f"loss with threshold of {i}: {loss}")

    if ping_cytoscape == True:
        cyto_social = correlated_graphs.generate_from_prob_matrix(lt_matrix) # Adds probabilistic edge weight attributes

        # Verify connection to Cytoscape
        print(p4c.cytoscape_ping())

        # Export the NetworkX graph to Cytoscape
        network1 = p4c.create_network_from_networkx(cyto_social, collection="My Network Collection", title=f"Social after L.T. (Tau = {i})")
        
        # Apply a layout (e.g., force-directed)
        p4c.layout_network("force-directed")

        # Apply a default visual style
        p4c.set_visual_style("default")
    
# Ensure graph starts at right x value
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