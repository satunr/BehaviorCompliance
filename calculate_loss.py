import networkx as nx
import matplotlib.pyplot as plt
import parse
import correlated_graphs
import SIR
import numpy as np
import py4cytoscape as p4c
from copy import deepcopy

ping_cytoscape = True

# Lazy way of parsing the loss data from the HPC, if used
def parse_loss_data(file_path):
    thresholds = []
    losses = []
    
    with open(file_path, 'r') as file:
        for line in file:
            # Remove leading/trailing whitespace
            line = line.strip()
            if line.startswith('loss with threshold of'):
                # Split the line into parts
                parts = line.split(':')
                if len(parts) == 2:
                    # Extract threshold (between 'of' and ':')
                    threshold_part = parts[0].split('of')[1].strip()
                    # Extract loss
                    loss_part = parts[1].strip()
                    try:
                        threshold = int(threshold_part)
                        loss = float(loss_part)
                        thresholds.append(threshold)
                        losses.append(loss)
                    except ValueError:
                        continue  # Skip lines that can't be converted to numbers
    
    return thresholds, losses


#---------
#
#  Optimization problem: Minimize loss between I.C., L.T. models, 
#    to arrive at a deterministic construction of unknown social network
#    Assumption: Threshold value is the same for all nodes
#    f: Z -> R, and we want f(X*) approx A, where A is the observed I.C. results
#      Learnable parameter: Tau - Threshold for L.T. simulation.
#
#---------

T = 50
Repeat = 1
beta = 0.15  #infection rate
gamma = 0.05  # recovery rate
mu = 0.10   # immunity loss
init = 0.3  # initial fraction of infected nodes

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

ic_results = SIR.Simulate_SIR(contact_network=deepcopy(contact_network),social_network=deepcopy(social_network),T=T,Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init,average_data=False,q=True,allow_restoration=True,save_all=True)
data1 = ic_results[3]

if ping_cytoscape == True:
    cyto_contact = ic_results[0]

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
    # Inferred data using L.T.
    lt_results = SIR.Simulate_SIR(contact_network=deepcopy(contact_network),social_network=deepcopy(social_network),T=T,Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init,average_data=False,q=True,allow_restoration=True,save_all=True,lt_threshold=i)

    data2 = lt_results[3]

    # Calculate loss between observed, inferred
    loss = np.abs(data1 - data2)
    loss = np.sum(loss) # Sum over times and nodes
    loss = np.round(loss, 2)
    losses.append(loss)
    print(f"loss with threshold of {i}: {loss}")

    if ping_cytoscape == True:
        cyto_contact = correlated_graphs.generate_from_prob_matrix(lt_results[0]) # Adds probabilistic edge weight attributes
        cyto_social = lt_results[5]

        # Verify connection to Cytoscape
        print(p4c.cytoscape_ping())

        # Export the NetworkX graph to Cytoscape
        network1 = p4c.create_network_from_networkx(cyto_contact, collection="My Network Collection", title=f"Contact after L.T. (Tau = {i})")
        
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