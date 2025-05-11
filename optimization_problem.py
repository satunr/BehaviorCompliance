import networkx as nx
import matplotlib.pyplot as plt
import parse
import correlated_graphs
import SIR
import numpy as np
import py4cytoscape as p4c

ping_cytoscape = True

# Achieves rough estimate of "average" SIR quarantine data
def modify_array(arr: np.array):
    # Step 0: Keep zeros as 0 (no action needed yet)
    
    # Step 1: Find the minimum positive value (> 0)
    positive_vals = arr[arr > 0]
    if positive_vals.size == 0:
        # Handle case where there are no positive values
        print("No positive values in array. Returning original array.")
        return arr.copy()
    
    min_val = np.min(positive_vals)
    
    # Step 2: Create a copy of the array to modify
    result = arr.copy()
    
    # Step 3: Replace min_val with 0
    result[result == min_val] = 0
    
    # Step 4: Replace all other non-zero values with 1
    result[(result != 0) & (result > 0)] = 1
    
    return result

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
#    f: Z -> Z, and we want f(X*) approx A in Z, where A is the observed I.C. results
#      Learnable parameter: Tau - Threshold for L.T. simulation.
#
#---------

T = 50
Repeat = 1

beta = 0.09  #infection rate
gamma = 0.05  # recovery rate
mu = 0.10   # immunity loss
init = 0.03

# Real-world network data
# Specify the filename
filename = 'contact_network_text.txt'
# # Create the graph from the file
contact_graph = parse.parse(filename)

# Test on tree-like network
# depth = 6
# contact_network = nx.balanced_tree(r=3, h=depth-1)  # r=2 for binary tree, h=depth-1 gives 2^7-1 nodes
# print("Num of nodes: ", len(contact_network.nodes()))

# Test on a highly-connected graph
# Create Erdős-Rényi graph with 100 nodes and edge probability p=0.5
# contact_network = nx.erdos_renyi_graph(100, 0.3)

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

data1_avg = []
cyto_contact = None
cyto_social = None
ic_results = None

for i in range(0,6):
    # Array of arrays of quarantine statuses
    ic_results = SIR.Simulate_SIR(contact_network=contact_network,social_network=social_network,T=T,Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init,average_data=False,q=True,allow_restoration=True,save_all=True)

    data1 = ic_results[3]
    data1 = np.array(data1, dtype=float)
    # Normalize: Set non-zero values to 1, keep zeros as 0
    data1 = np.where(data1 != 0, 1, 0)
    data1_avg.append(data1)

ic_results = SIR.Simulate_SIR(contact_network=contact_network,social_network=social_network,T=T,Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init,average_data=False,q=True,allow_restoration=True,save_all=True)
data1 = ic_results[3]
data1 = np.array(data1, dtype=float)
# Normalize: Set non-zero values to 1, keep zeros as 0
data1 = np.where(data1 != 0, 1, 0)

cyto_contact = ic_results[0]
cyto_social = ic_results[5]

if ping_cytoscape == True:
    # Verify connection to Cytoscape
    print(p4c.cytoscape_ping())

    # Export the NetworkX graph to Cytoscape
    network1 = p4c.create_network_from_networkx(cyto_contact, collection="My Network Collection", title="Contact after I.C.")

    # Apply a layout (e.g., force-directed)
    p4c.layout_network("force-directed")

    # Apply a default visual style
    p4c.set_visual_style("default")


    # Verify connection to Cytoscape
    print(p4c.cytoscape_ping())

    # Export the NetworkX graph to Cytoscape
    network2 = p4c.create_network_from_networkx(cyto_social, collection="My Network Collection", title="Social after I.C.")

    # Apply a layout (e.g., force-directed)
    p4c.layout_network("force-directed")

    # Apply a default visual style
    p4c.set_visual_style("default")



#-------
#
#  Inferred data using L.T.
#
#-------

losses = []
for i in range(2,11):
    # Inferred data using L.T.
    lt_results = SIR.Simulate_SIR(contact_network=contact_network,social_network=social_network,T=T,Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init,average_data=False,q=True,allow_restoration=True,save_all=True,lt_threshold=i)
    cyto_contact = lt_results[0]
    cyto_social = lt_results[5]

    data2 = lt_results[3]
    data2 = np.array(data2, dtype=float)
    # Normalize: Set non-zero values to 1, keep zeros as 0
    data2 = np.where(data2 != 0, 1, 0)

    # Calculate loss between observed, inferred
    loss = (data1 - data2) ** 2
    loss = np.sum(loss)
    losses.append(loss)
    print(f"loss with threshold of {i}: {loss}")

    if ping_cytoscape == True:
        # Verify connection to Cytoscape
        print(p4c.cytoscape_ping())

        # Export the NetworkX graph to Cytoscape
        network1 = p4c.create_network_from_networkx(cyto_contact, collection="My Network Collection", title=f"Contact after L.T. (Tau = {i})")

        # Apply a layout (e.g., force-directed)
        p4c.layout_network("force-directed")

        # Apply a default visual style
        p4c.set_visual_style("default")


        # Verify connection to Cytoscape
        print(p4c.cytoscape_ping())

        # Export the NetworkX graph to Cytoscape
        network2 = p4c.create_network_from_networkx(cyto_social, collection="My Network Collection", title=f"Social after L.T. (Tau = {i})")

        # Apply a layout (e.g., force-directed)
        p4c.layout_network("force-directed")

        # Apply a default visual style
        p4c.set_visual_style("default")
    
x_vals = [i + 2 for i in list(range(len(losses)))]

# Create the bar graph
plt.figure(figsize=(12, 6))
plt.bar(x_vals, losses, color='skyblue', edgecolor='black')
plt.axhline(y=np.sum(data1), color='red', linestyle='--', linewidth=2, label='y = observed result')
plt.xlabel('Threshold')
plt.ylabel('Loss wrt # of informed')
plt.title('Loss between average observed and inferred quarantine data')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()



# Parse the data (assuming the data is in a file named 'loss_vals.txt')
# thresholds, losses = parse_loss_data('loss_vals.txt')

# # Create the plot
# plt.figure(figsize=(12, 6))
# plt.plot(thresholds, losses, 'b-', label='Loss vs Threshold')
# plt.title('Loss as a Function of Threshold', fontsize=14)
# plt.xlabel('Threshold', fontsize=12)
# plt.ylabel('Loss', fontsize=12)
# plt.grid(True)
# plt.legend()

# Adjust y-axis to accommodate large values
# plt.ylim(min(losses) - 100, max(losses) + 100)

# Save the plot
# plt.savefig('loss_vs_threshold.png')