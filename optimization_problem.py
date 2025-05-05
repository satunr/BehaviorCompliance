import networkx as nx
import matplotlib.pyplot as plt
import parse
import correlated_graphs
import SIR
import numpy as np

#---------
#
#  Optimization problem: Minimize loss between I.C., L.T. models, 
#    to arrive at a more deterministic construction of unknown social network
#    Assumption: Threshold value is the same for all nodes
#    f: Z -> Z, and we want f(X*) approx A in Z, where A is the observed I.C. results
#      Learnable parameter: Tau - Threshold for L.T. simulation.
#
#---------

T = 200
Repeat = 1

beta = 0.10  #infection rate
gamma = 0.05  # recovery rate
mu = 0.10   # immunity loss
init = 0.05

# Real-world network data
# Specify the filename
# filename = 'contact_network_text.txt'
# # Create the graph from the file
# contact_graph = parse.parse(filename)

# Test on tree-like network
depth = 6
contact_network = nx.balanced_tree(r=3, h=depth-1)  # r=2 for binary tree, h=depth-1 gives 2^7-1 nodes
print("Num of nodes: ", len(contact_network.nodes()))

# Relabel nodes in parsed graph to avoid off by 1 errors in SIR.py
# Create a mapping from old node to new node: i -> i - 1
# mapping = {node: node - 1 for node in contact_graph.nodes()}

# Relabel the nodes
# contact_network = nx.relabel_nodes(contact_graph, mapping)
social_network = correlated_graphs.create_w_k_hop_correlation(contact_network,k=2)[0]   # We just want the graph part of this output

# Set NumPy print options to show all rows and columns data
np.set_printoptions(threshold=np.inf)

#--------
#
#  Observed data (from I.C.)
#
#--------

# Average the data
data1_avg = []
# Array of arrays of quarantine statuses
ic_results = SIR.Simulate_SIR(contact_network=contact_network,social_network=social_network,T=T,Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init,average_data=False,q=True,allow_restoration=True,save_all=True)
data1 = ic_results[3]
data1 = np.array(data1, dtype=float)
# Normalize: Set non-zero values to 1, keep zeros as 0
data1 = np.where(data1 != 0, 1, 0)

#-------
#
#  Inferred data using L.T.
#
#-------

losses = []
for i in range(1,30):
    # Inferred data using L.T.
    lt_results = SIR.Simulate_SIR(contact_network=contact_network,social_network=social_network,T=T,Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init,average_data=False,q=True,allow_restoration=True,save_all=True,lt_threshold=i)
    data2 = lt_results[3]
    data2 = np.array(data2, dtype=float)
    # Normalize: Set non-zero values to 1, keep zeros as 0
    data2 = np.where(data2 != 0, 1, 0)

    # Calculate loss between observed, inferred
    loss = (data1 - data2) ** 2
    loss = np.sum(loss)
    losses.append(loss)
    print(f"loss with threshold of {i}: {loss}")

x_vals = list(range(len(losses)))  # Integer x-values: 0 to 99

# Create the bar graph
plt.figure(figsize=(12, 6))
plt.bar(x_vals, losses, color='skyblue', edgecolor='black')
plt.axhline(y=np.sum(data1), color='red', linestyle='--', linewidth=2, label='y = observed result')
plt.xlabel('Node Index')
plt.ylabel('Node Degree')
plt.title('Loss between average observed and inferred quarantine data')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
