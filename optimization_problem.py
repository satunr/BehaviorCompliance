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

# Specify the filename
filename = 'contact_network_text.txt'
# Create the graph from the file
contact_graph = parse.parse(filename)
# Relabel nodes in parsed graph to avoid off by 1 errors in SIR.py
# Create a mapping from old node to new node: i -> i - 1
mapping = {node: node - 1 for node in contact_graph.nodes()}

# Relabel the nodes
contact_network = nx.relabel_nodes(contact_graph, mapping)
social_network = correlated_graphs.create_w_k_hop_correlation(contact_network,k=1)[0]   # We just want the graph part of this output

# Set NumPy print options to show all rows and columns for the following data
np.set_printoptions(threshold=np.inf)

# Array of arrays of quarantine statuses
ic_results = SIR.Simulate_SIR(contact_network=contact_network,social_network=social_network,T=T,Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init,verbose=False,q=True,allow_restoration=True,save_all=True)
data1 = ic_results[3]
data1 = np.array(data1, dtype=float)
# Normalize: Set non-zero values to 1, keep zeros as 0
data1 = np.where(data1 != 0, 1, 0)

print("data1 normalized: ")
print(data1)

lt_results = SIR.Simulate_SIR(contact_network=contact_network,social_network=social_network,T=T,Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init,verbose=False,q=True,allow_restoration=True,save_all=True,lt_threshold=3)
data2 = lt_results[3]
infections = lt_results[4]
data2 = np.array(data2, dtype=float)
# Normalize: Set non-zero values to 1, keep zeros as 0
data2 = np.where(data2 != 0, 1, 0)

print("data2 normalized: ")
print(data2)

print("infections: ", infections)
# conversion: Extract values from each dictionary
infections_array = np.array([list(d.values()) for d in infections], dtype=np.int64)
# Normalize: Set non-zero values to 1, keep zeros as 0
infections = np.where(infections != 0, 1, 0)

Q = data2 * infections
loss = (data1 - Q) ** 2
loss = np.sum(loss)
print("loss: ", loss)

