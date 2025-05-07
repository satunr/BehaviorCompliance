import networkx as nx
import SIR
import parse
import correlated_graphs
import matplotlib.pyplot as plt


n = 100
T = 60
Repeat = 1

beta = 0.10  #infection rate
gamma = 0.05  # recovery rate
mu = 0.10   # immunity loss
init = 0.05

# Specify the filename
filename = 'contact_network_text.txt'
# Create the graph from the file
contact_graph = parse.parse(filename)

# Test on a treelike network
# depth = 5
# contact_network = nx.balanced_tree(r=3, h=depth-1)  # r=2 for binary tree, h=depth-1 gives 2^7-1 nodes

# Relabel nodes in parsed graph to avoid off by 1 errors in SIR.py
# Create a mapping from old node to new node: i -> i - 1
mapping = {node: node - 1 for node in contact_graph.nodes()}

# Relabel the nodes
contact_network = nx.relabel_nodes(contact_graph, mapping)

#---------
#
#  Generate social network w/ 1-hop correlation
#
#---------

social_network = correlated_graphs.create_w_k_hop_correlation(contact_network,k=1)[0]   # We just want the graph part of this output

# NOTE: This data is derived from 2 separate runs of Simulate_SIR, and is therefore only an approx. comparison
data1 = SIR.Simulate_SIR(contact_network=contact_network,social_network=social_network,T=T,Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init,average_data=False,q=True,allow_restoration=True)[2]
data2 = SIR.Simulate_SIR(contact_network=contact_network,social_network=social_network,T=T,Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init,average_data=False,q=True,allow_restoration=False)[2]
data3 = SIR.Simulate_SIR(contact_network=contact_network,social_network=social_network,T=T,Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init,average_data=False,q=False,allow_restoration=False)[2]

# Extract x and y data from arrays
x1, y1 = data1[0], data1[1]
x2, y2 = data2[0], data2[1]
x3, y3 = data3[0], data3[1]

# Create the plot
plt.plot(x1, y1, label='With Restoration', color='blue', marker='o')
plt.plot(x2, y2, label='Without Restoration', color='red', marker='o')
plt.plot(x3, y3, label='No quarantine', color='green', marker='o')

# Customize the plot
plt.xlabel('Time')
plt.ylabel('# of Infected')
plt.title('Quarantine Comparison Social 1-hop')
plt.legend()  # Add legend to distinguish the lines
plt.grid(True)

# Show the plot
plt.show()

print("Social constructed from 1-hop")
print("data1: ", data1)
print("data2: ", data2)
print("data3: ", data3)


#----------
#
#  Generate social network with 2-hop correlation
#
#----------

social_network = correlated_graphs.create_w_k_hop_correlation(contact_network,k=2)[0]   # We just want the graph part of this output

# NOTE: This data is derived from 2 separate runs of Simulate_SIR, and is therefore only an approx. comparison
data1 = SIR.Simulate_SIR(contact_network=contact_network,social_network=social_network,T=T,Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init,average_data=False,q=True,allow_restoration=True)[2]
data2 = SIR.Simulate_SIR(contact_network=contact_network,social_network=social_network,T=T,Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init,average_data=False,q=True,allow_restoration=False)[2]
data3 = SIR.Simulate_SIR(contact_network=contact_network,social_network=social_network,T=T,Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init,average_data=False,q=False,allow_restoration=False)[2]

# Extract x and y data from arrays
x1, y1 = data1[0], data1[1]
x2, y2 = data2[0], data2[1]
x3, y3 = data3[0], data3[1]

# Create the plot
plt.plot(x1, y1, label='With Restoration', color='blue', marker='o')
plt.plot(x2, y2, label='Without Restoration', color='red', marker='o')
plt.plot(x3, y3, label='No quarantine', color='green', marker='o')

# Customize the plot
plt.xlabel('Time')
plt.ylabel('# of Infected')
plt.title('Quarantine Comparison Social 2-hop')
plt.legend()  # Add legend to distinguish the lines
plt.grid(True)

# Show the plot
plt.show()

print("Social constructed from 1-hop")
print("data1: ", data1)
print("data2: ", data2)
print("data3: ", data3)