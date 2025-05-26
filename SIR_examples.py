import networkx as nx
import SIR
import parse
import correlated_graphs
import matplotlib.pyplot as plt
from copy import deepcopy

n = 100
T = 60
Repeat = 1

beta = 0.20  #infection rate
gamma = 0.03  # recovery rate
mu = 0.10   # immunity loss
init = 0.1

# Parameters for misinformation
misinformation = 0.2

# Specify the filename
filename = 'contact_network_text.txt'
# Create the graph from the file
contact_graph = parse.parse(filename)

# Relabel nodes in parsed graph to avoid off by 1 errors in SIR.py
# Create a mapping from old node to new node: i -> i - 1
mapping = {node: node - 1 for node in contact_graph.nodes()}

# Relabel the nodes
contact_network = nx.relabel_nodes(contact_graph, mapping)

def social_network_comp():
    #---------
    #
    #  Generate social network w/ 1-hop correlation
    #
    #---------

    social_network = correlated_graphs.create_w_k_hop_correlation(contact_network,k=1)[0]   # We just want the graph part of this output

    # NOTE: This data is derived from 2 separate runs of Simulate_SIR, and is therefore only an approx. comparison
    data1 = SIR.Simulate_SIR(contact_network=deepcopy(contact_network),social_network=deepcopy(social_network),T=T,Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init,average_data=False,q=True,allow_restoration=True)[2]
    data2 = SIR.Simulate_SIR(contact_network=deepcopy(contact_network),social_network=deepcopy(social_network),T=T,Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init,average_data=False,q=True,allow_restoration=False)[2]
    data3 = SIR.Simulate_SIR(contact_network=deepcopy(contact_network),social_network=deepcopy(social_network),T=T,Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init,average_data=False,q=False,allow_restoration=False)[2]

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

    #----------
    #
    #  Generate social network with 2-hop correlation
    #
    #----------

    social_network = correlated_graphs.create_w_k_hop_correlation(contact_network,k=2)[0]   # We just want the graph part of this output

    # NOTE: This data is derived from 2 separate runs of Simulate_SIR, and is therefore only an approx. comparison
    data1 = SIR.Simulate_SIR(contact_network=deepcopy(contact_network),social_network=deepcopy(social_network),T=T,Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init,average_data=False,q=True,allow_restoration=True)[2]
    data2 = SIR.Simulate_SIR(contact_network=deepcopy(contact_network),social_network=deepcopy(social_network),T=T,Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init,average_data=False,q=True,allow_restoration=False)[2]
    data3 = SIR.Simulate_SIR(contact_network=deepcopy(contact_network),social_network=deepcopy(social_network),T=T,Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init,average_data=False,q=False,allow_restoration=False)[2]

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

def misinformation_comp():
    #----------
    #
    #  Run SIR with misinformation
    #
    #----------

    social_network = correlated_graphs.create_w_k_hop_correlation(contact_network,k=2)[0]   # We just want the graph part of this output

    # NOTE: This data is derived from 2 separate runs of Simulate_SIR, and is therefore only an approx. comparison
    data1 = SIR.Simulate_SIR(contact_network=deepcopy(contact_network),social_network=deepcopy(social_network),T=T,Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init,
                             average_data=False,q=True,allow_restoration=True, misinformation_prob=None)[2]
    data2 = SIR.Simulate_SIR(contact_network=deepcopy(contact_network),social_network=deepcopy(social_network),T=T,Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init,
                             average_data=False,q=True,allow_restoration=False, misinformation_prob=misinformation)[2]

    # Extract x and y data from arrays
    x1, y1 = data1[0], data1[1]
    x2, y2 = data2[0], data2[1]

    # Create the plot
    plt.plot(x1, y1, label='Without Misinformation', color='blue', marker='o')
    plt.plot(x2, y2, label='With Misinformation', color='red', marker='o')

    # Customize the plot
    plt.xlabel('Time')
    plt.ylabel('# of Infected')
    plt.title('Misinformation Comparison')
    plt.legend()  # Add legend to distinguish the lines
    plt.grid(True)

    # Show the plot
    plt.show()

misinformation_comp()

