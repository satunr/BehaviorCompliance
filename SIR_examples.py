import networkx as nx
import SIR
import parse
import correlated_graphs
import matplotlib.pyplot as plt
from copy import deepcopy
import pickle

n = 100
T = 60
Repeat = 1

beta = 0.20  #infection rate
gamma = 0.07  # recovery rate
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
# social_network = correlated_graphs.create_w_k_hop_correlation(contact_network, k=2)[0]  # We just want the graph part of this output
social_network = correlated_graphs.Jaccard_similarity_plot(contact_network)

# Figure 1
# Function to compare SIR runs with and without informed individuals. Quarantines are permanent
def informed_vs_noninformed():
    #----------
    #
    #  Run SIR with informed individuals
    #
    #----------

    data1 = SIR.Simulate_SIR(contact_network=deepcopy(contact_network),social_network=deepcopy(social_network),T=T,Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init,average_data=False,q=True,allow_restoration=False)[2]

    # Extract x and y data from arrays
    x1, y1 = data1[0], data1[1]

    #----------
    #
    #  Run SIR without informed individuals
    #
    #----------

    data2 = SIR.Simulate_SIR(contact_network=deepcopy(contact_network),social_network=deepcopy(social_network),T=T,Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init,average_data=False,q=False,allow_restoration=False)[2]

    # Extract x and y data from arrays
    x2, y2 = data2[0], data2[1]

    # Create the plot
    plt.plot(x1, y1, label='With Informed', color='blue', marker='o')
    plt.plot(x2, y2, label='Without Informed', color='red', marker='o')

    # Customize the plot
    plt.xlabel('Time')
    plt.ylabel('# of Infected')
    plt.title('Informed vs Non-Informed (Permanent Quarantine)')
    plt.legend()  # Add legend to distinguish the lines
    plt.grid(True)

    # Show the plot
    plt.show()

    return data1, data2

# Figure 2
# Same as above, but with temporary quarantines
def const_quarantines():
    #----------
    #
    #  Run SIR with a constant quarantine for informed individuals
    #
    #----------

    quarantine_constant = 25

    data1 = SIR.Simulate_SIR(contact_network=deepcopy(contact_network),social_network=deepcopy(social_network),T=T,Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init,average_data=False,q=quarantine_constant,allow_restoration=True)[2]

    # Extract x and y data from arrays
    x1, y1 = data1[0], data1[1]

    #----------
    #
    #  Run SIR without quarantines
    #
    #----------

    data2 = SIR.Simulate_SIR(contact_network=deepcopy(contact_network),social_network=deepcopy(social_network),T=T,Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init,average_data=False,q=False,allow_restoration=False)[2]

    # Extract x and y data from arrays
    x2, y2 = data2[0], data2[1]

    # Create the plot
    plt.plot(x1, y1, label=f'With quarantine (constant value of {quarantine_constant})', color='blue', marker='o')
    plt.plot(x2, y2, label='Without quarantine', color='red', marker='o')

    # Customize the plot
    plt.xlabel('Time')
    plt.ylabel('# of Infected')
    plt.title('Informed vs Non-Informed (Temporary Quarantine)')
    plt.legend()  # Add legend to distinguish the lines
    plt.grid(True)

    # Show the plot
    plt.show()

    return data1, data2

# Figure 3
def normal_dist_quarantines():
    #----------
    #
    #  Run SIR with a normal distribution for quarantine times
    #
    #----------

    data1 = SIR.Simulate_SIR(contact_network=deepcopy(contact_network),social_network=deepcopy(social_network),T=T,Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init,average_data=False,q=True,allow_restoration=True)[2]

    # Extract x and y data from arrays
    x1, y1 = data1[0], data1[1]

    #----------
    #
    #  Run SIR without quarantines
    #
    #----------

    data2 = SIR.Simulate_SIR(contact_network=deepcopy(contact_network),social_network=deepcopy(social_network),T=T,Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init,average_data=False,q=False,allow_restoration=False)[2]

    # Extract x and y data from arrays
    x2, y2 = data2[0], data2[1]

    # Create the plot
    plt.plot(x1, y1, label='With quarantine', color='blue', marker='o')
    plt.plot(x2, y2, label='Without quarantine', color='red', marker='o')

    # Customize the plot
    plt.xlabel('Time')
    plt.ylabel('# of Infected')
    plt.title('Informed vs Non-Informed (Normal Dist. Quarantine)')
    plt.legend()  # Add legend to distinguish the lines
    plt.grid(True)

    # Show the plot
    plt.show()

    return data1, data2

# Function to plot Jaccard similarity between contact and social networks (2-hop creation) as X, and existence of edge (0 or 1) as Y
def plot_jaccard_similarity():
    social = correlated_graphs.Jaccard_similarity_plot(contact_network)
    return social

# Pickle results from the functions
def SIR_pickle_dump(filename='pickles.pkl'):
    # We will pickle these parameters along with the results for later reference
    presets = {'T': T, 'Repeat': Repeat, 'beta': beta, 'gamma': gamma, 'mu': mu, 'init': init}

    # data1, data2 = informed_vs_noninformed()
    data3, data4 = const_quarantines()
    # data5, data6 = normal_dist_quarantines()
    # data7 = plot_jaccard_similarity()

    with open(filename, 'wb') as f:
        # Clear the file before writing
        f.truncate(0)

        pickle.dump({'presets': presets}, f)
        # pickle.dump({'data1': data1, 'data2': data2}, f)
        pickle.dump({'data3': data3, 'data4': data4}, f)
        # pickle.dump({'data5': data5, 'data6': data6}, f)
        # pickle.dump({'data7': data7}, f)
    print("Data has been pickled successfully.")

SIR_pickle_dump()

def pickle_load(filename='pickles.pkl'):
    # Open the file in binary read mode
    with open(filename, 'rb') as file:
        data = pickle.load(file)

    # Now `data` holds the deserialized object
    print(data)

# def misinformation_comp():
#     #----------
#     #
#     #  Run SIR with misinformation
#     #
#     #----------

#     social_network = correlated_graphs.create_w_k_hop_correlation(contact_network,k=2)[0]   # We just want the graph part of this output

#     # NOTE: This data is derived from 2 separate runs of Simulate_SIR, and is therefore only an approx. comparison
#     data1 = SIR.Simulate_SIR(contact_network=deepcopy(contact_network),social_network=deepcopy(social_network),T=T,Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init,
#                              average_data=False,q=True,allow_restoration=True, misinformation_prob=None)[2]
#     data2 = SIR.Simulate_SIR(contact_network=deepcopy(contact_network),social_network=deepcopy(social_network),T=T,Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init,
#                              average_data=False,q=True,allow_restoration=False, misinformation_prob=misinformation)[2]

#     # Extract x and y data from arrays
#     x1, y1 = data1[0], data1[1]
#     x2, y2 = data2[0], data2[1]

#     # Create the plot
#     plt.plot(x1, y1, label='Without Misinformation', color='blue', marker='o')
#     plt.plot(x2, y2, label='With Misinformation', color='red', marker='o')

#     # Customize the plot
#     plt.xlabel('Time')
#     plt.ylabel('# of Infected')
#     plt.title('Misinformation Comparison')
#     plt.legend()  # Add legend to distinguish the lines
#     plt.grid(True)

#     # Show the plot
#     plt.show()

# misinformation_comp()