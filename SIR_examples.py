import networkx as nx
import SIR
import parse
import correlated_graphs
import matplotlib.pyplot as plt
from copy import deepcopy
import pickle
import numpy as np

n = 100
T = 100
Repeat = 1

beta = 0.15  #infection rate
gamma = 0.07  # recovery rate
mu = 0.10   # immunity loss
init = 0.15

# Parameters for misinformation
misinformation = 0.2

# Specify the filename
filename = 'contact_network_text.txt'
# Create the graph from the file
# contact_graph = parse.parse(filename)

# # Relabel nodes in parsed graph to avoid off by 1 errors in SIR.py
# # Create a mapping from old node to new node: i -> i - 1
# mapping = {node: node - 1 for node in contact_graph.nodes()}

# # Relabel the nodes
# contact_network = nx.relabel_nodes(contact_graph, mapping)

contact_network = nx.erdos_renyi_graph(100, 0.05, seed=42)

# # Plot the contact network with pyplot
# print("Mean node degree:", sum(dict(contact_network.degree()).values()) / len(contact_network.nodes()))
# plt.figure(figsize=(10, 10))
# nx.draw(contact_network, with_labels=True, node_size=50, font_size=8, font_color='black', node_color='blue', edge_color='gray')
# plt.title('Contact Network')
# plt.show()


# social_network = correlated_graphs.create_w_k_hop_correlation(contact_network, k=2)[0]  # We just want the graph part of this output
# social_network = correlated_graphs.Jaccard_similarity_plot(contact_network)

social_network = nx.erdos_renyi_graph(100, 0.05, seed=42)  # Placeholder for social network, replace with actual creation logic
social_network = social_network.to_directed()  # Ensure the social network is directed

# Figure 1
# Function to compare SIR runs with and without informed individuals. Quarantines are permanent
# def informed_vs_noninformed():
#     #----------
#     #
#     #  Run SIR with informed individuals
#     #
#     #----------

#     # Average out runs with matplotlib fill-between
#     data1 = SIR.Simulate_SIR(contact_network=deepcopy(contact_network),social_network=deepcopy(social_network),T=T,Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init,average_data=False,q=True,allow_restoration=False)[2]

#     # Extract x and y data from arrays
#     x1, y1 = data1[0], data1[1]

#     #----------
#     #
#     #  Run SIR without informed individuals
#     #
#     #----------

#     data2 = SIR.Simulate_SIR(contact_network=deepcopy(contact_network),social_network=deepcopy(social_network),T=T,Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init,average_data=False,q=False,allow_restoration=False)[2]

#     # Extract x and y data from arrays
#     x2, y2 = data2[0], data2[1]

#     # Create the plot
#     plt.plot(x1, y1, label='With Informed', color='blue', marker='o')
#     plt.plot(x2, y2, label='Without Informed', color='red', marker='o')

#     # Customize the plot
#     plt.xlabel('Time')
#     plt.ylabel('# of Infected')
#     plt.title('Informed vs Non-Informed (Permanent Quarantine)')
#     plt.legend()  # Add legend to distinguish the lines
#     plt.grid(True)

#     # Show the plot
#     plt.show()

#     return data1, data2

def informed_vs_noninformed():
    T_runs = []
    Y_runs_informed = []
    Y_runs_noninformed = []

    num_trials = 10  # Number of simulations to average over

    for _ in range(num_trials):
        # Informed
        data_inf = SIR.Simulate_SIR(
            contact_network=deepcopy(contact_network),
            social_network=deepcopy(social_network),
            T=T, Repeat=Repeat,
            beta=beta, gamma=gamma, mu=mu, init=init,
            average_data=False, q=True, allow_restoration=False
        )[2]
        T_runs.append(data_inf[0])
        Y_runs_informed.append(data_inf[1])

        # Non-informed
        data_noninf = SIR.Simulate_SIR(
            contact_network=deepcopy(contact_network),
            social_network=deepcopy(social_network),
            T=T, Repeat=Repeat,
            beta=beta, gamma=gamma, mu=mu, init=init,
            average_data=False, q=False, allow_restoration=False
        )[2]
        Y_runs_noninformed.append(data_noninf[1])

    # Use the first time array (assumes they're the same for all runs)
    x = T_runs[0]

    # Convert lists to numpy arrays for easier manipulation
    y_informed = np.array(Y_runs_informed)
    y_noninformed = np.array(Y_runs_noninformed)

    # Compute mean and std deviation
    mean_inf = np.mean(y_informed, axis=0)
    std_inf = np.std(y_informed, axis=0)

    mean_noninf = np.mean(y_noninformed, axis=0)
    std_noninf = np.std(y_noninformed, axis=0)

    # Plot with fill_between for both
    plt.plot(x, mean_inf, label='With Informed', color='blue')
    plt.fill_between(x, mean_inf - std_inf, mean_inf + std_inf, color='blue', alpha=0.3)

    plt.plot(x, mean_noninf, label='Without Informed', color='red')
    plt.fill_between(x, mean_noninf - std_noninf, mean_noninf + std_noninf, color='red', alpha=0.3)

    # Customize plot
    plt.xlabel('Time')
    plt.ylabel('# of Infected')
    plt.title('Informed vs Non-Informed (Permanent Quarantine)')
    plt.legend()
    plt.grid(True)

    plt.show()

    return (x, mean_inf, std_inf), (x, mean_noninf, std_noninf)


# Figure 2
# Same as above, but with temporary quarantines
# def const_quarantines():
#     #----------
#     #
#     #  Run SIR with a constant quarantine for informed individuals
#     #
#     #----------

#     quarantine_constant = 14

#     data1 = SIR.Simulate_SIR(contact_network=deepcopy(contact_network),social_network=deepcopy(social_network),T=T,Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init,average_data=False,q=quarantine_constant,allow_restoration=True)[2]

#     # Extract x and y data from arrays
#     x1, y1 = data1[0], data1[1]

#     #----------
#     #
#     #  Run SIR without quarantines
#     #
#     #----------

#     data2 = SIR.Simulate_SIR(contact_network=deepcopy(contact_network),social_network=deepcopy(social_network),T=T,Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init,average_data=False,q=False,allow_restoration=False)[2]

#     # Extract x and y data from arrays
#     x2, y2 = data2[0], data2[1]

#     # Create the plot
#     plt.plot(x1, y1, label=f'With quarantine (constant value of {quarantine_constant})', color='blue', marker='o')
#     plt.plot(x2, y2, label='Without quarantine', color='red', marker='o')

#     # Customize the plot
#     plt.xlabel('Time')
#     plt.ylabel('# of Infected')
#     plt.title('Informed vs Non-Informed (Temporary Quarantine)')
#     plt.legend()  # Add legend to distinguish the lines
#     plt.grid(True)

#     # Show the plot
#     plt.show()

#     return data1, data2

def const_quarantines():
    quarantine_constant = 14
    num_trials = 10

    T_runs = []
    Y_runs_quarantine = []
    Y_runs_noquarantine = []

    for _ in range(num_trials):
        # With constant quarantine
        data1 = SIR.Simulate_SIR(
            contact_network=deepcopy(contact_network),
            social_network=deepcopy(social_network),
            T=T, Repeat=Repeat,
            beta=beta, gamma=gamma, mu=mu, init=init,
            average_data=False, q=quarantine_constant,
            allow_restoration=True
        )[2]
        T_runs.append(data1[0])
        Y_runs_quarantine.append(data1[1])

        # Without quarantine
        data2 = SIR.Simulate_SIR(
            contact_network=deepcopy(contact_network),
            social_network=deepcopy(social_network),
            T=T, Repeat=Repeat,
            beta=beta, gamma=gamma, mu=mu, init=init,
            average_data=False, q=False,
            allow_restoration=False
        )[2]
        Y_runs_noquarantine.append(data2[1])

    x = T_runs[0]
    y_q = np.array(Y_runs_quarantine)
    y_noq = np.array(Y_runs_noquarantine)

    mean_q = np.mean(y_q, axis=0)
    std_q = np.std(y_q, axis=0)

    mean_noq = np.mean(y_noq, axis=0)
    std_noq = np.std(y_noq, axis=0)

    plt.plot(x, mean_q, label=f'With quarantine (q={quarantine_constant})', color='blue')
    plt.fill_between(x, mean_q - std_q, mean_q + std_q, color='blue', alpha=0.3)

    plt.plot(x, mean_noq, label='Without quarantine', color='red')
    plt.fill_between(x, mean_noq - std_noq, mean_noq + std_noq, color='red', alpha=0.3)

    plt.xlabel('Time')
    plt.ylabel('# of Infected')
    plt.title('Informed vs Non-Informed (Temporary Quarantine)')
    plt.legend()
    plt.grid(True)
    plt.show()

    return (x, mean_q, std_q), (x, mean_noq, std_noq)


# Figure 3
# def normal_dist_quarantines():
#     #----------
#     #
#     #  Run SIR with a normal distribution for quarantine times
#     #
#     #----------

#     data1 = SIR.Simulate_SIR(contact_network=deepcopy(contact_network),social_network=deepcopy(social_network),T=T,Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init,average_data=False,q=True,allow_restoration=True)[2]

#     # Extract x and y data from arrays
#     x1, y1 = data1[0], data1[1]

#     #----------
#     #
#     #  Run SIR without quarantines
#     #
#     #----------

#     data2 = SIR.Simulate_SIR(contact_network=deepcopy(contact_network),social_network=deepcopy(social_network),T=T,Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init,average_data=False,q=False,allow_restoration=False)[2]

#     # Extract x and y data from arrays
#     x2, y2 = data2[0], data2[1]

#     # Create the plot
#     plt.plot(x1, y1, label='With quarantine', color='blue', marker='o')
#     plt.plot(x2, y2, label='Without quarantine', color='red', marker='o')

#     # Customize the plot
#     plt.xlabel('Time')
#     plt.ylabel('# of Infected')
#     plt.title('Informed vs Non-Informed (Normal Dist. Quarantine)')
#     plt.legend()  # Add legend to distinguish the lines
#     plt.grid(True)

#     # Show the plot
#     plt.show()

#     return data1, data2

def normal_dist_quarantines():
    num_trials = 10

    T_runs = []
    Y_runs_normal = []
    Y_runs_noquarantine = []

    for _ in range(num_trials):
        # With normally-distributed quarantine
        data1 = SIR.Simulate_SIR(
            contact_network=deepcopy(contact_network),
            social_network=deepcopy(social_network),
            T=T, Repeat=Repeat,
            beta=beta, gamma=gamma, mu=mu, init=init,
            average_data=False, q=True,
            allow_restoration=True
        )[2]
        T_runs.append(data1[0])
        Y_runs_normal.append(data1[1])

        # Without quarantine
        data2 = SIR.Simulate_SIR(
            contact_network=deepcopy(contact_network),
            social_network=deepcopy(social_network),
            T=T, Repeat=Repeat,
            beta=beta, gamma=gamma, mu=mu, init=init,
            average_data=False, q=False,
            allow_restoration=False
        )[2]
        Y_runs_noquarantine.append(data2[1])

    x = T_runs[0]
    y_norm = np.array(Y_runs_normal)
    y_noq = np.array(Y_runs_noquarantine)

    mean_norm = np.mean(y_norm, axis=0)
    std_norm = np.std(y_norm, axis=0)

    mean_noq = np.mean(y_noq, axis=0)
    std_noq = np.std(y_noq, axis=0)

    plt.plot(x, mean_norm, label='With Normal Dist. Quarantine', color='blue')
    plt.fill_between(x, mean_norm - std_norm, mean_norm + std_norm, color='blue', alpha=0.3)

    plt.plot(x, mean_noq, label='Without Quarantine', color='red')
    plt.fill_between(x, mean_noq - std_noq, mean_noq + std_noq, color='red', alpha=0.3)

    plt.xlabel('Time')
    plt.ylabel('# of Infected')
    plt.title('Informed vs Non-Informed (Normal Dist. Quarantine)')
    plt.legend()
    plt.grid(True)
    plt.show()

    return (x, mean_norm, std_norm), (x, mean_noq, std_noq)

# Function to plot Jaccard similarity between contact and social networks (2-hop creation) as X, and existence of edge (0 or 1) as Y
def plot_jaccard_similarity():
    social = correlated_graphs.Jaccard_similarity_plot(contact_network, plot=True)
    return social

# Pickle results from the functions
def SIR_pickle_dump(filename='pickles.pkl'):
    # We will pickle these parameters along with the results for later reference
    presets = {'T': T, 'Repeat': Repeat, 'beta': beta, 'gamma': gamma, 'mu': mu, 'init': init}

    informed_vs_noninformed()
    const_quarantines()
    normal_dist_quarantines()
    # data7 = plot_jaccard_similarity()

    with open(filename, 'wb') as f:
        # Clear the file before writing
        f.truncate(0)

        pickle.dump({'presets': presets}, f)
        # pickle.dump({'data1': data1, 'data2': data2}, f)
        # pickle.dump({'data3': data3, 'data4': data4}, f)
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