# import SIR
# import networkx as nx
# from copy import deepcopy
# import matplotlib.pyplot as plt
# import numpy as np
# import correlated_graphs
# import pickle

# n = 100
# T = 100
# Repeat = 1

# beta = 0.15  # infection rate
# gamma = 0.07  # recovery rate
# mu = 0.10     # immunity loss
# init = 0.15

# H = nx.read_gml('Freeman3.gml')
# H = nx.convert_node_labels_to_integers(H, first_label=0)
# contact_graph = H.to_undirected()

# print("# of nodes in contact graph:", contact_graph.number_of_nodes())

# num_trials = 10

# # Run the simulation once to get the observed/true dynamics
# T_runs = []
# Y_runs_normal = []

# for _ in range(num_trials):
#     data1 = SIR.Simulate_SIR(
#         contact_network=deepcopy(contact_graph), social_network=None,
#         T=T, Repeat=Repeat,
#         beta=beta, gamma=gamma, mu=mu, init=init,
#         average_data=False, q=True,
#         allow_restoration=True
#     )[2]
#     T_runs.append(data1[0])
#     Y_runs_normal.append(data1[1])

# x = T_runs[0]
# y_norm = np.array(Y_runs_normal)
# mean_norm = np.mean(y_norm, axis=0)
# std_norm = np.std(y_norm, axis=0)

# # Plot the observed SIR data once
# plt.plot(x, mean_norm, label='Observed SIR data', color='blue')
# plt.fill_between(x, mean_norm - std_norm, mean_norm + std_norm, color='blue', alpha=0.3)

# # # Now add inferred runs with varying social graph edge counts
# nE_min = 50
# nE_max = 150
# step = 10
# KL_divergences = []
# peak_uncertainties = []

# for num_edges in range(nE_min, nE_max, step):
#     assumed_social_graph = correlated_graphs.create_social_graph(contact_graph, nE=num_edges)[0]

#     T_runs_infer = []
#     Y_runs_normal_infer = []

#     for _ in range(num_trials):
#         data2 = SIR.Simulate_SIR(
#             contact_network=deepcopy(contact_graph), social_network=deepcopy(assumed_social_graph),
#             T=T, Repeat=Repeat,
#             beta=beta, gamma=gamma, mu=mu, init=init,
#             average_data=False, q=True,
#             allow_restoration=True
#         )[2]
#         T_runs_infer.append(data2[0])
#         Y_runs_normal_infer.append(data2[1])

#     x_infer = T_runs_infer[0]
#     y_norm_infer = np.array(Y_runs_normal_infer)
#     mean_norm_infer = np.mean(y_norm_infer, axis=0)
#     std_norm_infer = np.std(y_norm_infer, axis=0)

#     peak_uncertainty = np.argmax(mean_norm_infer + std_norm_infer)  # Find the peak of the uncertainty of the inferred SIR curve
#     peak_uncertainties.append(peak_uncertainty)

#     # Compute the KL divergences. Simple assumption: 
#     # assume that P(i is infected) = mean # of infected at time t / N, where N is the number of nodes in the contact graph.
#     sum = 0.0
#     N = len(contact_graph.nodes())
#     for i, val in enumerate(mean_norm_infer):
#         sum += (mean_norm[i] / N) * (np.log(mean_norm[i] / val)) if val > 0 else 0  # Avoid log(0)
#     KL_divergences.append(sum)

#     plt.plot(x_infer, mean_norm_infer, label=f'Inferred SIR (edges={num_edges})', alpha=0.6)
#     plt.fill_between(x_infer, mean_norm_infer - std_norm_infer, mean_norm_infer + std_norm_infer, alpha=0.2)

# plt.xlabel('Time')
# plt.ylabel('# of Infected')
# plt.title("Observed vs Inferred SIR Dynamics")
# plt.legend()
# plt.grid(True)
# plt.show()

# # Plot the KL divergences
# plt.figure(figsize=(10, 5))
# plt.plot(range(len(KL_divergences)), KL_divergences, marker='o', linestyle='-', color='orange')
# plt.xlabel('Time Step')
# plt.ylabel('KL Divergences')
# plt.title('KL Divergences for Varying Social Graph Edge Counts')
# plt.xticks(range(len(KL_divergences)), [f'Edges={nE_min + i * step}' for i in range(len(KL_divergences))], rotation=45)
# plt.tight_layout()
# plt.grid(True)
# plt.show()

# # Plot the peak uncertainties
# plt.figure(figsize=(10, 5))
# plt.plot(range(len(peak_uncertainties)), peak_uncertainties, marker='o', linestyle='-', color='green')
# plt.xlabel('Time Step')
# plt.ylabel('Peak Uncertainty')
# plt.title('Peak Uncertainties for Varying Social Graph Edge Counts')
# plt.xticks(range(len(peak_uncertainties)), [f'Edges={nE_min + i * step}' for i in range(len(peak_uncertainties))], rotation=45)
# plt.tight_layout()
# plt.grid(True)
# plt.show()

# # Dictionary to store all relevant variables
# variables = {
#     'n': n,
#     'T': T,
#     'Repeat': Repeat,
#     'beta': beta,
#     'gamma': gamma,
#     'mu': mu,
#     'init': init,
#     'contact_graph': contact_graph,
#     'num_trials': num_trials,
#     'T_runs': T_runs,
#     'Y_runs_normal': Y_runs_normal,
#     'x': x,
#     'y_norm': y_norm,
#     'mean_norm': mean_norm,
#     'std_norm': std_norm,
#     'nE_min': nE_min,
#     'nE_max': nE_max,
#     'step': step,
#     'KL_divergences': KL_divergences,
#     'peak_uncertainties': peak_uncertainties
# }

# # Save variables to a pickle file
# with open('experimenting_pickle.pkl', 'wb') as f:
#     pickle.dump(variables, f)

# print("All variables saved to sir_simulation_data.pkl")