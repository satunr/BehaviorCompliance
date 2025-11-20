import networkx as nx
import SIR
import parse
import correlated_graphs
import matplotlib.pyplot as plt
from copy import deepcopy
import pickle
import numpy as np
import pandas as pd
import random
import find_seeds

n = 100
T = 100
Repeat = 1

beta = 0.15  #infection rate
gamma = 0.07  # recovery rate
mu = 0.10   # immunity loss
init = 0.15

contact_network = nx.erdos_renyi_graph(100, 0.05, seed=42)

# social_network = nx.erdos_renyi_graph(100, 0.05, seed=42)  # Placeholder for social network, replace with actual creation logic
social_network = correlated_graphs.create_social_graph(contact_network)[0]
social_network = social_network.to_directed()  # Ensure the social network is directed

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
    plt.plot(x, mean_inf, label='With quarantine', color='blue')
    plt.fill_between(x, mean_inf - std_inf, mean_inf + std_inf, color='blue', alpha=0.3)

    plt.plot(x, mean_noninf, label='Without quarantine', color='red')
    plt.fill_between(x, mean_noninf - std_noninf, mean_noninf + std_noninf, color='red', alpha=0.3)

    # Customize plot
    plt.xlabel('Time')
    plt.ylabel('# of Infected')
    plt.title('SIR with Permanent Quarantine')
    plt.legend()
    plt.grid(True)

    plt.show()

    return (x, mean_inf, std_inf), (x, mean_noninf, std_noninf)

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

    plt.plot(x, mean_q, label=f'With constant quarantine (q={quarantine_constant})', color='blue')
    plt.fill_between(x, mean_q - std_q, mean_q + std_q, color='blue', alpha=0.3)

    plt.plot(x, mean_noq, label='Without quarantine', color='red')
    plt.fill_between(x, mean_noq - std_noq, mean_noq + std_noq, color='red', alpha=0.3)

    plt.xlabel('Time')
    plt.ylabel('# of Infected')
    plt.title('SIR with Constant Quarantine')
    plt.legend()
    plt.grid(True)
    plt.show()

    return (x, mean_q, std_q), (x, mean_noq, std_noq)

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
    plt.title('SIR with Normally Distributed Quarantine')
    plt.legend()
    plt.grid(True)
    plt.show()

    return (x, mean_norm, std_norm), (x, mean_noq, std_noq)

# Function to plot Jaccard similarity between contact and social networks (2-hop creation) as X, and existence of edge (0 or 1) as Y
def plot_jaccard_similarity(num_runs=20, bins=10):
    # Load and preprocess graph
    H = nx.read_gml('experiment_data/Freeman3.gml')
    H = nx.convert_node_labels_to_integers(H, first_label=0)
    contact_graph = H.to_undirected()

    # Store bin_means for each run
    all_bin_means = []

    for run in range(num_runs):
        social_graph, sim = correlated_graphs.create_social_graph(contact_graph)
        I = social_graph.to_undirected()
        data = [(sim[pair], int(I.has_edge(*pair))) for pair in sim]
        df = pd.DataFrame(data, columns=['similarity', 'edge'])

        # Bin similarities and average edge existence
        df['bin'] = pd.cut(df['similarity'], bins=bins)
        bin_means = df.groupby('bin')['edge'].mean()
        all_bin_means.append(bin_means.values)

    # Convert to array shape: (num_runs, num_bins)
    all_bin_means = np.array(all_bin_means)

    # Mean and std across runs
    mean_bin_means = np.mean(all_bin_means, axis=0)
    std_bin_means = np.std(all_bin_means, axis=0)

    # Bin labels (same across runs)
    bin_labels = bin_means.index.astype(str)

    # Plot with std "brackets" (error bars)
    x = np.arange(len(bin_labels))
    plt.figure(figsize=(10, 6))
    plt.bar(x, mean_bin_means, yerr=std_bin_means, capsize=5,
            color='skyblue', edgecolor='black', alpha=0.8)

    plt.ylabel("Probability of edge existence")
    plt.xlabel("Jaccard similarity bin")
    plt.title("Social graph creation algorithm")
    plt.xticks(x, bin_labels, rotation=45)
    plt.tight_layout()
    plt.show()

    return df, bin_means 

# num_comparisons is the number of random vs non-random seed sets to compare
def random_vs_nonrandom_seeds(num_comparisons):
    repeat = 10
    infections_random = []
    infections_nonrandom = []

    # List nodes by degree:
    degree_dict = dict(contact_network.degree())
    sorted_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)

    print("Nodes sorted by degree: ", sorted_nodes)

    random_seeds = [random.choice(list(contact_network.nodes())) for _ in range(num_comparisons)]
    non_random_seeds = find_seeds.find_seed_set(deepcopy(contact_network), num_seeds=num_comparisons, exponent=20)

    for i in range(1, num_comparisons+1):
        print("Run #: ", i)
        
        inner_random_avg = []
        for _ in range(repeat):
            data_random = SIR.Simulate_SIR(
                contact_network=deepcopy(contact_network),
                social_network=deepcopy(social_network),
                T=T, Repeat=Repeat,
                beta=beta, gamma=gamma, mu=mu, init=init, q="r",
                allow_restoration=True,
                seeds=random_seeds[:i]
            )[2]
            inner_random_avg.append(np.mean(data_random[1]))
        infections_random.append(inner_random_avg)

        inner_nonrandom_avg = []
        for _ in range(repeat):
            data_nonrandom = SIR.Simulate_SIR(
                contact_network=deepcopy(contact_network),
                social_network=deepcopy(social_network),
                T=T, Repeat=Repeat,
                beta=beta, gamma=gamma, mu=mu, init=init, q="r",
                allow_restoration=True,
                seeds=non_random_seeds[:i]
            )[2]
            inner_nonrandom_avg.append(np.mean(data_nonrandom[1]))
        infections_nonrandom.append(inner_nonrandom_avg)

    # Compute means and stds
    avg_infections_random = [np.mean(run) for run in infections_random]
    std_infections_random = [np.std(run) for run in infections_random]

    avg_infections_nonrandom = [np.mean(run) for run in infections_nonrandom]
    std_infections_nonrandom = [np.std(run) for run in infections_nonrandom]

    # Plotting as bar chart with error bars
    x = np.arange(1, num_comparisons+1)
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, avg_infections_random, width,
            yerr=std_infections_random, capsize=5, label="Random Seeds", color="orange", alpha=0.7)
    plt.bar(x + width/2, avg_infections_nonrandom, width,
            yerr=std_infections_nonrandom, capsize=5, label="Non-Random Seeds", color="blue", alpha=0.7)
    plt.ylim(25, max(max(avg_infections_random), max(avg_infections_nonrandom)) + 5)
    plt.xlabel("Comparison Index (# of Seeds)")
    plt.ylabel("Avg # of Infected")
    plt.title("Random vs Non-Random Seed Selection")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

# Pickle results from the functions
def SIR_pickle_dump(filename='experiment_data/pickles.pkl'):
    # We will pickle these parameters along with the results for later reference
    presets = {'T': T, 'Repeat': Repeat, 'beta': beta, 'gamma': gamma, 'mu': mu, 'init': init}

    informed_vs_noninformed()
    const_quarantines()
    normal_dist_quarantines()
    plot_jaccard_similarity()

    with open(filename, 'wb') as f:
        # Clear the file before writing
        f.truncate(0)

        pickle.dump({'SIR presets': presets}, f)
        pickle.dump({'Informed vs Non-Informed': informed_vs_noninformed()}, f)
        pickle.dump({'Constant Quarantines': const_quarantines()}, f)
        pickle.dump({'Normal Dist. Quarantines': normal_dist_quarantines()}, f)
    print("Data has been pickled successfully.")

# SIR_pickle_dump()

def pickle_load(filename='experiment_data/pickles.pkl'):
    # Open the file in binary read mode
    with open(filename, 'rb') as file:
        data = pickle.load(file)

    # Now `data` holds the deserialized object
    print(data)

def compare_infections_adherence(adherence):
    num_trials = 10

    T_runs = []
    Y_runs_full_adherence = []
    Y_runs_partial_adherence = []

    for i in range(num_trials):
        print("Run #: ", i)

        # With constant quarantine
        data1 = SIR.Simulate_SIR(
            contact_network=deepcopy(contact_network),
            social_network=deepcopy(social_network),
            T=T, Repeat=Repeat,
            beta=beta, gamma=gamma, mu=mu, init=init, q="r",
            allow_restoration=True
        )[2]
        T_runs.append(data1[0])
        Y_runs_full_adherence.append(data1[1])

        # Without quarantine
        data2 = SIR.Simulate_SIR(
            contact_network=deepcopy(contact_network),
            social_network=deepcopy(social_network),
            T=T, Repeat=Repeat,
            beta=beta, gamma=gamma, mu=mu, init=init,q="r",
            allow_restoration=True, adherence=adherence
        )[2]
        Y_runs_partial_adherence.append(data2[1])

    x = T_runs[0]
    y_full = np.array(Y_runs_full_adherence)
    y_partial = np.array(Y_runs_partial_adherence)

    mean_full = np.mean(y_full, axis=0)
    std_full = np.std(y_full, axis=0)

    mean_partial = np.mean(y_partial, axis=0)
    std_partial = np.std(y_partial, axis=0)

    plt.plot(x, mean_full, label=f'With full adherence', color='blue')
    plt.fill_between(x, mean_full - std_full, mean_full + std_full, color='blue', alpha=0.3)

    plt.plot(x, mean_partial, label='With partial adherence', color='red')
    plt.fill_between(x, mean_partial - std_partial, mean_partial + std_partial, color='red', alpha=0.3)

    plt.xlabel('Time')
    plt.ylabel('# of Infected')
    plt.title('SIR with Constant Quarantine')
    plt.legend()
    plt.grid(True)
    plt.show()

    return (x, mean_full, std_full), (x, mean_partial, std_partial)

# SIRS w/ param q = "r" (edges removed until recovery) and adherence parameter
def r_quarantine():
    num_trials = 10

    T_runs = []
    Y_runs_r_quarantine = []
    Y_runs_noquarantine = []

    for _ in range(num_trials):
        # With r-quarantine
        data1 = SIR.Simulate_SIR(
            contact_network=deepcopy(contact_network),
            social_network=deepcopy(social_network),
            T=T, Repeat=Repeat,
            beta=beta, gamma=gamma, mu=mu, init=init,
            average_data=False, q="r",
            allow_restoration=True
        )[2]
        T_runs.append(data1[0])
        Y_runs_r_quarantine.append(data1[1])

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
    y_rq = np.array(Y_runs_r_quarantine)
    y_noq = np.array(Y_runs_noquarantine)

    mean_rq = np.mean(y_rq, axis=0)
    std_rq = np.std(y_rq, axis=0)

    mean_noq = np.mean(y_noq, axis=0)
    std_noq = np.std(y_noq, axis=0)

    plt.plot(x, mean_rq, label='With r-quarantine', color='blue')
    plt.fill_between(x, mean_rq - std_rq, mean_rq + std_rq, color='blue', alpha=0.3)

    plt.plot(x, mean_noq, label='Without quarantine', color='red')
    plt.fill_between(x, mean_noq - std_noq, mean_noq + std_noq, color='red', alpha=0.3)

    plt.xlabel('Time')
    plt.ylabel('# of Infected')
    plt.title('SIR with r-Quarantine')
    plt.legend()
    plt.grid(True)
    plt.show()

    return (x, mean_rq, std_rq), (x, mean_noq, std_noq)

# compare_infections_adherence(0.75)

# plot_jaccard_similarity()

random_vs_nonrandom_seeds(4)