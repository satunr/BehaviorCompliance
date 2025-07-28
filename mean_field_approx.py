import sys
import SIR
import networkx as nx
import correlated_graphs
from copy import deepcopy
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import random
import os
import pickle

social_graph = None
contact_graph = None

# Generate contact and social graphs
n = 125
p = 0.05

clear = False  # Set clear to True if you want to use a new network or clear data files. False if you want to keep the existing one.
verbose = False  # Set verbose to True if you want to see detailed output during optimization

# Simulation parameters
T = 125
Repeat = 1
beta = 0.09
gamma = 0.10
mu = 0.10
init = 0.10
q = True  
adherence = 0.0

# Clear data files
def truncate_files():
    files_to_truncate = ["experiment_data/mfa_results.txt", "experiment_data/mfa_contact.gml", 
                         "experiment_data/mfa_social.gml", "experiment_data/mfa_xy_data.pkl", "experiment_data/mfa_avgs.pkl"]
    for file in files_to_truncate:
        if os.path.exists(file):
            with open(file, 'w') as f:
                f.truncate(0)
if clear:
    truncate_files()

try:
    if not os.path.exists("experiment_data/mfa_contact.gml") or not os.path.exists("experiment_data/mfa_social.gml"):
        raise FileNotFoundError

    contact_graph = nx.read_gml("experiment_data/mfa_contact.gml")
    social_graph = nx.read_gml("experiment_data/mfa_social.gml")

    # Convert node labels from strings to integers
    contact_graph = nx.relabel_nodes(contact_graph, lambda x: int(x))
    social_graph = nx.relabel_nodes(social_graph, lambda x: int(x))

except (FileNotFoundError, OSError, ValueError, nx.NetworkXError) as e:
    print("No valid graphs found in mfa_*.gml. Generating new graphs.")

    contact_graph = nx.erdos_renyi_graph(n, p, seed=42)
    social_graph = correlated_graphs.create_social_graph(contact_graph, nE=2 * len(contact_graph.edges()))[0]

    nx.write_gml(contact_graph, "experiment_data/mfa_contact.gml")
    nx.write_gml(social_graph, "experiment_data/mfa_social.gml")

SIR_results = SIR.Simulate_SIR(
    contact_network=deepcopy(contact_graph),
    social_network=deepcopy(social_graph),
    T=T, Repeat=Repeat,
    beta=beta, gamma=gamma, mu=mu, init=init,
    average_data=False, q=q, allow_restoration=q, save_all=True, adherence=adherence
)

deg_lst = SIR_results[6]
true_dynamics = SIR_results[4]

def given_at_time(time):
    new_r_ratio = sum(1 for node in contact_graph.nodes() 
                            if true_dynamics[time][node] == 2 and true_dynamics[time-1][node] < 2) / n
    new_i_ratio = sum(1 for node in contact_graph.nodes() 
                            if true_dynamics[time][node] == 1 and true_dynamics[time-1][node] != 1) / n
    x1 = beta * (new_r_ratio / gamma)
    x2 = 1 - (new_r_ratio / gamma)

    return (x1, x2, new_i_ratio)

def y_true(time):
    x1, x2, _ = given_at_time(time)
    true_w1, true_w2 = true_w[time]

    return (true_w1 * x1) * (x2 - true_w2)

def loss(params, w1_run_avg, prev_w2, T_gen, eps_w1, eps_w2):
    w1, w2 = params
    x1, x2, _ = given_at_time(T_gen)
    y_pred = (w1 * x1) * (x2 - w2)

    penalties = 0
    # Model constraints
    if w1 > binomial_bound or w1 < 0 or w2 > 1 or w2 < 0:
        penalties += 1000
        
    # Convergence constraints
    if w1_run_avg is not None:
        delta = abs(w1 - w1_run_avg)
        if delta > eps_w1:
            penalties += 100 * (delta - eps_w1)**2

    # Smoothness constraints
    if prev_w2 is not None:
        delta = abs(w2 - prev_w2)
        if delta > eps_w2:
            penalties += 100 * (delta - eps_w2)**2

    return ((y_pred - y_true(T_gen)) ** 2) + penalties

# Mean-field approximation loses accuracy for small i
# -> Trim simulation; find where newly infected ratio < smallest_accurate_ratio
start = 2  # Ignore insufficient numbers at the start. Experimentally determined.
smallest_accurate_ratio = 0.1  # Minimum ratio of infected nodes to total nodes to consider the simulation accurate. Experimentally determined.
for t in range(start, T):
    i = sum(1 for node in contact_graph.nodes() if true_dynamics[t][node] == 1) / len(contact_graph.nodes())
    if i < smallest_accurate_ratio:
        T = t + 1
        break

# Shorten true_dynamics and deg_lst to match new T
true_dynamics = true_dynamics[:T]
deg_lst = deg_lst[:T]

mean_degrees = np.mean(deg_lst)   # Mean degree of contact network over time
binomial_bound = n * p + np.sqrt(n * p * (1 - p))

# Compute true w1 and w2 values: Mean degree and fraction of recovered nodes
true_w = []
for time in range(T):
    true_w1 = mean_degrees
    true_w2 = sum(1 for node in contact_graph.nodes() if true_dynamics[time][node] == 2) / n  # Recovered portion
    true_w.append((true_w1, true_w2))


#---------
#
#  Optimization problem: y = w1 * x1 * (x2 - w2), where w1, w2 are unknown (mean node degree and fraction of recovered nodes, respectively)
#
#----------

eps_w1 = binomial_bound  # Controls exploration of w1
alpha = 0.7  # Controls how much w1 is influenced by the run average
eps_w2 = 0.075  # Controls smoothness of w2
bounds = [(1, binomial_bound), (0, 1)]
num_runs = 3
w1_avg = []
w2_avg = []
for _ in range(num_runs):
    w1_estimates = []
    w1_run_avg = None
    temp = eps_w1
    w2_estimates = []
    prev_w2 = None

    for t in range(1, T): 
        T_gen = t
        init_guess = None
        eps_w1 = temp * ((1 - t/T)**0.5)  # Polynomial root decay of eps_w1 over time

        if w1_run_avg == None or prev_w2 == None:
            init_guess = [random.uniform(1, binomial_bound), true_w[T_gen][1]]
        else:
            #  Scipy will handle bounds issues here if they occur
            init_guess = [w1_run_avg + random.uniform(-eps_w1, eps_w1), prev_w2 + random.uniform(-eps_w2, eps_w2)]

        result = None
        result = minimize(lambda params: loss(params, w1_run_avg, prev_w2, T_gen, eps_w1=eps_w1, eps_w2=eps_w2), init_guess, method='L-BFGS-B', bounds=bounds)

        w1 = result.x[0]
        w1 = alpha * w1 + (1 - alpha) * w1_run_avg if w1_run_avg is not None else w1  # Apply run average smoothing
        w1_estimates.append(w1)
        w1_run_avg = np.mean(w1_estimates)  # Update run average for w1
        w2 = result.x[1]
        w2_estimates.append(w2)

        if verbose:
            print("Time:", t)
            print("Initial guess:", init_guess)
            print("True w1:", true_w[T_gen][0])
            print("True w2:", true_w[T_gen][1])
            print("Estimated w1:", w1)
            print("Estimated w2:", result.x[1])
            print("x values at T_gen:", given_at_time(T_gen))
            print("Squared error:", result.fun)
            print("Mean node degree:", deg_lst[T_gen])
            print("\n")

        prev_w2 = w2 # Update prev_w2 for the next iteration

    w1_avg.append(w1_estimates)  # List of w1 estimates for this run
    w2_avg.append(w2_estimates)  # List of w2 estimates for this run

    eps_w1 = temp  # Reset eps_w1 for the next run

#  Run-wise average of w1 across runs
w1_avg = np.mean(w1_avg, axis=0)
w2_avg = np.mean(w2_avg, axis=0)

#-------------
#
#  Write data for plotting to file
#     Form: <label>:\n
#           x: <x_values>\n
#           y: <y_values>\n
#     where x_values and y_values are comma-separated lists of values
#         for the following data: SIR simulation, w1 true, w1 estimated, w2 true, and w2 estimated
#
#-------------

# Generate all required data
x_vals = list(range(1, T))  # Common x for most data
x_vals_full = list(range(T))  # For SIR inset which uses full range

# Compute y values
w1_true_y = [true_w[t][0] for t in x_vals]
w1_est_y = w1_avg
w2_true_y = [true_w[t][1] for t in x_vals]
w2_est_y = w2_avg
sir_infections_y = [sum(1 for node in contact_graph.nodes() if true_dynamics[t][node] == 1) for t in x_vals_full]

# Round to 2 decimal places where appropriate
w1_est_y = [round(y, 2) for y in w1_est_y]
w2_est_y = [round(y, 2) for y in w2_est_y]

# Save to mfa_xy_data.txt
# This is for single runs under a given SIR configuration
def save_xy_data():
    with open("experiment_data/mfa_xy_data.txt", "a") as f:
        f.write("==New Sample==\n")
        f.write("SIR Infections (Inset):\n")
        f.write(f"x: {','.join(map(str, x_vals_full))}\n")
        f.write(f"y: {','.join(map(str, sir_infections_y))}\n\n")

        f.write("w1 True (Mean Node Degree):\n")
        f.write(f"x: {','.join(map(str, x_vals))}\n")
        f.write(f"y: {','.join(map(str, w1_true_y))}\n\n")

        f.write("w1 Estimated:\n")
        f.write(f"x: {','.join(map(str, x_vals))}\n")
        f.write(f"y: {','.join(map(str, w1_est_y))}\n\n")

        f.write("w2 True (Recovered Fraction):\n")
        f.write(f"x: {','.join(map(str, x_vals))}\n")
        f.write(f"y: {','.join(map(str, w2_true_y))}\n\n")

        f.write("w2 Estimated:\n")
        f.write(f"x: {','.join(map(str, x_vals))}\n")
        f.write(f"y: {','.join(map(str, w2_est_y))}\n\n")

save_xy_data()

# This is for averaging multiple runs under a given SIR configuration
def save_results():
    results = {
        "w1_avg": w1_avg,
        "w2_avg": w2_avg,
    }
    # mfa_compute.pkl will be used for computing averages for a given SIR config
    # mfa_xy_data.pkl will be used to hold these averages for plotting
    with open("experiment_data/mfa_compute.pkl", "ab") as f:
        pickle.dump(results, f)

# If running as a subprocess, i.e. we are averaging runs, save the results to mfa_compute.pkl
save_results()