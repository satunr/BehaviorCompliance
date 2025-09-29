import SIR
import networkx as nx
import correlated_graphs
from copy import deepcopy
import numpy as np
from scipy.optimize import minimize
import random
import os

#----------
#
#  Mean-field approximation for SIR model: n_i = beta * <k> (1 - n_r / gamma - r) * n_r / gamma
#  Assuming n_i, n_r, SIR parameters are known, this gives us the constrained, non-linear optimization problem:
#    y = w1 * x1 * (x2 - w2), where <w1, w2> are the model weights to be learned (mean node degree, recovered ratio, respectively).
#
#----------

social_graph = None
contact_graph = None

# Generate contact graph
n = 125  # number of nodes
p = 0.05  # probability of edge

clear = False  # Set clear to True if you want to use a new network or clear data files. False if you want to keep the existing one.
verbose = False  # Set verbose to True if you want to see detailed output during optimization

# Simulation parameters
T = 100
Repeat = 1
beta = 0.11
gamma = 0.10
mu = 0.10
init = 0.15 # Initial infected portion
q = "r"
adherence = 0.7  # Adherence = None -> Full adherence, otherwise it is a float between 0 and 1
split_point = 25  # Set to None if you want to optimize over the full SIR simulation, or a specific time point to split the optimization
seeds = None  # Set to None for random seeds, or a list of node IDs to use as seeds. Ex. [0, 1, 2] for nodes 0, 1, and 2 as seeds
density_social = None  # Set to None for default density, or an integer number of edges in the social graph

# Clear data files
def truncate_files():
    files_to_truncate = ["experiment_data/mfa_contact.gml", "experiment_data/mfa_social.gml", 
                         "experiment_data/mfa_xy_data.txt"]
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
    social_graph = correlated_graphs.create_social_graph(contact_graph, nE=density_social)[0]

    nx.write_gml(contact_graph, "experiment_data/mfa_contact.gml")
    nx.write_gml(social_graph, "experiment_data/mfa_social.gml")

SIR_results = SIR.Simulate_SIR(
    contact_network=deepcopy(contact_graph),
    social_network=deepcopy(social_graph),
    T=T, Repeat=Repeat,
    beta=beta, gamma=gamma, mu=mu, init=init,
    q=q, allow_restoration=q, 
    save_all=True, adherence=adherence, begin_q=split_point, 
    seeds=seeds
)

deg_lst = SIR_results[6]
true_dynamics = SIR_results[4]
informed_over_time = SIR_results[7]

def given_at_time(time):
    new_r_ratio = sum(1 for node in contact_graph.nodes() 
                            if true_dynamics[time][node] == 2 and true_dynamics[time-1][node] != 2) / n
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

total_time = T  # Save this value for writing daily infections, recoveries to file
# Mean-field approximation loses accuracy for small i
# -> Trim simulation; find where newly infected ratio < smallest_accurate_ratio
start = 2  # Ignore insufficient numbers at the start. Experimentally determined.
smallest_accurate_ratio = 0.1  # Minimum ratio of infected nodes to total nodes to consider the simulation accurate. Experimentally determined.
for t in range(start, T):
    i = sum(1 for node in contact_graph.nodes() if true_dynamics[t][node] == 1) / len(contact_graph.nodes())
    if i < smallest_accurate_ratio:
        T = t + 1
        break

# Save the full time-series data so we can average out daily infected/recovered results elsewhere
full_dynamics = deepcopy(true_dynamics)

# Shorten true_dynamics and deg_lst to match new T, where optimizer is well-behaved
true_dynamics = true_dynamics[:T]
deg_lst = deg_lst[:T]

# If split_point > T, indexxing issues will arise
assert split_point <= T, "Split point must be less than or equal to T"

if split_point is not None:
    seg1 = deg_lst[:split_point]
    mean1 = np.mean(seg1)
    seg2 = deg_lst[split_point:]
    mean2 = np.mean(seg2)
    #  Split mean_degrees into two halves with their own respective means
    mean_degrees = [mean1] * len(seg1) + [mean2] * len(seg2)
else:
    mean_degrees = [np.mean(deg_lst) for _ in range(T)]   # Mean degree of contact network over time
binomial_bound = n * p + np.sqrt(n * p * (1 - p))

# Compute true w1 and w2 values: Mean degree and fraction of recovered nodes
true_w = []
for time in range(T):
    true_w1 = mean_degrees[time]
    true_w2 = sum(1 for node in contact_graph.nodes() if true_dynamics[time][node] == 2) / n  # Recovered portion
    true_w.append((true_w1, true_w2))

# eps_w1: Controls exploration of w1
# alpha: Controls how much w1 is influenced by the run average
# eps_w2: Controls smoothness of w2
def optimize_segment(start=1, end=T, bounds = [(1, binomial_bound), (0, 1)],eps_w1=binomial_bound, eps_w2=0.075, alpha=0.7, num_runs=20):

    #---------
    #
    #  Optimization problem: y = w1 * x1 * (x2 - w2), where w1, w2 are unknown (mean node degree and fraction of recovered nodes, respectively)
    #
    #----------

    w1_avg = []
    w2_avg = []
    for _ in range(num_runs):
        w1_estimates = []
        w1_run_avg = None
        temp = eps_w1
        w2_estimates = []
        prev_w2 = None

        for t in range(start, end): 
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

    return w1_avg, w2_avg

# Split_point: None means no split, otherwise it is the time to split the optimization
#   This allows us to find missing parameters in the initial stage of SIR, where <k> is indep. from beta parameter,
#     and then optimize the rest of the simulation where <k> is dependent on beta.
def drive_optimizer(split_point=None):
    w1_all_runs = []
    w2_avg = []
    if split_point is None:
        w1_all_runs, w2_avg = optimize_segment()

        #  Run-wise average of w1 across runs
        w1_run_avg = np.mean(w1_all_runs, axis=0)
        w2_run_avg = np.mean(w2_avg, axis=0)

        # Return w1_avg to plot std band in extract_mfa.py
        return w1_run_avg, w2_run_avg, w1_all_runs

    w1_avg1, w2_avg1 = None, None
    w1_avg2, w2_avg2 = None, None
    if split_point is not None:
        w1_avg1, w2_avg1 = optimize_segment(start=1, end=split_point)
        w1_avg2, w2_avg2 = optimize_segment(start=split_point, end=T)

        #  Run-wise average of w1 across runs
        w1_avg1 = np.mean(w1_avg1, axis=0)
        w2_avg1 = np.mean(w2_avg1, axis=0)
        w1_avg2 = np.mean(w1_avg2, axis=0)
        w2_avg2 = np.mean(w2_avg2, axis=0)

        return w1_avg1, w2_avg1, w1_avg2, w2_avg2

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

# Save to mfa_xy_data.txt
# This is for single runs under a given SIR configuration
# Split: Tuple (split_point, half), where split_point is the time to split the optimization, and half is either 1 or 2
def save_xy_data(w1_avg=None, w2_avg=None, w1_all_runs=None, split=None):
    # Compute y values
    w1_true_y = [true_w[t][0] for t in range(T)]
    w1_est_y = w1_avg
    w2_true_y = [true_w[t][1] for t in range(T)]
    w2_est_y = w2_avg
    sir_infections_y = [sum(1 for node in contact_graph.nodes() if true_dynamics[t][node] == 1) for t in range(T)]

    # Bring informed_over_time into
    informed = deepcopy(informed_over_time)

    # Round to 2 decimal places where appropriate
    w1_est_y = [round(y, 2) for y in w1_est_y]
    w2_est_y = [round(y, 2) for y in w2_est_y]

    x_ofs = 0

    # If there is a split point, adjust x_vals and y values accordingly
    if split is not None:
        sp, half = split
        if half == 1:
            x_ofs = 0  # Offset for generating x values for plotting
            sir_infections_y = sir_infections_y[:sp]
            w1_true_y = w1_true_y[:sp]
            w1_est_y = w1_est_y[:sp]
            w2_true_y = w2_true_y[:sp]
            w2_est_y = w2_est_y[:sp]
            informed = informed[:sp]
        if half == 2:
            x_ofs = sp  # Offset for generating x values for plotting
            sir_infections_y = sir_infections_y[sp:]
            w1_true_y = w1_true_y[sp:-1]
            w1_est_y = w1_est_y[sp:-1]
            w2_true_y = w2_true_y[sp:-1]
            w2_est_y = w2_est_y[sp:-1]
            informed = informed[sp:-1]

    with open("experiment_data/mfa_xy_data.txt", "a") as f:
        f.write("==New Sample==\n")

        f.write("Number of nodes: " + str(n) + "\n\n")

        f.write("SIR Infections (Inset):\n")
        f.write(f"x: {','.join(map(str, range(x_ofs, len(sir_infections_y) + x_ofs)))}\n")
        f.write(f"y: {','.join(map(str, sir_infections_y))}\n\n")

        f.write("w1 True (Mean Node Degree):\n")
        f.write(f"x: {','.join(map(str, range(x_ofs, len(w1_true_y) + x_ofs)))}\n")
        f.write(f"y: {','.join(map(str, w1_true_y))}\n\n")

        f.write("w1 Estimated:\n")
        f.write(f"x: {','.join(map(str, range(x_ofs, len(w1_est_y) + x_ofs)))}\n")
        f.write(f"y: {','.join(map(str, w1_est_y))}\n\n")

        if w1_all_runs:
            f.write("w1 all runs:\n")
            f.write(f"x: {','.join(map(str, range(x_ofs, len(w1_all_runs[0]) + x_ofs)))}\n")
            f.write(f"y: {','.join(map(str, w1_all_runs))}\n\n")

        f.write("w2 True (Recovered Fraction):\n")
        f.write(f"x: {','.join(map(str, range(x_ofs, len(w2_true_y) + x_ofs)))}\n")
        f.write(f"y: {','.join(map(str, w2_true_y))}\n\n")

        f.write("w2 Estimated:\n")
        f.write(f"x: {','.join(map(str, range(x_ofs, len(w2_est_y) + x_ofs)))}\n")
        f.write(f"y: {','.join(map(str, w2_est_y))}\n\n")

        f.write("Informed and Infected Over Time:\n")
        # Keep x values aligned with sir_infections_y; recall we only run the optimizer up to T where i sufficiently large
        f.write(f"x: {','.join(map(str, range(x_ofs, len(sir_infections_y) + x_ofs)))}\n")
        f.write(f"y: {','.join(map(str, informed[:len(sir_infections_y)]))}\n\n")

if split_point is None:
    w1_run_avg, w2_run_avg, w1_all_runs = drive_optimizer(split_point=None)
    save_xy_data(w1_avg=w1_run_avg, w2_avg=w2_run_avg, w1_all_runs=w1_all_runs)
else:
    w1_avg1, w2_avg1, w1_avg2, w2_avg2 = drive_optimizer(split_point=split_point)
    save_xy_data(w1_avg=w1_avg1, w2_avg=w2_avg1, split=(split_point, 1))
    save_xy_data(w1_avg=w1_avg2, w2_avg=w2_avg2, split=(split_point, 2))

# Save daily infected and daily recovered counts to file
# Will be used in comparing ideal vs actual quarantine dynamics
def save_daily_infected_recovered():
    with open("experiment_data/infected_recovered.txt", "w") as f:
        f.write(f"Split point: {split_point}\n")
        f.write(f"Total nodes: {n}\n")
        f.write("Day,Newly Infected,Newly Recovered\n")
        
        # Start from day 1 because day 0 has no "previous day" to compare against
        for day in range(1, total_time):
            newly_infected = sum(
                1
                for node in contact_graph.nodes()
                if full_dynamics[day][node] == 1 and full_dynamics[day - 1][node] != 1
            )
            newly_recovered = sum(
                1
                for node in contact_graph.nodes()
                if full_dynamics[day][node] == 2 and full_dynamics[day - 1][node] != 2
            )
            
            f.write(f"{day},{newly_infected},{newly_recovered}\n")

save_daily_infected_recovered()