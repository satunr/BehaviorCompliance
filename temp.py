import SIR
import networkx as nx
import correlated_graphs
from copy import deepcopy
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import random

# ------------------
# Graph Parameters
# ------------------
n = 150
p = 0.05
contact_graph = nx.erdos_renyi_graph(n, p, seed=42)
social_graph = correlated_graphs.create_social_graph(contact_graph, nE=2 * len(contact_graph.edges()))[0]

# ------------------
# SIR Parameters
# ------------------
T = 150
Repeat = 1
beta = 0.11
gamma = 0.10
mu = 0.10
init = 0.10

# ------------------
# SIR Simulations
# ------------------
SIR_results = SIR.Simulate_SIR(
    contact_network=deepcopy(contact_graph),
    social_network=deepcopy(social_graph),
    T=T, Repeat=Repeat,
    beta=beta, gamma=gamma, mu=mu, init=init,
    average_data=False, q=True, allow_restoration=True, save_all=True
)

SIR_results_noq = SIR.Simulate_SIR(
    contact_network=deepcopy(contact_graph),
    social_network=deepcopy(social_graph),
    T=T, Repeat=Repeat,
    beta=beta, gamma=gamma, mu=mu, init=init,
    average_data=False, q=False, allow_restoration=False, save_all=True
)

# ------------------
# Optimization Routine
# ------------------
def estimate_w_values(SIR_results, contact_graph, T_initial, eps_w1, alpha, eps_w2, num_runs):
    deg_lst = SIR_results[6]
    true_dynamics = SIR_results[4]

    T = T_initial
    smallest_accurate_ratio = 0.1
    for t in range(2, T):
        if sum(1 for node in contact_graph.nodes() if true_dynamics[t][node] == 1) / len(contact_graph.nodes()) < smallest_accurate_ratio:
            T = t + 1
            break

    true_dynamics = true_dynamics[:T]
    deg_lst = deg_lst[:T]
    num_nodes = len(contact_graph.nodes())
    mean_degrees = np.mean(deg_lst)
    binomial_bound = num_nodes * p + np.sqrt(num_nodes * p * (1 - p))

    true_w = []
    for time in range(T):
        true_w1 = mean_degrees
        true_w2 = sum(1 for node in contact_graph.nodes() if true_dynamics[time][node] == 2) / num_nodes
        true_w.append((true_w1, true_w2))

    def given_at_time(time):
        new_r_ratio = sum(1 for node in contact_graph.nodes()
                            if true_dynamics[time][node] == 2 and true_dynamics[time-1][node] < 2) / num_nodes
        x1 = beta * (new_r_ratio / gamma)
        x2 = 1 - (new_r_ratio / gamma)
        return (x1, x2)

    def y_true(time):
        x1, x2 = given_at_time(time)
        true_w1, true_w2 = true_w[time]
        return (true_w1 * x1) * (x2 - true_w2)

    def loss(params, w1_run_avg, prev_w2, T_gen, eps_w1, eps_w2):
        w1, w2 = params
        x1, x2 = given_at_time(T_gen)
        y_pred = (w1 * x1) * (x2 - w2)
        penalties = 0
        if w1 > binomial_bound or w1 < 0 or w2 > 1 or w2 < 0:
            penalties += 1000
        if w1_run_avg is not None:
            delta = abs(w1 - w1_run_avg)
            if delta > eps_w1:
                penalties += 100 * (delta - eps_w1)**2
        if prev_w2 is not None:
            delta = abs(w2 - prev_w2)
            if delta > eps_w2:
                penalties += 100 * (delta - eps_w2)**2
        return ((y_pred - y_true(T_gen)) ** 2) + penalties

    w1_all_runs = []
    w2_all_runs = []
    for _ in range(num_runs):
        w1_estimates, w2_estimates = [], []
        w1_run_avg, prev_w2 = None, None
        temp_eps = eps_w1
        for t in range(1, T):
            T_gen = t
            eps_w1 = temp_eps * ((1 - t/T)**0.5)
            if w1_run_avg is None or prev_w2 is None:
                init_guess = [random.uniform(1, binomial_bound), true_w[T_gen][1]]
            else:
                init_guess = [w1_run_avg + random.uniform(-eps_w1, eps_w1),
                              prev_w2 + random.uniform(-eps_w2, eps_w2)]
            result = minimize(lambda params: loss(params, w1_run_avg, prev_w2, T_gen, eps_w1, eps_w2),
                              init_guess, method='L-BFGS-B', bounds=[(1, binomial_bound), (0, 1)])
            w1 = result.x[0]
            w1 = alpha * w1 + (1 - alpha) * w1_run_avg if w1_run_avg is not None else w1
            w1_estimates.append(w1)
            w1_run_avg = np.mean(w1_estimates)
            w2 = result.x[1]
            w2_estimates.append(w2)
            prev_w2 = w2
        w1_all_runs.append(w1_estimates)
        w2_all_runs.append(w2_estimates)
    return T, true_w, np.mean(w1_all_runs, axis=0), np.mean(w2_all_runs, axis=0)

# ------------------
# Run Estimation on Both Sets
# ------------------
eps_w1 = n * p + np.sqrt(n * p * (1 - p))
alpha = 0.6
eps_w2 = 0.075
num_runs = 50

T_q, true_w_q, w1_q, w2_q = estimate_w_values(SIR_results, contact_graph, T, eps_w1, alpha, eps_w2, num_runs)
T_noq, true_w_noq, w1_noq, w2_noq = estimate_w_values(SIR_results_noq, contact_graph, T, eps_w1, alpha, eps_w2, num_runs)

# ------------------
# Plotting Results
# ------------------
def plot_results(T, true_w, w1_est, w2_est, title_suffix):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(1, T), [true_w[t][0] for t in range(1, T)], label='True M.N.D.', marker='o')
    ax.plot(range(1, T), w1_est, label='Estimated M.N.D.', marker='x')
    ax.set_xlabel('Time')
    ax.set_ylabel('Mean Node Degree')
    ax.set_title(f'True vs Estimated Mean Node Degree {title_suffix}')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(1, T), [true_w[t][1] for t in range(1, T)], label='True r', marker='o')
    ax.plot(range(1, T), w2_est, label='Estimated r', marker='x')
    ax.set_xlabel('Time')
    ax.set_ylabel('Fraction Recovered')
    ax.set_title(f'True vs Estimated Recovered Fraction {title_suffix}')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

plot_results(T_q, true_w_q, w1_q, w2_q, '(With Quarantine)')
plot_results(T_noq, true_w_noq, w1_noq, w2_noq, '(No Quarantine)')
