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

social_graph = None
contact_graph = None

# Generate contact and social graphs
n = 150
p = 0.05

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

# Simulation parameters
T = 150
Repeat = 1
beta = 0.11
gamma = 0.10
mu = 0.10
init = 0.10
q = True

SIR_results = SIR.Simulate_SIR(
    contact_network=deepcopy(contact_graph),
    social_network=deepcopy(social_graph),
    T=T, Repeat=Repeat,
    beta=beta, gamma=gamma, mu=mu, init=init,
    average_data=False, q=q, allow_restoration=q, save_all=True
)

deg_lst = SIR_results[6]
true_dynamics = SIR_results[4]

# Mean-field approximation loses accuracy for small i
# -> Trim simulation; find where newly infected ratio < 0.1
start = 2  # Start at time not affected by smallest_accurate_ratio
smallest_accurate_ratio = 0.1
for t in range(start, T):
    if sum(1 for node in contact_graph.nodes() if true_dynamics[t][node] == 1) / len(contact_graph.nodes()) < smallest_accurate_ratio:
        T = t + 1
        break

# Shorten true_dynamics and deg_lst to match T
true_dynamics = true_dynamics[:T]
deg_lst = deg_lst[:T]

mean_degrees = np.mean(deg_lst)   # Mean degree of contact network over time
num_nodes = len(contact_graph.nodes())
binomial_bound = n * p + np.sqrt(n * p * (1 - p))

if num_nodes == 0:
    raise ValueError("The contact graph has no nodes. Please check the graph generation parameters.")

# Compute true w1 and w2 values: Mean degree and fraction of recovered nodes
true_w = []
for time in range(T):
    true_w1 = mean_degrees
    true_w2 = sum(1 for node in contact_graph.nodes() if true_dynamics[time][node] == 2) / num_nodes  # Recovered portion
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

#---------
#
#  Optimization problem: y = w1 * x1 * (x2 - w2), where w1, w2 are unknown (mean node degree and fraction of recovered nodes, respectively)
#
#----------

eps_w1 = binomial_bound  # Controls exploration of w1
# eps_w1 = 3
alpha = 0.6  # Controls how much w1 is influenced by the run average
eps_w2 = 0.075  # Controls smoothness of w2
bounds = [(1, binomial_bound), (0, 1)]
num_runs = 50
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

#---------
#
#  Plotting results for w1: Mean Node Degree
#
#---------

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(range(1, T), [true_w[t][0] for t in range(1, T)], label='True M.N.D.', marker='o')
ax.plot(range(1, T), w1_avg, label='Estimated M.N.D.', marker='x')
ax.set_xticks(range(T))
ax.set_xlabel('Time')
ax.set_ylabel('mean node degree values')
ax.set_title('True vs Estimated Mean Node Degree (M.N.D.) over Time')
ax.legend()
ax.grid(True)

# # Inset: SIR infections
sir_infections = [sum(1 for node in contact_graph.nodes() if true_dynamics[t][node] == 1) for t in range(T)]
ax_inset = inset_axes(ax, width="10%", height="10%", loc='lower right')
ax_inset.plot(range(T), sir_infections, color='gray', linestyle='--')
ax_inset.set_title("SIR Infections", fontsize=8)
ax_inset.tick_params(axis='both', which='major', labelsize=6)
ax_inset.grid(True)

plt.tight_layout()
plt.show()

#----------
#
#  Plotting results for w2: Fraction of Recovered Nodes
#
#----------

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(range(1, T), [true_w[t][1] for t in range(1, T)], label='True r', marker='o')
ax.plot(range(1, T), w2_avg, label='Estimated r', marker='x')
ax.set_xticks(range(T))
ax.set_xlabel('Time')
ax.set_ylabel('r values')
ax.set_title('True vs Estimated Recovered Fraction (r) over Time')
ax.legend()
ax.grid(True)

# # Inset: SIR infections
sir_infections = [sum(1 for node in contact_graph.nodes() if true_dynamics[t][node] == 1) for t in range(T)]
ax_inset = inset_axes(ax, width="10%", height="10%", loc='lower right')
ax_inset.plot(range(T), sir_infections, color='gray', linestyle='--')
ax_inset.set_title("SIR Infections", fontsize=8)
ax_inset.tick_params(axis='both', which='major', labelsize=6)
ax_inset.grid(True)

plt.tight_layout()
plt.show()

# Write avg_w1 and avg_w2 to a file
# Make avg_w2 a dictionary with time as keys
avg_w2_dict = {t: w2_avg[t-1] for t in range
               (1, T)}

# round w2_avg values to 2 decimal places
w2_avg = [round(value, 2) for value in w2_avg]

with open("experiment_data/mfa_results.txt", "a") as f:
    f.write("avg_w1:\n")
    f.write(f"{round(np.mean(w1_avg), 2)}")
    f.write("\n \n")
    f.write("avg_w2:\n")
    f.write("\n".join(f"{t}: {w2_avg[t-1]}" for t in range(1, T)) + "\n")

# --- Optional: Surface and gradient plot ---

def f(w1, w2, T_gen):
    x1, x2 = given_at_time(T_gen)
    return (w1 * x1) * (x2 - w2)

def plot_surface(T_gen):
    w1_range = np.linspace(1, num_nodes, 100)
    w2_range = np.linspace(0, 1, 100)
    W1, W2 = np.meshgrid(w1_range, w2_range)
    Z = f(W1, W2, T_gen)

    fig_surface = plt.figure(figsize=(10, 8))
    ax_surface = fig_surface.add_subplot(111, projection='3d')
    ax_surface.plot_surface(W1, W2, Z, cmap='viridis', alpha=0.8)
    ax_surface.set_xlabel('w1')
    ax_surface.set_ylabel('w2')
    ax_surface.set_zlabel('Function Value')
    ax_surface.set_title(f'Surface Plot of f(w1, w2) at T={T_gen}')
    plt.tight_layout()
    plt.show()

def plot_gradient(T_gen):
    w1_range = np.linspace(1, num_nodes, 100)
    w2_range = np.linspace(0, 1, 100)
    W1, W2 = np.meshgrid(w1_range, w2_range)
    Z = f(W1, W2, T_gen)
    dZ_dw1, dZ_dw2 = np.gradient(Z, w1_range, w2_range)

    fig_gradient = plt.figure(figsize=(10, 8))
    ax_gradient = fig_gradient.add_subplot(111)
    ax_gradient.quiver(W1, W2, dZ_dw1, dZ_dw2, color='blue', alpha=0.5)
    ax_gradient.set_xlabel('w1')
    ax_gradient.set_ylabel('w2')
    ax_gradient.set_title(f'Gradient Field of f(w1, w2) at T={T_gen}')
    plt.tight_layout()
    plt.show()
