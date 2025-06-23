import SIR
import networkx as nx
import correlated_graphs
from copy import deepcopy
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import random
from scipy.optimize import differential_evolution

# Generate contact and social graphs
n = 100
p = 0.05
contact_graph = nx.erdos_renyi_graph(n, p, seed=42)
social_graph = correlated_graphs.create_social_graph(contact_graph, nE=2 * len(contact_graph.edges()))[0]

# Simulation parameters
T = 50
Repeat = 1
beta = 0.15
gamma = 0.10
mu = 0.10
init = 0.10

# Run SIR simulation
SIR_results = SIR.Simulate_SIR(
    contact_network=deepcopy(contact_graph),
    social_network=deepcopy(social_graph),
    T=T, Repeat=Repeat,
    beta=beta, gamma=gamma, mu=mu, init=init,
    average_data=False, q=False, allow_restoration=False, save_all=True
)

mean_degrees = np.mean(SIR_results[6])   # Mean degree of contact network over time
true_dynamics = SIR_results[4]
num_nodes = len(contact_graph.nodes())

# Compute true w1 and w2 values
true_w = []
for time in range(T):
    true_w1 = mean_degrees
    true_w2 = sum(1 for node in contact_graph.nodes() if true_dynamics[time][node] == 2) / num_nodes
    true_w.append((true_w1, true_w2))

def given_at_time(time):
    num_new_recovered = sum(1 for node in contact_graph.nodes() 
                            if true_dynamics[time][node] == 2 and true_dynamics[time-1][node] < 2)
    x1 = beta * (num_new_recovered / gamma)
    x2 = 1 - (num_new_recovered / gamma)
    return (x1, x2)

# True output function
def y_true(time):
    x1, x2 = given_at_time(time)
    true_w1, true_w2 = true_w[time]

    return (true_w1 * x1) * (x2 - true_w2)

# Loss function takes params and previous w2
def loss(params, prev_w2, T_gen):
    w1, w2 = params
    
    x1, x2 = given_at_time(T_gen)
    y_pred = (w1 * x1) * (x2 - w2)

    penalties = 0
    if w1 > num_nodes or w1 < 0 or w2 > 1 or w2 < 0:
        penalties += 1000
        
    if prev_w2 is not None:
        delta = abs(w2 - prev_w2)
        if delta > eps:
            penalties += 100 * (delta - eps)**2

    return ((y_pred - y_true(T_gen)) ** 2) + penalties

# Optimization
eps = 0.1
binomial_bound = n * p + np.sqrt(n * p * (1 - p))
bounds = [(1, binomial_bound), (0, 1)]
Repeat = 5
w1_avg = []
w2_avg = []
for _ in range(Repeat):
    w1_estimates = []
    w2_estimates = []
    prev_w2 = None

    for t in range(1, T): 
        T_gen = t
        # Initialize with true initial conditions
        init_guess = [random.uniform(1, binomial_bound), min(max(prev_w2 + random.uniform(-eps, eps), 0), 1)] if prev_w2 is not None else [true_w[T_gen][0], true_w[T_gen][1]]

        # Use lambda to pass fixed T_gen and prev_w2 to loss
        # result = differential_evolution(lambda params: loss(params, prev_w2, T_gen), bounds=bounds, maxiter=1000, popsize=20)

        result = None
        result = minimize(lambda params: loss(params, prev_w2, T_gen), init_guess, method='L-BFGS-B', bounds=bounds)

        w1_estimates.append(result.x[0])
        w1 = result.x[0]
        w2 = result.x[1]
        w2_estimates.append(result.x[1])

        # Try with differential evolution

        print("Time:", t)
        print("Initial guess:", init_guess)
        print("True w1:", true_w[T_gen][0])
        print("True w2:", true_w[T_gen][1])
        print("Estimated w1:", result.x[0])
        print("Estimated w2:", result.x[1])
        print("x values at T_gen:", given_at_time(T_gen))
        print("Squared error:", result.fun)
        print("\n")

        prev_w2 = result.x[1] # Update prev_w2 for the next iteration

    w1_avg.append(w1_estimates)
    w2_avg.append(w2_estimates)

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
ax_inset = inset_axes(ax, width="30%", height="30%", loc='upper left')
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
ax_inset = inset_axes(ax, width="30%", height="30%", loc='upper left')
ax_inset.plot(range(T), sir_infections, color='gray', linestyle='--')
ax_inset.set_title("SIR Infections", fontsize=8)
ax_inset.tick_params(axis='both', which='major', labelsize=6)
ax_inset.grid(True)

plt.tight_layout()
plt.show()

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
