import SIR
import networkx as nx
import correlated_graphs
from copy import deepcopy
import numpy as np
from scipy.optimize import minimize
import random
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import csv

contact_graph = nx.erdos_renyi_graph(200, 0.05, seed=42)
# Assume a social graph that is twice as dense
social_graph = correlated_graphs.create_social_graph(contact_graph, nE=2 * len(contact_graph.edges()))[0]

T = 25
Repeat = 1

beta = 0.15  #infection rate
gamma = 0.10  # recovery rate
mu = 0.10   # immunity loss
init = 0.15

SIR_results = SIR.Simulate_SIR(
    contact_network=deepcopy(contact_graph),
    social_network=deepcopy(social_graph),
    T=T, Repeat=Repeat,
    beta=beta, gamma=gamma, mu=mu, init=init,
    average_data=False, q=True, allow_restoration=True, save_all=True
)

mean_degrees = SIR_results[6]

# Form of these tuples is (node, status, timestamp)
# For status, 0 = susceptible, 1 = infected, 2 = recovered
true_dynamics = SIR_results[4]
print("True dynamics:", true_dynamics[0])
num_nodes = len(contact_graph.nodes())

#---------
#
#  Optimization problem: y = (w1 * x1) * (x2 - w2) + noise
#
#--------

true_w = []
for time in range(T):
    # w_1: mean degree of the contact graph
    true_w1 = mean_degrees[time]
    # w_2: proportion of recovered nodes at time 'time'
    true_w2 = sum(1 for node in contact_graph.nodes() if true_dynamics[time][node] == 2) / num_nodes

    true_w.append((true_w1, true_w2))

def given_at_time(time):
    # if time == 0: # At t=0, we are dealing with inital conditions of the simulation
    #     num_new_recovered = sum(1 for node in contact_graph.nodes() if true_dynamics[time][node] == 2)
    #     num_new_infected = sum(1 for node in contact_graph.nodes() if true_dynamics[time][node] == 1)
    # elif time > 0 and time < T:
        # Look for state changes to find newly recovered, infected
    num_new_recovered = sum(1 for node in contact_graph.nodes() if true_dynamics[time][node] == 2 and true_dynamics[time-1][node] < 2)
    num_new_infected = sum(1 for node in contact_graph.nodes() if true_dynamics[time][node] == 1 and true_dynamics[time-1][node] != 1)

    x1 = beta * (num_new_recovered / gamma)
    x2 = 1 - (num_new_infected / gamma)

    return (x1, x2)

def y_true(time):
    x1, x2 = given_at_time(time)

    true_w1, true_w2 = true_w[time]
    y = (true_w1 * x1) * (x2 - true_w2)

    return y

# Define the loss function to minimize (MSE)
# Form of params is [w1, w2]

T_gen = 0

def loss(params):
    w1, w2 = params
    x1, x2 = given_at_time(T_gen)
    y_pred = (w1 * x1) * (x2 - w2)

    penalties = 0
    infinity = 1e10  # Large penalty for out-of-bounds parameters
    if w1 > num_nodes or w1 < 0 or w2 > 1 or w2 < 0:
        penalties = infinity
        
    return np.mean((y_pred - y_true(T_gen)) ** 2) + penalties

# Initial guess for parameters
init_guess = [1.0, 1.0]
bounds = [(1, num_nodes), (0.0001, 1)]

# Perform optimization
# result = minimize(loss, init_guess, method='L-BFGS-B', bounds=bounds)  # You can try 'Nelder-Mead' or others too

w1_estimates = []
w2_estimates = []

for t in range(1, T):
    T_gen = t
    result = minimize(loss, init_guess, method='L-BFGS-B', bounds=bounds)
    
    print("T_gen =", T_gen)
    print("True w1:", true_w[T_gen][0])
    print("True w2:", true_w[T_gen][1])
    print("Estimated w1:", result.x[0])
    print("Estimated w2:", result.x[1])
    print("Squared error:", result.fun)

    w1_estimates.append(result.x[0])
    w2_estimates.append(result.x[1])

# Main plot
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(range(1, T), [true_w[t][0] for t in range(1, T)], label='True w1', marker='o')
ax.plot(range(1, T), w1_estimates, label='Estimated w1', marker='x')
ax.plot(range(1, T), [true_w[t][1] for t in range(1, T)], label='True w2', marker='o')
ax.plot(range(1, T), w2_estimates, label='Estimated w2', marker='x')

ax.set_xticks(range(T))
ax.set_xlabel('Time')
ax.set_ylabel('w1 and w2 values')
ax.set_title('True vs Estimated w1 and w2 over Time')
ax.legend()
ax.grid(True)

# Add inset for SIR infections
sir_infections = [sum(1 for node in contact_graph.nodes() if true_dynamics[t][node] == 1) for t in range(T)]
ax_inset = inset_axes(ax, width="30%", height="30%", loc='center')  # You can tweak size/position
ax_inset.plot(range(T), sir_infections, color='gray', linestyle='--')
ax_inset.set_title("SIR Infections", fontsize=8)
ax_inset.tick_params(axis='both', which='major', labelsize=6)
ax_inset.grid(True)

plt.tight_layout()
plt.show()


# Choose a specific time step for loss landscape visualization
T_gen = 5  # You can try different values like 5, 10, 15...

# Create a grid of w1 and w2 values
w1_vals = np.linspace(1, num_nodes, 100)
w2_vals = np.linspace(0.0001, 1.0, 100)

W1, W2 = np.meshgrid(w1_vals, w2_vals)
Z = np.zeros_like(W1)

# Compute loss at each (w1, w2) pair
for i in range(W1.shape[0]):
    for j in range(W1.shape[1]):
        Z[i, j] = loss((W1[i, j], W2[i, j]))

# Plot the loss landscape
fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(W1, W2, Z, levels=50, cmap='viridis')
plt.colorbar(contour, ax=ax, label='Loss')
ax.set_title(f"Loss Landscape at T_gen = {T_gen}")
ax.set_xlabel("w1")
ax.set_ylabel("w2")

# Optionally, plot the true and estimated parameters
ax.plot(true_w[T_gen][0], true_w[T_gen][1], 'ro', label='True (w1, w2)')
result = minimize(loss, init_guess, method='L-BFGS-B', bounds=bounds)
ax.plot(result.x[0], result.x[1], 'bx', label='Estimated (w1, w2)')
ax.legend()

plt.tight_layout()
plt.show()