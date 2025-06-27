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
n = 150
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
q = False

SIR_results = SIR.Simulate_SIR(
    contact_network=deepcopy(contact_graph),
    social_network=deepcopy(social_graph),
    T=T, Repeat=Repeat,
    beta=beta, gamma=gamma, mu=mu, init=init,
    average_data=False, q=q, allow_restoration=False, save_all=True
)

mean_degrees = np.mean(SIR_results[6])   # Mean degree of contact network over time
true_dynamics = SIR_results[4]
num_nodes = len(contact_graph.nodes())

# Compute true w1 and w2 values: Mean degree and fraction of recovered nodes
true_w = []
for time in range(T):
    true_w1 = mean_degrees
    true_w2 = sum(1 for node in contact_graph.nodes() if true_dynamics[time][node] == 2) / num_nodes  # Recovered portion
    true_w.append((true_w1, true_w2))

def given_at_time(time):
    num_new_recovered = sum(1 for node in contact_graph.nodes() 
                            if true_dynamics[time][node] == 2 and true_dynamics[time-1][node] < 2)
    x1 = beta * (num_new_recovered / gamma)
    x2 = 1 - (num_new_recovered / gamma)
    return (x1, x2)

def y_true(time):
    x1, x2 = given_at_time(time)
    true_w1, true_w2 = true_w[time]

    return (true_w1 * x1) * (x2 - true_w2)

def loss(params, prev_w2, T_gen):
    eps = 0.1

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

for i in range(1, T):
    print(f"Given x's at time {i}: {given_at_time(i)}")
    print(f"New recovered at time {i}: {sum(1 for node in contact_graph.nodes() if true_dynamics[i][node] == 2 and true_dynamics[i-1][node] < 2)}")