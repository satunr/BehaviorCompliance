import networkx as nx
import numpy as np
import random
import math
import matplotlib.pyplot as plt

# Assume find_seeds and IM are available and work as intended
import find_seeds

from copy import deepcopy

# Set to True if you want to see difference in infection spread w/
# quarantining and non-quarantining for your given param.s.
compare_quarantining_non_quarantining = True

# I modified this code to allow infected individuals to no longer be infected after a certain period
def sirs_step(G, state, L, beta, gamma, mu):
    # Copy the current state to avoid modifying the dictionary while iterating
    new_state = state.copy()

    # Spread infection: Infected individuals attempt to infect susceptible neighbors
    for u in G.nodes:
        if state[u] == 1:  # If u is infected
            for v in G.neighbors(u):
                # Check if v exists and is susceptible, and u can infect v
                if v in state and state[v] == 0 and L[u] == 0:
                    # Probabilistic infection spread
                    if random.random() < beta:
                        new_state[v] = 1  # v becomes infected

    # Recovery: Infected individuals may recover
    for u in G.nodes:
        if state[u] == 1:  # If u is infected
            if random.random() < gamma:
                new_state[u] = 2  # u recovers

    # Immunity loss: Recovered individuals may become susceptible again
    for u in G.nodes:
        if state[u] == 2:
            # If u is recovered
            if random.random() < mu:
                new_state[u] = 0  # u becomes susceptible again

    return new_state


def transition(L, P_prime):
    n_total = len(L) # Total number of nodes based on L length
    all_0s = [i for i in range(n_total) if L[i] == 0]
    all_1s = [i for i in range(n_total) if L[i] == 1]

    # Ensure we don't try to choose more elements than available
    size0 = min(abs(P_prime), len(all_0s))
    size1 = min(abs(P_prime), len(all_1s))

    if P_prime > 0 and len(all_0s) > 0:
        t0 = np.random.choice(all_0s, size=size0, replace=False)
        for i in t0:
            L[i] = 1
    elif P_prime < 0 and len(all_1s) > 0:
        t1 = np.random.choice(all_1s, size=size1, replace=False)
        for i in t1:
            L[i] = 0

    return L


def update_N_P(P, N, n, a=0.5, v=0.05):
    if n == 0: # Avoid division by zero
        return 0, P, N

    X = (P - N) / n
    X_prime = (1 - X) * v * np.exp(a * X) - (1 + X) * v * np.exp(-a * X)  # Compute X'

    # Ensure P_prime calculation is robust if n is small or X_prime is small
    P_prime = math.ceil((n * X_prime) / 2.0) if n * X_prime != 0 else 0

    # Prevent P from going below 0 or above n, adjust N accordingly
    new_P = P + P_prime
    if new_P < 0:
        P_prime = -P # Change P to 0
        new_P = 0
    elif new_P > n:
        P_prime = n - P # Change P to n
        new_P = n

    P = new_P
    # N should be n - P, ensure N doesn't go negative if P > n somehow, though capped above
    N = max(0, n - P)


    return P_prime, P, N

#-----------
#
#  This defines what quarantining will actually do (remove neighboring edges),
#  and its corresponding probabilities
#
#-----------

# g: A graph, states: dictionary mapping node to state
def quarantine_edge_removal(g, node, states):
    is_informed = "Informed" in g.nodes[node] # Check if attribute exists first

    # Check if node's state is 1 (infected)
    # Add check if node still exists in graph, as edges might be removed
    if node in g and states.get(node) == 1 and is_informed == True: # Use .get for safety
        # Get all neighbors of current node *that still exist in g*
        # Need list copy as we modify the graph while iterating
        neighbors = list(g.neighbors(node))
        # Remove edges to all neighbors
        for neighbor in neighbors:
             # Check if edge still exists before removing
             if g.has_edge(node, neighbor):
                g.remove_edge(node, neighbor)

def restore_edges(g_init, g, node):
    # Check if the node exists in both graphs
    if node not in g_init or node not in g:
        return g # Node doesn't exist, nothing to restore

    # Find edges present in g_init but missing in g for the specific node
    initial_neighbors = set(g_init.neighbors(node))
    current_neighbors = set(g.neighbors(node))

    edges_to_add = []
    for neighbor in initial_neighbors:
        if neighbor not in current_neighbors:
            # Ensure the neighbor also exists in the current graph 'g' before adding edge
            if neighbor in g:
                 edges_to_add.append((node, neighbor))

    g.add_edges_from(edges_to_add)
    return g


#-----------
#
#  Use this one if you care about when a particular node is infected
#  Data set consists of ordered triples: (node, state, time of state transition)
#
#-----------

# MODIFIED: Accepts initial graph, state, and durations
def Simulate_SIR(n, T, beta, gamma, mu,
                 initial_graph, initial_state, quarantine_durations,
                 simulation_seed, # Seed for dynamics within this run
                 q=True, allow_restoration=False, verbose=False):

    # Use the provided seed for the dynamics of THIS simulation run
    random.seed(simulation_seed)
    np.random.seed(simulation_seed)

    # --- Use Copies of Initial Conditions ---
    # Create deep copies to avoid modifying the originals passed in
    G = deepcopy(initial_graph)
    state = deepcopy(initial_state)
    d = deepcopy(quarantine_durations)
    # Keep a pristine copy of the graph structure for restoration
    G_initial_structure = deepcopy(initial_graph) # Graph structure ONLY

    # --- Initial Setup based on state ---
    P = sum(1 for node_state in state.values() if node_state == 1) # Count initially infected
    N = n - P # Initially susceptible or recovered (assuming only 0 and 1 initially based on original code)
              # Adjust if initial state can include recovered (2)

    state_changes = []
    # Record initial states at T=0
    for u in range(n):
        state_changes.append((u, state[u], 0)) # Store initial state and time 0

    # Vector with informed (1) and uninformed (0) - This logic seems tied to P/N split, review needed.
    # Let's base L on the initial P and N derived from the actual 'state' dict for consistency
    L = [0] * N + [1] * P # Initial allocation based on counts
    random.shuffle(L) # Shuffle to randomly assign informed/uninformed status initially

    PList = [P]
    Inf = [sum(1 for node_state in state.values() if node_state == 1)] # Initial infection count from state


    # This will hold sums: +1 to ith element if node i is infected 1 at both T(n-1) and T(n)
    quarantine_statuses = [0] * n # Initialize quarantine day counters

    # Assume max_influence/Informed logic is handled externally or adapted if needed
    # Original code had:
    max_influence = find_seeds.initialize_social_IM(G,network_size=n,k=5,p=0.025,num_seeds=15)
    for node in max_influence:
        if state[node] == 1:
            nx.set_node_attributes(G, {node: {'Informed?': 'Informed'}})
    # This needs careful thought on how 'Informed' status should interact/be set initially

    # --- Simulation Loop ---
    for t in range(T):
        copy_state = deepcopy(state) # State at the beginning of the timestep

        # Dynamic updates for P/N (Information Spread Model)
        P_prime, P, N = update_N_P(P, N, n) # n is total nodes
        L = transition(L, P_prime)

        # Update state using SIRS model step (Infection/Recovery/Waning Immunity)
        state = sirs_step(G, state, L, beta, gamma, mu)

        # Process state changes and quarantine logic
        for u in range(n):
            if copy_state.get(u) != state.get(u): # Check if state changed (use .get for safety)
                # Record state change: find existing entry and update, or append if needed (more robust: use dict for state_changes)
                # Simple update assumes index corresponds to node u:
                if u < len(state_changes):
                    state_changes[u] = (u, state[u], t + 1) # Time t+1 for change occurring *during* step t

                # If state changed, reset quarantine counter
                quarantine_statuses[u] = 0

                # If node *stopped* being infected (e.g., recovered), potentially restore edges if not allowing restoration naturally
                # This might be complex if allow_restoration=False is meant to *never* restore
                if copy_state.get(u) == 1 and state.get(u) != 1 and allow_restoration:
                     # Check if edges were previously removed for this node before restoring
                     # This requires tracking which nodes had edges removed.
                     # Simplification: Assume if allow_restoration is True, recovery always restores.
                     restore_edges(G_initial_structure, G, u)


            # If node u remained infected and quarantining is enabled
            elif copy_state.get(u) == 1 and state.get(u) == 1 and q:
                quarantine_statuses[u] += 1 # Increment quarantine days

                # Check if it's the *first* day of sustained infection (counter is 1)
                # OR if you want quarantine to start immediately upon infection (check t=0 case)
                # Apply edge removal ONCE when quarantine starts
                # A simple check: if counter is 1, it's the first day *after* infection started
                if quarantine_statuses[u] == 1: # Start quarantine: remove edges
                     # Add informed check here if necessary
                     quarantine_edge_removal(g=G, node=u, states=state) # Pass current state

                # Check if quarantine duration 'd[u]' is met
                if quarantine_statuses[u] >= d[u]:
                    # Quarantine finished
                    state[u] = 2 # Node recovers (or moves to recovered state)
                    quarantine_statuses[u] = 0 # Reset counter

                    # Restore edges ONLY if allow_restoration is True
                    if allow_restoration:
                        restore_edges(G_initial_structure, G, u)
                    # If allow_restoration is False, edges removed by quarantine_edge_removal STAY removed.


        # Update infection count for plotting
        Inf.append(sum(1 for node_state in state.values() if node_state == 1))
        PList.append(P) # Track informed count

    # --- Prepare Return Data ---
    return_data = []
    x_data = list(range(T + 1))  # Time points
    y_data_inf = Inf  # Infection frequency over time
    return_data.append(x_data)
    return_data.append(y_data_inf)

    # Assign final attributes (optional, if needed)
    for i, node_state, timestamp in state_changes:
       if i in G: # Check node exists
           G.nodes[i]['Infection Status'] = node_state
           G.nodes[i]['Timestamp'] = timestamp


    # Note: Verbose plotting is removed, handled outside now.
    # Return the final graph (might be modified), state changes (might need rework), and plot data
    return G, state_changes, return_data


# ==================================
# Simulation Setup and Execution
# ==================================

n = 100
T = 200
Repeat = 1 # Repeat is now 1, as we compare within one set of initial conditions

# Epidemic parameters
beta = 0.4   # Infection rate
gamma = 0.05 # Recovery rate
mu = 0.05    # Immunity loss rate (Changed from 0.5 which is very high)
init = 0.05  # Initial infection probability

# --- Generate Initial Conditions ONCE ---
master_seed = 42 # Use a fixed seed for reproducibility of initial conditions
random.seed(master_seed)
np.random.seed(master_seed)

# 1. Initial Human contact network
G_initial = nx.erdos_renyi_graph(n, p=0.025, directed=False, seed=master_seed) # Seed graph gen
G_initial = nx.relabel_nodes(G_initial, {u: int(u) for u in G_initial.nodes()})

# 2. Initial state dictionary (0: susceptible, 1: infected, 2: immune/recovered)
#    Ensure seeding for initial state generation
state_initial = {u: np.random.choice(a=[1, 0], size=1, p=[init, 1 - init])[0]
                 for u in range(n)}

# 3. Quarantine durations per individual
samples_array = np.random.normal(loc=14, scale=5, size=n)
d_durations = [abs(round(x)) for x in samples_array] # Ensure non-negative integer durations


# --- Run Simulations with Shared Initial Conditions ---
if compare_quarantining_non_quarantining:
    print("Running simulation WITH edge restoration...")
    # Use a different seed for the simulation dynamics if desired, or reuse master_seed
    # Using the same seed ensures the *random events within the loop* are the same
    simulation_run_seed = 123

    _, _, data1 = Simulate_SIR(n=n, T=T, beta=beta, gamma=gamma, mu=mu,
                               initial_graph=G_initial,
                               initial_state=state_initial,
                               quarantine_durations=d_durations,
                               simulation_seed=simulation_run_seed, # Seed for this run's dynamics
                               q=True, allow_restoration=True, verbose=False)

    print("Running simulation WITHOUT edge restoration...")
    _, _, data2 = Simulate_SIR(n=n, T=T, beta=beta, gamma=gamma, mu=mu,
                               initial_graph=G_initial,
                               initial_state=state_initial,
                               quarantine_durations=d_durations,
                               simulation_seed=simulation_run_seed, #<< USE SAME SEED HERE
                               q=True, allow_restoration=False, verbose=False) # The only difference

    # --- Plotting Comparison ---
    # Extract x and y data from arrays
    x1, y1 = data1[0], data1[1]
    x2, y2 = data2[0], data2[1]

    plt.figure(figsize=(10, 6)) # Make plot larger
    plt.plot(x1, y1, label='With Restoration', color='blue', marker='.', linestyle='-', markersize=4)
    plt.plot(x2, y2, label='Without Restoration', color='red', marker='.', linestyle='--', markersize=4)

    # Customize the plot
    plt.xlabel('Time Step (Days)')
    plt.ylabel('Number of Infected Individuals')
    plt.title('Infection Spread Comparison: Edge Restoration vs. No Restoration')
    plt.legend()  # Add legend to distinguish the lines
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(bottom=0) # Ensure y-axis starts at 0
    plt.tight_layout() # Adjust layout

    # Show the plot
    plt.show()