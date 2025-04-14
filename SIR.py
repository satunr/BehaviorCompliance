import networkx as nx
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import find_seeds
import correlated_graphs
import parse

from copy import deepcopy

# Set to True if you want to see difference in infection spread w/ 
#   quarantining and non-quarantining for your given param.s.
compare_quarantining_non_quarantining = True

# I modified this code to allow infected individuals to no longer be infected after a certain period
def sirs_step(G, state, L, beta, gamma, mu):

    # Copy the current state to avoid modifying the dictionary while iterating
    new_state = state.copy()

    print("state: ", state)

    # Spread infection: Infected individuals attempt to infect susceptible neighbors
    for u in G.nodes:
        if state[u] == 1:  # If u is infected
            for v in G.neighbors(u):
                if state[v] == 0 and L[u] == 0:  # If v is susceptible and u can infect v
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

    all_0s = [i for i in range(len(L)) if L[i] == 0]
    all_1s = [i for i in range(len(L)) if L[i] == 1]

    t0 = np.random.choice(all_0s, size=P_prime)
    t1 = np.random.choice(all_1s, size=P_prime)

    if P_prime > 0:
        for i in t0:
            L[i] = 1
    elif P_prime < 0:
        for i in t1:
            L[i] = 0

    return L


def update_N_P(P, N, n, a=0.5, v=0.05):

    X = (P - N) / n
    X_prime = (1 - X) * v * np.exp(a * X) - (1 + X) * v * np.exp(-a * X)  # Compute X'

    P_prime = math.ceil((n * X_prime) / 2.0)

    P = P + P_prime
    N = N - P_prime

    return P_prime, P, N

#-----------
#
#  This defines what quarantining will actually do (remove neighboring edges), 
#   and its corresponding probabilities
#
#-----------

# g: A graph, (mean, std): Parameters assoc. with taking P(edge removal) as a normal dist. sample, 
#   w/ threshold as another param.
def quarantine_edge_removal(g, node, states):
    # Boolean value; Checks if "Informed" is an attribute of the node under consideration
    is_informed = "Informed" in g.nodes[node].values()

    # Check if node's state is 1 (active)
    if states[node] == 1 and is_informed == True:
        # Get all neighbors of current node
        neighbors = list(g.neighbors(node))
        # Remove edges to all neighbors
        for neighbor in neighbors:
            g.remove_edge(node, neighbor)

def restore_edges(g_init, g, node):
    initial_connections = [(node, neighbor) for neighbor in g_init.neighbors(node)]
    final_connections = [(node, neighbor) for neighbor in g.neighbors(node)]
    set_diff = set(initial_connections) - set(final_connections)

    g.add_edges_from(set_diff)

    return g


#-----------
#
#  Use this one if you care about when a particular node is infected
#  Data set consists of ordered triples: (node, state, time of state transition)
#
#-----------

# Wrapper I wrote around the example code that was here to begin with
# q: Set to true if you want quarantine periods to be factored into algorithm
def Simulate_SIR(contact_network,social_network,T,Repeat,beta,gamma,mu,init,verbose,q=True,allow_restoration=False):
    n = len(contact_network.nodes())
    All_Init = {t: [] for t in range(T + 1)}
    for _ in range(Repeat):
        N = n - 1
        P = n - N

        # k: # of seeds we want
        max_influence = find_seeds.initialize_social_IM(social_network=social_network,k=5,p=0.025,num_seeds=15)  # Create a social network, run IM on it

        # We are saving the initial state of G so we know what connections to restore later
        G_initial = deepcopy(contact_network)

        # Initial state dictionary (0: susceptible, 1: infected, 2: immune)
        state = {u: np.random.choice(a=[1, 0], size=1, p=[init, 1 - init])[0]
                 for u in range(n)}  # Note: Initial setup only uses 0 and 1

        state_changes = []
        # Record initial states at T=0
        for u in range(n):
            state_changes.append((u, state[u], 0))

        # Vector with informed (1) and uninformed (0)
        L = [0 for _ in range(N)] + [1 for _ in range(P)]
        PList = [P]
        Inf = [len([u for u in state.keys() if state[u] == 1])]

        # d[i]: # of days individual i chooses to quarantine
        samples_array = np.random.normal(loc=14, scale=5, size=n)
        # Take abs. value of the samples, and round to nearest int. val.
        d = [abs(round(x)) for x in samples_array]

        # This will hold sums: +1 to ith element if node i is infected 1 at both T(n-1) and T(n)
        quarantine_statuses = [0 for _ in range(0, n)]

        for node in max_influence:
            if state[node] == 1:
                nx.set_node_attributes(contact_network, {node: {'Informed?': 'Informed'}})

        for t in range(T):
            copy_state = deepcopy(state)

            # Dynamic updates
            P_prime, P, N = update_N_P(P, N, n)
            L = transition(L, P_prime)

            # Update state using SIRS model
            state = sirs_step(contact_network, state, L, beta, gamma, mu)

            # Analyze state changes across all nodes
            for u in range(n):
                if copy_state[u] != state[u]:
                    # Record in state changes the time at which said change occurred
                    state_changes[u] = (u, state[u], t)

                    # If there was a state change for node u, we can say that node u has not been in quarantine from T(n-1) to T(n),
                    # regardless of what it becomes here
                    quarantine_statuses[u] = 0
                
                # if applicable (node u infected in both T(n-1) and T(n)), update quarantine status for node u
                elif copy_state[u] == state[u] and copy_state[u] == 1 and q == True:

                    # Individual chooses, at the first step in their infection, whether they want to quarantine
                    if quarantine_statuses[u] == 1:
                        quarantine_edge_removal(g=contact_network, node=u, states=state)

                    # Add +1 to their quarantine period
                    quarantine_statuses[u] = quarantine_statuses[u] + 1

                    # If the quarantine time has been reached for the individual, remove from the infected category
                    if quarantine_statuses[u] >= d[u]:
                        if allow_restoration:
                            restore_edges(G_initial, contact_network, u)
                        state[u] = 2

            # Update infection count for plotting
            Inf.append(len([u for u in state.keys() if state[u] == 1]))
            PList.append(P)

            # Assign attributes to each node
            for i, _, _ in state_changes:
                contact_network.nodes[i]['Infection Status'] = state_changes[i][1]
                contact_network.nodes[i]['Timestamp'] = state_changes[i][2]

        return_data = []
        # Store initial infection data
        x_data = [t for t in range(T + 1)]  # Time points
        y_data_inf = Inf  # Infection frequency
        return_data.append(x_data)
        return_data.append(y_data_inf)

        if verbose:
            # May delete this part later

            # Plot infection frequency
            plt.plot([t for t in range(T + 1)], Inf, alpha=0.1)
            for t in range(T + 1):
                All_Init[t].append(Inf[t])

            plt.plot([t for t in range(T + 1)],
                     [np.mean(All_Init[t]) for t in range(T + 1)], linewidth=3)
            plt.xlabel('Time', fontsize=12)
            plt.ylabel('Number of individuals', fontsize=12)
            plt.ylim([0, 60])
            plt.legend()
            plt.tight_layout()
            plt.savefig('Information.png')
            plt.show()

    # Return the graph and the list of state change tuples
    # G's attributes are changed here
    return contact_network, state_changes, return_data


#----------
#
#  Comparison of quarantining vs no quarantining
#
#----------

if compare_quarantining_non_quarantining == True:
    n = 100
    T = 200
    Repeat = 1

    beta = 0.10  #infection rate
    gamma = 0.05  # recovery rate
    mu = 0.10   # immunity loss
    init = 0.05

    #---------
    #
    #  Generate social network w/ 1-hop correlation
    #
    #---------

    # contact_network = nx.erdos_renyi_graph(n, p=0.025, directed=False)
    # contact_network = nx.relabel_nodes(contact_network, {u: int(u) for u in contact_network.nodes()})

    # Specify the filename
    filename = 'contact_network_text.txt'
    # Create the graph from the file
    contact_graph = parse.parse(filename)

    # Relabel nodes in parsed graph to avoid off by 1 errors in SIR.py
    # Create a mapping from old node to new node: i -> i - 1
    mapping = {node: node - 1 for node in contact_graph.nodes()}

    # Relabel the nodes
    contact_network = nx.relabel_nodes(contact_graph, mapping)

    social_network = correlated_graphs.create_w_k_hop_correlation(contact_network,k=1)[0]   # We just want the graph part of this output

    # NOTE: This data is derived from 2 separate runs of Simulate_SIR, and is therefore only an approx. comparison
    data1 = Simulate_SIR(contact_network=contact_network,social_network=social_network,T=T,Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init,verbose=False,q=True,allow_restoration=True)[2]
    data2 = Simulate_SIR(contact_network=contact_network,social_network=social_network,T=T,Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init,verbose=False,q=True,allow_restoration=False)[2]
    data3 = Simulate_SIR(contact_network=contact_network,social_network=social_network,T=T,Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init,verbose=False,q=False,allow_restoration=False)[2]

    # Extract x and y data from arrays
    x1, y1 = data1[0], data1[1]
    x2, y2 = data2[0], data2[1]
    x3, y3 = data3[0], data3[1]

    # Create the plot
    plt.plot(x1, y1, label='With Restoration', color='blue', marker='o')
    plt.plot(x2, y2, label='Without Restoration', color='red', marker='o')
    plt.plot(x3, y3, label='No quarantine', color='green', marker='o')

    # Customize the plot
    plt.xlabel('Time')
    plt.ylabel('# of Infected')
    plt.title('Quarantine Comparison Social 1-hop')
    plt.legend()  # Add legend to distinguish the lines
    plt.grid(True)

    # Show the plot
    plt.show()

    #----------
    #
    #  Generate social network with 2-hop correlation
    #
    #----------

    social_network = correlated_graphs.create_w_k_hop_correlation(contact_network,k=2)[0]   # We just want the graph part of this output

    # NOTE: This data is derived from 2 separate runs of Simulate_SIR, and is therefore only an approx. comparison
    data1 = Simulate_SIR(contact_network=contact_network,social_network=social_network,T=T,Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init,verbose=False,q=True,allow_restoration=True)[2]
    data2 = Simulate_SIR(contact_network=contact_network,social_network=social_network,T=T,Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init,verbose=False,q=True,allow_restoration=False)[2]
    data3 = Simulate_SIR(contact_network=contact_network,social_network=social_network,T=T,Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init,verbose=False,q=False,allow_restoration=False)[2]

    # Extract x and y data from arrays
    x1, y1 = data1[0], data1[1]
    x2, y2 = data2[0], data2[1]
    x3, y3 = data3[0], data3[1]

    # Create the plot
    plt.plot(x1, y1, label='With Restoration', color='blue', marker='o')
    plt.plot(x2, y2, label='Without Restoration', color='red', marker='o')
    plt.plot(x3, y3, label='No quarantine', color='green', marker='o')

    # Customize the plot
    plt.xlabel('Time')
    plt.ylabel('# of Infected')
    plt.title('Quarantine Comparison Social 2-hop')
    plt.legend()  # Add legend to distinguish the lines
    plt.grid(True)

    # Show the plot
    plt.show()