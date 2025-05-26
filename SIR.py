import networkx as nx
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import find_seeds
import correlated_graphs
# import parse
import IM

from copy import deepcopy

# Set to True if you want to see difference in infection spread w/ 
#   quarantining and non-quarantining for your given param.s.
compare_quarantining_non_quarantining = False

# I modified this code to allow infected individuals to no longer be infected after a certain period
def sirs_step(G, state, L, beta, gamma, mu):

    # Copy the current state to avoid modifying the dictionary while iterating
    new_state = state.copy()

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
    # Determine connections that were removed
    initial_connections = [(node, neighbor) for neighbor in g_init.neighbors(node)]
    # Determine connections that are currently present
    final_connections = [(node, neighbor) for neighbor in g.neighbors(node)]
    # Find the difference between the two sets
    set_diff = set(initial_connections) - set(final_connections)

    g.add_edges_from(set_diff)

    return g

# misinformation_prob: Probability of informed nodes becoming misinformed
# quarantine_required: Number of days a node choses to quarantine before restoring edges
def misinformation_update(g, informed, quarantining, misinformation_prob, quarantine_required):
    misinformed = []
    for node in informed:
        # Randomly decide whether to update the misinformation status
        if random.random() < misinformation_prob:
            # Update the node's misinformation status
            g.nodes[node]['Informed?'] = 'Misinformed'
            misinformed.append(node)
            # Misinformed chooses hold long to not quarantine ~ N. dist.
            quarantine_required[node] = round(np.random.normal(12, 3))
            # Add new edges to the node to spread misinformation ~ N. dist.
            for _ in range(0, round(np.random.normal(5, 2))):
                # Get list of nodes that are not current neighbors
                non_neighbors = [n for n in g.nodes() if n != node and n not in g.neighbors(node) and quarantining[n] == 0]
                # Only add edge if there are non-neighbor nodes available
                if non_neighbors != []:
                    # Randomly select a non-neighbor to connect to
                    neighbor = random.choice(non_neighbors)
                    # Ensure the neighbor is not quarantining
                    if quarantining[neighbor] == 0:
                        g.add_edge(node, neighbor)

    return misinformed, quarantine_required


#-----------
#
#  Use this one if you care about when a particular node is infected
#  Data set consists of ordered triples: (node, state, time of state transition)
#
#-----------

# Wrapper I wrote around the example code that was here to begin with
# q: Set to true if you want quarantine periods to be factored into algorithm
# save_all: Returns large array containing quarantine states at every time interval
def Simulate_SIR(contact_network,social_network,T,Repeat,beta,gamma,mu,init,average_data,
                 q=True,allow_restoration=False,save_all=False,lt_threshold=None, misinformation_prob=None):
    if social_network == None:
        social_network = correlated_graphs.create_w_k_hop_correlation(contact_network, k=2)[0]

    n = len(contact_network.nodes())
    All_Init = {t: [] for t in range(T + 1)}
    for _ in range(Repeat):
        N = n - 1
        P = n - N

        informed = find_seeds.find_seed_set(social_network, num_seeds=15,exponent=1)

        # Set labels for informed set
        for node in informed:
            nx.set_node_attributes(contact_network, {node: {'Informed?': 'Informed'}})
            nx.set_node_attributes(social_network, {node: {'Informed?': 'Informed'}})

        for node in contact_network.nodes(): 
            if node not in informed:
                nx.set_node_attributes(contact_network, {node: {'Informed?': 'Uninformed'}})
                nx.set_node_attributes(social_network, {node: {'Informed?': 'Uninformed'}})

        # We are saving the initial state of G so we know what connections to restore later
        G_initial = deepcopy(contact_network)

        # Initial state dictionary (0: susceptible, 1: infected, 2: immune)
        state = {u: np.random.choice(a=[1, 0], size=1, p=[init, 1 - init])[0]
                 for u in range(n)}  # Note: Initial setup only uses 0 and 1

        state_changes = []
        # Record initial states at T=0
        for u in range(n):
            state_changes.append((u, state[u], 0))

        # L[i] == 0: Can't infect. L[i] == 1: Can infect
        L = [0 for _ in range(N)] + [1 for _ in range(P)]
        PList = [P]
        Inf = [len([u for u in state.keys() if state[u] == 1])]

        # d[i]: # of days individual i chooses to quarantine (or not quarantine if misinformed)
        samples_array = np.random.normal(loc=14, scale=5, size=n)
        # Take abs. value of the quarantine period samples, and round to nearest int. val.
        d = [abs(round(x)) for x in samples_array]

        # Holds dynamic quarantine lengths
        quarantine_statuses = [0 for _ in range(0, n)]

        all_quaratines = []
        all_infections = []

        quarantine_prob_matrix = np.zeros((T, n)) # T x n matrix: hold probabilities of quarantine for each node at each time step
        
        misinformed = []
        if misinformation_prob: 
            misinformed = misinformation_update(contact_network, informed=informed, quarantining=quarantine_statuses,
                                            misinformation_prob=misinformation_prob, quarantine_required=d)[0]

        for t in range(T):

            #--------
            #
            #  Make changes to the set of informed nodes
            #
            #-------

            if lt_threshold == None: # If we are using I.C., that is
                ic_results = IM.IC_prob_matrix(social_network, S=informed, p=0.03, mc=1000, quarantining=quarantine_statuses)
                prob_matrix = ic_results[0]
                new_informed = ic_results[1]

                quarantine_prob_matrix[t] = prob_matrix

            else:
                lt_results = IM.lt_prob_matrix(social_network, threshold=lt_threshold, S=informed, quarantining=quarantine_statuses)
                prob_matrix = lt_results[0]
                new_informed = lt_results[1]

                quarantine_prob_matrix[t] = prob_matrix
            
            # Set labels for informed set
            for node in new_informed:
                nx.set_node_attributes(contact_network, {node: {'Informed?': 'Informed'}})
                nx.set_node_attributes(social_network, {node: {'Informed?': 'Informed'}})

            informed = informed + new_informed
            informed = list(set(informed))  # Remove duplicates

            if save_all == True:
                all_quaratines.append(quarantine_statuses.copy())
                all_infections.append(state)

            #--------
            #
            #  Make contact network changes
            #
            #--------

            # If we are using misinformation, update the contact network
            if misinformation_prob != None:
                misinformation_step = misinformation_update(g=contact_network, informed=informed, quarantining=quarantine_statuses, 
                                                            misinformation_prob=misinformation_prob, quarantine_required=d)
                misinformed = misinformed + misinformation_step[0]  # Update misinformed set
                d = misinformation_step[1]  # Update quarantine choice for uninformed: They choose a negative quarantine period
                if misinformed: informed = [item for item in informed if item not in misinformed]  # Update informed set to not include misinformed nodes

            # Dynamic updates
            P_prime, P, N = update_N_P(P, N, n)
            L = transition(L, P_prime)

            # Make copy before modifying
            copy_state = deepcopy(state)
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
                    if q == True:
                        quarantine_edge_removal(g=contact_network, node=u, states=state)

                # if applicable (node u infected in both T(n-1) and T(n), and is informed), update quarantine status for node u
                elif copy_state[u] == state[u] and copy_state[u] == 1 and q == True and contact_network.nodes[u]['Informed?'] == 'Informed':
                    # Individual chooses, at the first step in their infection, whether they want to quarantine
                    # Add +1 to their quarantine period
                    quarantine_statuses[u] = quarantine_statuses[u] + 1

                    # If the quarantine time has been reached for the individual, remove from the infected category
                    if quarantine_statuses[u] >= d[u]:
                        if allow_restoration:
                            restore_edges(G_initial, contact_network, u)
                        state[u] = 2

                # Update status for misinformed individuals to not quarantine
                if misinformation_prob != None and contact_network.nodes[u]['Informed?'] == 'Misinformed':
                    # "quarantine" here is really a choice to not quarantine, for misinformed individuals
                    quarantine_statuses[u] = quarantine_statuses[u] + 1

                    # If the quarantine time has been reached for the individual, remove from the misinformed category
                    if quarantine_statuses[u] >= d[u]:
                        # Remove the misinformed status
                        contact_network.nodes[u]['Informed?'] = 'Informed'
                        misinformed.remove(u)
                        informed.append(u)

                        # Individual makes choice on quarantine distribution, instead of non-quarantine distribution
                        d[u] = round(np.random.normal(loc=14, scale=5))
                        

            # Update infection count for plotting
            Inf.append(len([u for u in state.keys() if state[u] == 1]))
            PList.append(P)

            # Assign attributes to each node
            for i, _, _ in state_changes:
                contact_network.nodes[i]['Infection Status'] = state_changes[i][1]
                contact_network.nodes[i]['Timestamp'] = state_changes[i][2]

        infection_data = []
        # Store initial infection data
        x_data = [t for t in range(T + 1)]  # Time points
        y_data_inf = Inf  # Infection frequency
        infection_data.append(x_data)
        infection_data.append(y_data_inf)

        if average_data:
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

    # Return the graph and the list of state change tuples, and all quarantine statuses (if applicable)
    if save_all == True:
        return contact_network, state_changes, infection_data, quarantine_prob_matrix, all_infections, social_network
    return contact_network, state_changes, infection_data