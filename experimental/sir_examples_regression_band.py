import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from copy import deepcopy
import SIR  # Your custom SIR module
import correlated_graphs  # Optional, if used elsewhere
import pickle

# Parameters
n = 100
T = 100
Repeat = 1
beta = 0.20
gamma = 0.07
mu = 0.10
init = 0.15
num_trials = 10

# Networks
contact_network = nx.erdos_renyi_graph(n, 0.05, seed=42)
social_network = nx.erdos_renyi_graph(n, 0.05, seed=42).to_directed()


def linear_fit_with_error(x, y):
    x = np.array(x)  # Ensure x is a NumPy array
    a, b = np.polyfit(x, y, 1)
    y_est = a * x + b
    y_err = x.std() * np.sqrt(1 / len(x) + (x - x.mean())**2 / np.sum((x - x.mean())**2))
    return y_est, y_err, a, b


def const_quarantines():
    quarantine_constant = 14
    T_runs, Y_quarantine, Y_noquarantine = [], [], []

    for _ in range(num_trials):
        data1 = SIR.Simulate_SIR(deepcopy(contact_network), deepcopy(social_network), T, Repeat, beta, gamma, mu, init, False, q=quarantine_constant, allow_restoration=True)[2]
        data2 = SIR.Simulate_SIR(deepcopy(contact_network), deepcopy(social_network), T, Repeat, beta, gamma, mu, init, False, q=False, allow_restoration=False)[2]
        T_runs.append(data1[0])
        Y_quarantine.append(data1[1])
        Y_noquarantine.append(data2[1])

    x = T_runs[0]
    mean_q = np.mean(Y_quarantine, axis=0)
    mean_noq = np.mean(Y_noquarantine, axis=0)

    y_est_q, y_err_q, _, _ = linear_fit_with_error(x, mean_q)
    y_est_noq, y_err_noq, _, _ = linear_fit_with_error(x, mean_noq)

    plt.plot(x, y_est_q, '-', label='With Quarantine (q=14)', color='blue')
    plt.fill_between(x, y_est_q - y_err_q, y_est_q + y_err_q, color='blue', alpha=0.2)
    plt.plot(x, mean_q, 'o', color='blue')

    plt.plot(x, y_est_noq, '-', label='Without Quarantine', color='red')
    plt.fill_between(x, y_est_noq - y_err_noq, y_est_noq + y_err_noq, color='red', alpha=0.2)
    plt.plot(x, mean_noq, 'o', color='red')

    plt.title('Temporary Quarantine: Linear Fit with Error Band')
    plt.xlabel('Time')
    plt.ylabel('# of Infected')
    plt.legend()
    plt.grid(True)
    plt.show()


def normal_dist_quarantines():
    T_runs, Y_normal, Y_noquarantine = [], [], []

    for _ in range(num_trials):
        data1 = SIR.Simulate_SIR(deepcopy(contact_network), deepcopy(social_network), T, Repeat, beta, gamma, mu, init, False, q=True, allow_restoration=True)[2]
        data2 = SIR.Simulate_SIR(deepcopy(contact_network), deepcopy(social_network), T, Repeat, beta, gamma, mu, init, False, q=False, allow_restoration=False)[2]
        T_runs.append(data1[0])
        Y_normal.append(data1[1])
        Y_noquarantine.append(data2[1])

    x = T_runs[0]
    mean_norm = np.mean(Y_normal, axis=0)
    mean_noq = np.mean(Y_noquarantine, axis=0)

    y_est_norm, y_err_norm, _, _ = linear_fit_with_error(x, mean_norm)
    y_est_noq, y_err_noq, _, _ = linear_fit_with_error(x, mean_noq)

    plt.plot(x, y_est_norm, '-', label='With Normal Dist. Quarantine', color='blue')
    plt.fill_between(x, y_est_norm - y_err_norm, y_est_norm + y_err_norm, color='blue', alpha=0.2)
    plt.plot(x, mean_norm, 'o', color='blue')

    plt.plot(x, y_est_noq, '-', label='Without Quarantine', color='red')
    plt.fill_between(x, y_est_noq - y_err_noq, y_est_noq + y_err_noq, color='red', alpha=0.2)
    plt.plot(x, mean_noq, 'o', color='red')

    plt.title('Normal Dist. Quarantine: Linear Fit with Error Band')
    plt.xlabel('Time')
    plt.ylabel('# of Infected')
    plt.legend()
    plt.grid(True)
    plt.show()


def informed_vs_noninformed():
    T_runs, Y_informed, Y_noninformed = [], [], []

    for _ in range(num_trials):
        data1 = SIR.Simulate_SIR(deepcopy(contact_network), deepcopy(social_network), T, Repeat, beta, gamma, mu, init, False, q=True, allow_restoration=False)[2]
        data2 = SIR.Simulate_SIR(deepcopy(contact_network), deepcopy(social_network), T, Repeat, beta, gamma, mu, init, False, q=False, allow_restoration=False)[2]
        T_runs.append(data1[0])
        Y_informed.append(data1[1])
        Y_noninformed.append(data2[1])

    x = T_runs[0]
    mean_inf = np.mean(Y_informed, axis=0)
    mean_noinf = np.mean(Y_noninformed, axis=0)

    y_est_inf, y_err_inf, _, _ = linear_fit_with_error(x, mean_inf)
    y_est_noinf, y_err_noinf, _, _ = linear_fit_with_error(x, mean_noinf)

    plt.plot(x, y_est_inf, '-', label='With Informed', color='blue')
    plt.fill_between(x, y_est_inf - y_err_inf, y_est_inf + y_err_inf, color='blue', alpha=0.2)
    plt.plot(x, mean_inf, 'o', color='blue')

    plt.plot(x, y_est_noinf, '-', label='Without Informed', color='red')
    plt.fill_between(x, y_est_noinf - y_err_noinf, y_est_noinf + y_err_noinf, color='red', alpha=0.2)
    plt.plot(x, mean_noinf, 'o', color='red')

    plt.title('Permanent Quarantine: Linear Fit with Error Band')
    plt.xlabel('Time')
    plt.ylabel('# of Infected')
    plt.legend()
    plt.grid(True)
    plt.show()

# Pickle results from the functions
def SIR_pickle_dump(filename='pickles.pkl'):
    # We will pickle these parameters along with the results for later reference
    presets = {'T': T, 'Repeat': Repeat, 'beta': beta, 'gamma': gamma, 'mu': mu, 'init': init, 'num_trials': num_trials}
    with open(filename, 'wb') as f:
        # Clear the file before writing
        f.truncate(0)
        pickle.dump({'presets': presets}, f)
    print("Data has been pickled successfully.")

SIR_pickle_dump()

def pickle_load(filename='pickles.pkl'):
    # Open the file in binary read mode
    with open(filename, 'rb') as file:
        data = pickle.load(file)

    # Now `data` holds the deserialized object
    print(data)

# Run all
if __name__ == "__main__":
    informed_vs_noninformed()
    const_quarantines()
    normal_dist_quarantines()
