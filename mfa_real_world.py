import sys
from copy import deepcopy
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import random
import os
import pickle

#----------
#
#  Same computations as in mean_field_approximation.py, 
#    but SIR data is from real-world numerical data -> no network structures given
#
#----------

#----------
#
#  Mean-field approximation for SIR model: n_i = beta * <k> (1 - n_r / gamma - r) * n_r / gamma
#  Assuming n_i, n_r, SIR parameters are known, this gives us the constrained, non-linear optimization problem:
#    y = w1 * x1 * (x2 - w2), where <w1, w2> are the model weights to be learned (mean node degree, recovered ratio, respectively).
#
#----------

clear = False  # Set clear to True if you want to use a new network or clear data files. False if you want to keep the existing one.
verbose = False  # Set verbose to True if you want to see detailed output during optimization

#  NOTE: Real-world parameters. These are roughly based on the COVID-19 pandemic (worldwide)
population = 7.8 * 10**9  # World population (~7.8 billion)
beta = 0.3
gamma = 0.1
mu = 0.0055
max_mnd = 200  # Maximum mean node degree (w1)
split_point = 50  # Time to split the optimization (in days)

#  Load the daily Covid data from experiment_data/daily_covid.csv
data_path = "experiment_data/daily_covid.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file {data_path} not found. Please ensure the file exists.")

with open(data_path, 'r') as f:
    data = f.read()

# Parsing the CSV text:
lines = data.strip().split('\n')
headers = lines[0].split(',')

# We'll store data in a list indexed by integers (0-based)
df = []

for line in lines[1:]:
    fields = line.split(',')
    # Convert each field except date to int or float
    values = []
    for val in fields[1:]:
        try:
            values.append(int(val))
        except ValueError:
            values.append(float(val))
    # Create a dict with headers as keys
    entry = dict(zip(headers[1:], values))
    df.append(entry)

T = len(df)

# Clear data files
def truncate_files():
    files_to_truncate = ["experiment_data/mfa_xy_data.pkl", "experiment_data/mfa_avgs.pkl"]
    for file in files_to_truncate:
        if os.path.exists(file):
            with open(file, 'w') as f:
                f.truncate(0)
if clear:
    truncate_files()

def given_at_time(time):
    new_r_ratio = df[time]['New recovered'] / population
    x1 = beta * (new_r_ratio / gamma)
    x2 = 1 - (new_r_ratio / gamma)

    return (x1, x2)

def y_true(time):
    return df[time]['New cases'] / population

def loss(params, w1_run_avg, prev_w2, T_gen, eps_w1, eps_w2):
    w1, w2 = params
    x1, x2 = given_at_time(T_gen)
    y_pred = (w1 * x1) * (x2 - w2)

    penalties = 0
    # Model constraints
    if w1 > max_mnd or w1 < 0 or w2 > 1 or w2 < 0:
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

#  NOTE: Not sure if this is needed here, but it is in the simulated version
# Mean-field approximation loses accuracy for small i
# -> Trim simulation; find where newly infected ratio < smallest_accurate_ratio
start = 0  # Ignore insufficient numbers at the start. Experimentally determined.
# smallest_accurate_ratio = 0.0  # Minimum ratio of infected nodes to total nodes to consider the simulation accurate. Experimentally determined.
# for t in range(start, T):
#     i = df[t]['Active'] / population
#     if i < smallest_accurate_ratio:
#         start = start + 1
#     else:
#         break

# len1 = len(df)  # Length of the data before trimming
df = df[start:]  # Trim the data to start from the first accurate point
# len2 = len(df)  # Length of the data after trimming 
relative_split = split_point - start  # Relative split point in the trimmed data

#  Check subprocess flag. These are used by mfa_drive_compute.py to run the first and second halves of the optimization separately
if sys.argv[1] == "--first_half":
    df = df[:relative_split]  # First half of the data
elif sys.argv[1] == "--second_half":
    df = df[relative_split:]

T = len(df)  # Update T after trimming

lst = [df[t]['Recovered'] / population for t in range(T)]
#  Find largest reasonable difference to assume between recovered ratios at different times
max_diff = max(lst[t+1] - lst[t] for t in range(len(lst) - 1))
print("Max difference in recovered ratios:", max_diff)

# eps_w1: Controls exploration of w1
# alpha: Controls how much w1 is influenced by the run average
# eps_w2: Controls smoothness of w2
def optimize_segment(start=1, end=T, bounds = [(1, max_mnd), (0, 1)],eps_w1=max_mnd, eps_w2=max_diff, alpha=0.7, num_runs=10):

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
                #  Assume we know a good starting place for w2 (recovered portion)
                init_guess = [random.uniform(1, max_mnd), df[0]['Recovered'] / population]
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
                print("Estimated w1:", w1)
                print("Estimated w2:", result.x[1])
                print("x values at T_gen:", given_at_time(T_gen))
                print("Squared error:", result.fun)
                print("\n")

            prev_w2 = w2 # Update prev_w2 for the next iteration

        w1_avg.append(w1_estimates)  # List of w1 estimates for this run
        w2_avg.append(w2_estimates)  # List of w2 estimates for this run

        eps_w1 = temp  # Reset eps_w1 for the next run

    return w1_avg, w2_avg

# Split_point: None means no split, otherwise it is the time to split the optimization
#   This allows us to find missing parameters in the initial stage of SIR, where <k> is indep. from beta parameter,
#     and then optimize the rest of the simulation where <k> is dependent on beta.
def drive_optimizer():
    w1_avg, w2_avg = optimize_segment()

    #  Run-wise average of w1 across runs
    w1_avg = np.mean(w1_avg, axis=0)
    w2_avg = np.mean(w2_avg, axis=0)

    return w1_avg, w2_avg

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

x_ofs = 0
half = 1
if sys.argv[1] == "--second_half":
    half = 2
    x_ofs = split_point - start

sir_infections = [df[t]['Active'] for t in range(len(df))]
sir_infections = np.array(sir_infections) / population  # Normalize by population

# Save to mfa_xy_data.txt
# This is for single runs under a given SIR configuration
# Split: Tuple (split_point, half), where split_point is the time to split the optimization, and half is either 1 or 2
def save_xy_data(w1_avg=None, w2_avg=None, sir_infections=None):
    # Compute y values
    w1_est_y = w1_avg
    w2_est_y = w2_avg

    # Round to 2 decimal places where appropriate
    w1_est_y = [round(y, 2) for y in w1_est_y]
    w2_est_y = [round(y, 2) for y in w2_est_y]

    # If there is a split point, adjust x_vals and y values accordingly
    if split_point is not None:
        if half == 1:
            x_ofs = 0  # Offset for generating x values for plotting
            sir_infections = sir_infections[:split_point]
            w1_est_y = w1_est_y[:split_point-1]
            w2_est_y = w2_est_y[:split_point-1]
        if half == 2:
            x_ofs = split_point  # Offset for generating x values for plotting
            sir_infections = sir_infections[split_point:]
            w1_est_y = w1_est_y[split_point:-1]
            w2_est_y = w2_est_y[split_point:-1]

    with open("experiment_data/mfa_xy_data.txt", "a") as f:
        f.write("==New Sample==\n")
        f.write("SIR Infections (Inset):\n")
        f.write(f"x: {','.join(map(str, range(x_ofs, len(sir_infections) + x_ofs)))}\n")
        f.write(f"y: {','.join(map(str, sir_infections))}\n\n")

        f.write("w1 Estimated:\n")
        f.write(f"x: {','.join(map(str, range(x_ofs, len(w1_est_y) + x_ofs)))}\n")
        f.write(f"y: {','.join(map(str, w1_est_y))}\n\n")

        f.write("w2 Estimated:\n")
        f.write(f"x: {','.join(map(str, range(x_ofs, len(w2_est_y) + x_ofs)))}\n")
        f.write(f"y: {','.join(map(str, w2_est_y))}\n\n")

# This is for averaging multiple runs under a given SIR configuration
def save_results(w1_avg, w2_avg):
    results = {
        "w1_avg": w1_avg,
        "w2_avg": w2_avg,
    }
    # mfa_compute.pkl will be used for computing averages for a given SIR config
    # mfa_xy_data.pkl will be used to hold these averages for plotting
    with open("experiment_data/mfa_compute.pkl", "ab") as f:
        pickle.dump(results, f)

w1_avg, w2_avg = drive_optimizer()
save_xy_data(w1_avg=w1_avg, w2_avg=w2_avg, sir_infections=sir_infections)
save_results(w1_avg=w1_avg, w2_avg=w2_avg)

def plot_SIR(df):
    x = np.arange(len(sir_infections)) + x_ofs

    plt.figure(figsize=(10, 6))
    plt.plot(x, sir_infections, label='SIR Infections', color='blue')
    plt.xlabel('Time (days)')
    plt.ylabel('Infected Ratio')
    plt.title('SIR Infections Over Time')
    plt.legend()
    plt.grid()
    plt.show()

# plot_SIR(df)