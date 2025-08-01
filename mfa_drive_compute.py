# Use this file to run the mean field approximation several times and average the results

import pickle
import os
import subprocess
import sys
import numpy as np

#----------
#
#  File architecture: mfa_compute temporarily holds data from runs. Extracts averages and saves them to mfa_avgs.pkl. 
#    Cleared when a new SIR configuration is created.
#
#  Our averaging abstracts here away from the true values, so we're going off of just the optimizer's outputs.
#
#----------

repeat = 1

# We now have data in mfa_compute.pkl. We will perform averaging computations, then load the result into mfa_xy_data.pkl
def average_saved_results():
    all_results = []

    # Read all dictionaries from the appended pkl file
    with open("experiment_data/mfa_compute.pkl", "rb") as f:
        while True:
            try:
                result = pickle.load(f)
                all_results.append(result)
            except EOFError:
                break

    if not all_results:
        print("No data found in mfa_compute.pkl")
        return
    
    # Step 1: Extract the optimizer's results
    w1_last_values = [d['w1_avg'][-1] for d in all_results]  # Turn into scalars (recall that <k> is constant)
    w2 = [d['w2_avg'] for d in all_results]  # Stays as vectors (we want r as a function of time)

    # Trim every vector of w2 to be the length of the shortest one (so that we can take the average)
    min_length = min(len(w2_item) for w2_item in w2)
    w2 = [w2_item[:min_length] for w2_item in w2]

    # Step 2: Compute averages
    w1_last_avg = np.mean(w1_last_values)
    w2_avg = np.mean(w2, axis=0)

    return w1_last_avg, w2_avg

# Truncate the compute file (mfa_compute.pkl) upon each new SIR configuration run group
def truncate_compute_file():
    if os.path.exists("experiment_data/mfa_compute.pkl"):
        with open("experiment_data/mfa_compute.pkl", 'wb') as f:
            pass  # This will truncate the file

# Clear compute file for next SIR config averaging
truncate_compute_file()

for _ in range(repeat):
    subprocess.run([sys.executable, "mean_field_approx.py", "--subprocess"])

load_path = "experiment_data/mfa_avgs.pkl"

# Step 1: Load existing data safely
if os.path.exists(load_path) and os.path.getsize(load_path) > 0:
    with open(load_path, "rb") as f:
        try:
            data = pickle.load(f)  # Assumes the file contains a list
        except EOFError:
            data = []
else:
    data = []

# Step 2: Append the new result
run_avgs = average_saved_results()

data.append(run_avgs)

# Step 3: Save it back (overwrite with updated list)
with open(load_path, "wb") as f:
    pickle.dump(data, f)