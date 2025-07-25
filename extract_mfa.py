import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.cm as cm
import numpy as np
import pickle

plot_opt = False
plot_new = True

#----------
#
#  Parse data written by mean_field_approx.py to experiment_data/mfa_xy_data.txt
#
#----------

# Returns list of the form: <label>: (x_values, y_values)
def parse_opt_results(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    blocks = content.strip().split("=== New Sample ===")[1:]
    samples = []

    for block in blocks:
        lines = [line.strip() for line in block.strip().split('\n') if line.strip()]
        data = {}
        i = 0
        while i < len(lines):
            if i + 2 >= len(lines):
                break  # Not enough lines for x and y
            label = lines[i].rstrip(':')
            x_line = lines[i + 1]
            y_line = lines[i + 2]

            if not x_line.startswith("x:") or not y_line.startswith("y:"):
                i += 1  # Move forward to next line
                continue

            try:
                x = list(map(int, x_line.replace("x: ", "").split(", ")))
                y = list(map(float, y_line.replace("y: ", "").split(", ")))
                data[label] = (x, y)
                i += 3
            except ValueError:
                i += 1  # Skip to next line if something went wrong

        samples.append(data)

    return samples

def plot_group(samples, key_true, key_est, title):
    fig, ax = plt.subplots(figsize=(12, 6))

    num_samples = len(samples)
    cmap = cm.get_cmap('tab10') if num_samples <= 10 else cm.get_cmap('hsv')
    colors = [cmap(i / num_samples) for i in range(num_samples)]

    for i, sample in enumerate(samples):
        x_true, y_true = sample[key_true]
        x_est, y_est = sample[key_est]
        color = colors[i]

        ax.plot(x_true, y_true, marker='o', label=f"Sample {i+1} True", alpha=0.3, color=color)
        ax.plot(x_est, y_est, marker='x', label=f"Sample {i+1} Estimate", color=color)

    ax.set_xlabel('Time')
    ax.set_ylabel(key_true)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    # Add inset SIR plot
    ax_inset = inset_axes(ax, width="15%", height="15%", loc='upper left')
    for i, sample in enumerate(samples):
        x_sir, y_sir = sample["SIR Infections (Inset)"]
        ax_inset.plot(x_sir, y_sir, linestyle='--', color='gray', alpha=0.4)
    ax_inset.set_title("SIR", fontsize=8)
    ax_inset.tick_params(axis='both', labelsize=6)
    ax_inset.grid(True)

    plt.tight_layout()
    plt.show()

if plot_opt == True:
    filepath = "experiment_data/mfa_xy_data.txt"
    samples = parse_opt_results(filepath)

    # Group and plot
    plot_group(samples, 
                key_true="w1 True (Mean Node Degree)", 
                key_est="w1 Estimated", 
                title="True vs Estimated Mean Node Degree")

    plot_group(samples, 
                key_true="w2 True (Recovered Fraction)", 
                key_est="w2 Estimated", 
                title="True vs Estimated Recovered Fraction")
    

def parse_avgs(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    return data

def find_adherence(k0, k1):
    adherence_proportion = (k0 - k1) / k0 if k0 != 0 else 0

    # Error handling: if adherence proportion is negative (data was likely in wrong order in file), reverse the numbers
    if adherence_proportion < 0:
        adherence_proportion = (k1 - k0) / k1 if k1 != 0 else 0

    return adherence_proportion

if plot_new == True:

#-----------
#
#  Parse data written by mfa_drive_compute.py to experiment_data/mfa_avgs.pkl
#  Data is saved in the form: (w1_avg, w2_avg_vec) for each configuration.
#
# Approximate quarantine adherence proportion as: (<k> - k) / <k>, where <k> is mean node degree from optimization w/ no quarantine,
#    k is mean node degree from optimization w/ quarantine (this should be the data from experiment_data/mfa_xy_data)
#
# ----------

    samples = parse_avgs("experiment_data/mfa_avgs.pkl")

    # Print hypothesized adherence proportion
    k_sample0 = samples[0][0]
    k_sample1 = samples[1][0]

    print("Mean Node Degrees: ", k_sample0, " ", k_sample1)

    adherence_proportion = find_adherence(k_sample0, k_sample1)
    print("Adherence Proportion:", adherence_proportion)

    # avgs = parse_avgs("experiment_data/mfa_avgs.pkl")

    # # Data at this point should be a list of: (scalar, vector), representing 

    
