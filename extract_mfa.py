import matplotlib.pyplot as plt
import numpy as np
import re
from itertools import cycle, repeat
import ast
import os
import re
import scipy.optimize as opt
from scipy.optimize import minimize
from sympy import symbols, Function, diff, sin, exp, pprint, Matrix
from scipy.special import logit, expit
from numdifftools import Hessian
import random
from collections import defaultdict

# Experimental
# import networkx as nx
# import IM
# import random

#-----------
#
#  File purpose: Extract, plot, and analyze data for optimization results and adherence calculation
#
#-----------

# plot_opt = True -> Plot optimization results for 1 sample (no splitting the optimization) 
plot_opt = False
plot_opt_real_world = False  # Plot optimization results for 1 sample, but with real-world data (Ex. Covid csv file)
show_adherence = False
plot_split_opt = False  #  Plot average results for 1 configuration, but with splitting the optimization
plot_many_non_split = False  # Plot many <k> estimates without splitting
show_independence_k = True  # Show that <k> is independent of SIRS parameters when adherence = 0
plot_inf_vs_infm = False  # Plot informed, infected, informed and infected for 5 groups of post-quarantine samples
analyze_quarantine_dynamics = False # Compute adherence from quarantine dynamics


#----------
#
#  Parse data written by mean_field_approx.py to experiment_data/mfa_xy_data.txt
#
#----------

def parse_sample_data(filename):
    # Parse the MFA data file into a list of samples.
    samples = []
    current_sample = {}
    current_dataset = None
    x_data = []
    y_data = []

    def parse_w1_all_runs_y_line(line):
        y_str = line[2:].strip()
        # Remove np.float64(...) wrappers
        y_str_clean = re.sub(r'np\.float64\(([^)]+)\)', r'\1', y_str)
        try:
            return ast.literal_eval(y_str_clean)
        except Exception as e:
            print(f"Error parsing w1 all runs y line: {e}")
            return []

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()

            # Start of new sample
            if line == "==New Sample==":
                if current_dataset and x_data:
                    current_sample[current_dataset] = {'x': x_data, 'y': y_data}
                    x_data, y_data = [], []
                if current_sample:
                    samples.append(current_sample)
                    current_sample = {}
                current_dataset = None
                continue

            # Detect Number of nodes line
            if line.startswith("Number of nodes:"):
                try:
                    current_sample['Number of nodes'] = int(line.split(":")[1].strip())
                except Exception as e:
                    print(f"Error parsing Number of nodes: {e}")
                    current_sample['Number of nodes'] = None
                continue

            # Detect true adherence proportion
            if line.startswith("Adhering proportion:"):
                try:
                    current_sample['Adhering proportion'] = float(line.split(":")[1].strip())
                except Exception as e:
                    print(f"Error parsing Adhering proportion: {e}")
                    current_sample['Adhering proportion'] = None
                continue

            # Detect dataset name lines
            if line.startswith(("SIR Infections", "Dynamic degree", "w1 True", "w1 Estimated",
                                "w1 all runs", "w2 True", "w2 Estimated",
                                "Informed and Infected", "Given Newly Infected Ratio", "Informed")):
                if current_dataset and x_data:
                    current_sample[current_dataset] = {'x': x_data, 'y': y_data}
                    x_data, y_data = [], []
                current_dataset = line.split(':')[0].strip()
                continue

            # Parse x-data
            if line.startswith("x:"):
                try:
                    x_data = [float(x) for x in line[2:].split(',') if x]
                except ValueError as e:
                    print(f"Error parsing x-data: {e}")
                    x_data = []

            # Parse y-data
            elif line.startswith("y:"):
                if current_dataset == 'w1 all runs':
                    y_data = parse_w1_all_runs_y_line(line)
                elif current_dataset == 'Informed':
                    # Evaluate as Python object (list of lists)
                    try:
                        y_data = ast.literal_eval(line[2:].strip())
                    except Exception as e:
                        print(f"Error parsing Informed: {e}")
                        y_data = []
                else:
                    raw = line[2:].strip()
                    try:
                        # If starts with '[', evaluate as a Python object (nested lists)
                        if raw.startswith('['):
                            y_data = ast.literal_eval(raw)
                        else:
                            y_data = [float(y) for y in raw.split(',') if y]
                    except Exception as e:
                        print(f"Error parsing y-data: {e}")
                        y_data = []

        # Save last dataset of the last sample
        if current_dataset and x_data:
            current_sample[current_dataset] = {'x': x_data, 'y': y_data}
        if current_sample:
            samples.append(current_sample)

    return samples

def plot_sample(sample, sample_num):
    # --- Plot Mean Node Degree (w1) ---
    plt.figure(figsize=(10, 6))

    if 'w1 True (Mean Node Degree)' in sample:
        plt.plot(
            sample['w1 True (Mean Node Degree)']['x'],
            sample['w1 True (Mean Node Degree)']['y'],
            label='Ground truth',
            color='#1f77b4',
            linestyle='-'
        )

    if 'w1 Estimated' in sample:
        plt.plot(
            sample['w1 Estimated']['x'],
            sample['w1 Estimated']['y'],
            label='Estimated',
            color='#ff7f0e',
            linestyle='--'
        )

    # Add std band for w1 (from all runs)
    if 'w1 all runs' in sample:
        x = np.array(sample['w1 all runs']['x'])
        y_runs = np.array(sample['w1 all runs']['y'])  # shape: (num_runs, time_points)
        mean_y = np.mean(y_runs, axis=0)
        std_y = np.std(y_runs, axis=0)

        plt.fill_between(
            x, mean_y - std_y, mean_y + std_y,
            color='#ff7f0e', alpha=0.2, label='Estimate ± 1 std'
        )

    plt.xlabel('Time (in days)')
    plt.ylabel('M.N.D. estimated')
    plt.title('Mean node degree estimated vs ground truth')
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Plot Recovered Fraction (w2) ---
    plt.figure(figsize=(10, 6))

    if 'w2 True (Recovered Fraction)' in sample:
        plt.plot(
            sample['w2 True (Recovered Fraction)']['x'],
            sample['w2 True (Recovered Fraction)']['y'],
            label='w2 True (Recovered Fraction)',
            color='#2ca02c',
            linestyle='-'
        )

    if 'w2 Estimated' in sample:
        plt.plot(
            sample['w2 Estimated']['x'],
            sample['w2 Estimated']['y'],
            label='w2 Estimated',
            color='#d62728',
            linestyle='--'
        )

    # Add std band for w2 (from all runs)
    if 'w2 all runs' in sample:
        x = np.array(sample['w2 all runs']['x'])
        y_runs = np.array(sample['w2 all runs']['y'])
        mean_y = np.mean(y_runs, axis=0)
        std_y = np.std(y_runs, axis=0)

        plt.fill_between(
            x, mean_y - std_y, mean_y + std_y,
            color='#d62728', alpha=0.2, label='w2 ± 1 std'
        )

    plt.xlabel('Time')
    plt.ylabel('Recovered Fraction')
    plt.title(f'Sample {sample_num}: w2 True vs Estimated')
    plt.legend()
    plt.grid(True)
    plt.show()

if plot_opt == True:
    # Main execution
    filename = 'experiment_data/mfa_xy_data.txt'
    samples = parse_sample_data(filename)

    # Plot for Sample 1
    plot_sample(samples[0], 1)

# Plot description: mfa_xy should have many of 1 config, many of another (split = False in mean_field_approx.py), 
#   and this will plot the SIR curves and w1 True vs Estimated for each half (set of samples).
def plot_halves(samples):
    from itertools import cycle
    import matplotlib.pyplot as plt

    n_samples = len(samples)
    mid_point = (n_samples + 1) // 2
    first_half = samples[:mid_point]
    second_half = samples[mid_point:]

    colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])

    # Plot first half
    if first_half:
        # SIR infections
        plt.figure(figsize=(10, 6))
        for i, sample in enumerate(first_half, 1):
            color = next(colors)
            plt.plot(sample['SIR Infections']['x'], sample['SIR Infections']['y'],
                     label=f'Sample {i}', color=color)
        plt.xlabel('Time')
        plt.ylabel('SIR Infections')
        plt.title('First Half: SIR Infections')
        plt.grid(True)
        plt.legend()
        plt.show()

        colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])

        # w1 True and Estimated
        plt.figure(figsize=(10, 6))
        for i, sample in enumerate(first_half, 1):
            color = next(colors)
            if 'w1 True (Mean Node Degree)' in sample:
                plt.plot(sample['w1 True (Mean Node Degree)']['x'], sample['w1 True (Mean Node Degree)']['y'],
                         label=f'Sample {i} w1 True', color=color, linestyle='-')
            plt.plot(sample['w1 Estimated']['x'], sample['w1 Estimated']['y'],
                     label=f'Sample {i} w1 Estimated', color=color, linestyle='--')
        plt.xlabel('Time')
        plt.ylabel('Mean Node Degree')
        plt.title('First Half: w1 True vs Estimated')
        plt.grid(True)
        plt.legend()
        plt.show()

    # Plot second half
    colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])

    if second_half:
        # SIR infections
        plt.figure(figsize=(10, 6))
        for i, sample in enumerate(second_half, mid_point + 1):
            color = next(colors)
            plt.plot(sample['SIR Infections']['x'], sample['SIR Infections']['y'],
                     label=f'Sample {i}', color=color)
        plt.xlabel('Time')
        plt.ylabel('SIR Infections')
        plt.title('Second Half: SIR Infections')
        plt.grid(True)
        plt.legend()
        plt.show()

        colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])

        # w1 True and Estimated
        plt.figure(figsize=(10, 6))
        for i, sample in enumerate(second_half, mid_point + 1):
            color = next(colors)
            if 'w1 True (Mean Node Degree)' in sample:
                plt.plot(sample['w1 True (Mean Node Degree)']['x'], sample['w1 True (Mean Node Degree)']['y'],
                         label=f'Sample {i} w1 True', color=color, linestyle='-')
            plt.plot(sample['w1 Estimated']['x'], sample['w1 Estimated']['y'],
                     label=f'Sample {i} w1 Estimated', color=color, linestyle='--')
        plt.xlabel('Time')
        plt.ylabel('Mean Node Degree')
        plt.title('Second Half: w1 True vs Estimated')
        plt.grid(True)
        plt.legend()
        plt.show()

def find_adherence(k0, k1):
    adherence_proportion = (k0 - k1) / k0 if k0 != 0 else 0

    # Error handling: if adherence proportion is negative (data was likely in wrong order in file), reverse the numbers
    if adherence_proportion < 0:
        adherence_proportion = (k1 - k0) / k1 if k1 != 0 else 0

    return adherence_proportion

if plot_opt_real_world == True:
    # Parse the average results from the file
    samples = parse_sample_data("experiment_data/mfa_xy_data.txt")

    plot_halves(samples)

    #  Extract last w1 value from each sample
    k1 = samples[0]['w1 Estimated']['y'][-1]
    k2 = samples[1]['w1 Estimated']['y'][-1]

    adherence_proportion = find_adherence(k1, k2)
    print("Adherence Proportion:", adherence_proportion)

#  NOTE: Function is still very much a work in progress
#  NOTE: Not using this currently
def plot_unintertwined(samples):
    if not samples:
        print("No samples to plot.")
        return

    group_a = samples[0::2]  # A, A, A...
    group_b = samples[1::2]  # B, B, B...

    def safe_plot_sir(group, title):
        if not group:
            return
        colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        plt.figure(figsize=(10, 6))
        for i, sample in enumerate(group, 1):
            color = next(colors)
            if 'SIR Infections' in sample:
                x = sample['SIR Infections']['x']
                y = sample['SIR Infections']['y']
                plt.plot(x, y, label=f'{title} Sample {i}', color=color)
            else:
                # try fallback key name if present
                for key in sample:
                    if key.lower().startswith('sir infections'):
                        x = sample[key]['x']; y = sample[key]['y']
                        plt.plot(x, y, label=f'{title} Sample {i}', color=color)
                        break
        plt.xlabel('Time')
        plt.ylabel('SIR Infections')
        plt.title(f'{title}: SIR Infections')
        plt.grid(True)
        plt.legend()
        plt.show()

    def safe_plot_w1(group, title):
        if not group:
            return
        colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        plt.figure(figsize=(10, 6))
        for i, sample in enumerate(group, 1):
            color = next(colors)
            # w1 True (Mean Node Degree) may be missing; guard accordingly
            if 'w1 True (Mean Node Degree)' in sample:
                x_true = sample['w1 True (Mean Node Degree)']['x']
                y_true = sample['w1 True (Mean Node Degree)']['y']
                plt.plot(x_true, y_true, label=f'{title} Sample {i} w1 True', color=color, linestyle='-')
            elif 'w1 True' in sample:  # fallback shorter name
                x_true = sample['w1 True']['x']
                y_true = sample['w1 True']['y']
                plt.plot(x_true, y_true, label=f'{title} Sample {i} w1 True', color=color, linestyle='-')

            # w1 Estimated (expected to exist but guard anyway)
            if 'w1 Estimated' in sample:
                x_est = sample['w1 Estimated']['x']
                y_est = sample['w1 Estimated']['y']
                plt.plot(x_est, y_est, label=f'{title} Sample {i} w1 Estimated', color=color, linestyle='--')
            else:
                # try to find any key that contains 'w1' and 'Estimated'
                for k in sample:
                    if 'w1' in k.lower() and 'estimate' in k.lower():
                        x_est = sample[k]['x']; y_est = sample[k]['y']
                        plt.plot(x_est, y_est, label=f'{title} Sample {i} w1 Estimated', color=color, linestyle='--')
                        break
        plt.xlabel('Time')
        plt.ylabel('Mean Node Degree')
        plt.title(f'{title}: w1 True vs Estimated')
        plt.grid(True)
        plt.legend()
        plt.show()

    # Plot group A then group B
    safe_plot_sir(group_a, 'Group A (even indices)')
    safe_plot_w1(group_a, 'Group A (even indices)')

    safe_plot_sir(group_b, 'Group B (odd indices)')
    safe_plot_w1(group_b, 'Group B (odd indices)')

if plot_split_opt == True:
    # Parse the average results from the file
    samples = parse_sample_data("experiment_data/mfa_xy_data.txt")

    # Plot for each group
    plot_unintertwined(samples)

# Plot w1_all_runs for specified half
# even_or_odd = 0: only plot first half (k_0) estimates
# even_or_odd = 1: only plot second half (k_q) estimates
def plot_independence(samples, even_or_odd=0):
    # Extract w1_all_runs for specified half
    w1_runs = []
    for i, sample in enumerate(samples):
        if 'w1 all runs' in sample:
            if even_or_odd == 0 and i % 2 == 0:
                w1_runs.extend(sample['w1 all runs']['y'])
            elif even_or_odd == 1 and i % 2 == 1:
                w1_runs.extend(sample['w1 all runs']['y'])
    
    # Convert to numpy array for easier manipulation
    w1_array = np.array(w1_runs)  # shape: (num_runs, time_points)
    mean_w1 = np.mean(w1_array, axis=0)
    std_w1 = np.std(w1_array, axis=0)
    time_points = np.arange(w1_array.shape[1])

    plt.figure(figsize=(10, 6))
    plt.plot(time_points, mean_w1, label='Mean M.N.D. Estimate', color='#ff7f0e')
    plt.fill_between(time_points, mean_w1 - std_w1, mean_w1 + std_w1,
                        color='#ff7f0e', alpha=0.2, label='Estimate ± 1 std')
    # Plot ground truth
    if even_or_odd == 0:
        plt.hlines(y=samples[0]['w1 True (Mean Node Degree)']['y'][-1],
                   xmin=0, xmax=time_points[-1],
                   colors='blue', linestyles='--', label='Ground Truth M.N.D.')
    else:
        plt.hlines(y=samples[1]['w1 True (Mean Node Degree)']['y'][-1],
                   xmin=0, xmax=time_points[-1],
                   colors='blue', linestyles='--', label='Ground Truth M.N.D.')
    plt.xlabel('Time')
    plt.ylabel('Mean Node Degree Estimate')
    plt.title('M.N.D. Estimates Across Many Runs')
    plt.legend()
    plt.grid(True)
    plt.show()
    
if show_independence_k == True:
    samples = parse_sample_data("experiment_data/mfa_xy_data.txt")
    plot_independence(samples, even_or_odd=0)

if show_adherence == True:
    samples = parse_sample_data("experiment_data/mfa_xy_data.txt")

    for i, sample in enumerate(samples):
        if 'w1 Estimated' in sample and 'w1 True (Mean Node Degree)' in sample:
            w1_estimated = sample['w1 Estimated']['y'][-1]
            w1_true = sample['w1 True (Mean Node Degree)']['y'][-1]
            adherence = np.mean(np.abs(w1_estimated - w1_true) / w1_true)
            print(f"Sample {i}: Adherence = {adherence}")

if plot_inf_vs_infm == True:
    samples = parse_sample_data("experiment_data/mfa_xy_data.txt")
    odd_samples = [samples[i] for i in range(len(samples)) if i % 2 == 1]

    # Truncate to multiple of 5
    num_full_groups = len(odd_samples) // 5
    odd_samples = odd_samples[:num_full_groups * 5]

    if len(odd_samples) == 0:
        print("No post-quarantine samples found.")
    else:
        n = odd_samples[0]['Number of nodes']

        # Group by adherence (0.2, 0.4, 0.6, 0.8, 1.0)
        grouped = np.array(odd_samples).reshape(5, -1)

        adherence_levels = [0.2, 0.4, 0.6, 0.8, 1.0]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        # Line styles for the three curve types
        styles = {
            'Infected':            ('solid',  2.5),
            'Informed & Infected': ('dashed', 2.2),
            'Informed':            ('dotted', 2.5)
        }

        plt.figure(figsize=(14, 8))

        # We'll build the legend entries in exact order
        legend_handles = []
        legend_labels  = []

        for idx, (a, color) in enumerate(zip(adherence_levels, colors)):
            group = grouped[idx]

            # Collect data
            inf_data = [s['SIR Infections']['y'] for s in group if 'SIR Infections' in s]
            ii_data  = [s['Informed and Infected']['y'] for s in group if 'Informed and Infected' in s]
            i_data   = [s['Informed']['y'] for s in group if 'Informed' in s]

            t = np.arange(200)  # adjust if needed

            # Plot and create legend entry for each curve type
            for name, data_list, key in [
                ('Infected',            inf_data, 'SIR Infections'),
                ('Informed & Infected', ii_data,  'Informed and Infected'),
                ('Informed',            i_data,   'Informed')
            ]:
                if data_list:
                    mean_curve = np.mean(data_list, axis=0) / n
                    linestyle, lw = styles[name]
                    line = plt.plot(t[:len(mean_curve)], mean_curve,
                                    color=color, linestyle=linestyle, linewidth=lw,
                                    label=f'{name} (a = {a})')[0]

                    # Add to legend (exactly once per curve+adherence)
                    legend_handles.append(line)
                    legend_labels.append(f'{name} (a = {a})')

        plt.xlabel('Time Step', fontsize=13)
        plt.ylabel('Fraction of Population', fontsize=13)
        plt.title('Post-Quarantine Dynamics by Adherence Level', fontsize=15)
        plt.grid(True, alpha=0.3)
        plt.xlim(0, None)
        plt.ylim(0, 1.02)

        # Clean 15-item legend, 3 columns
        plt.legend(handles=legend_handles, labels=legend_labels,
                   ncol=3, fontsize=10.8,
                   loc='lower center', bbox_to_anchor=(0.5, -0.20),
                   frameon=True, fancybox=True, shadow=True)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.24)  # extra space for legend
        plt.show()

#-------------
#
#  Adherence Calculations:
#  We will use daily infected and recovered numbers, 
#  along with our MFA optimizer for <k> to compute difference
#  between ideal and actual quarantine dynamics
#
#-------------

def split_triples_from_file(filename="experiment_data/infected_recovered.txt"):
    # 1. Read all lines from file
    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # 2. Extract split point and total nodes
    split_point = int(lines[0].split(":")[1].strip()) - 1
    total_nodes = int(lines[1].split(":")[1].strip())
    
    # 3. Parse triples (skip header at index 1)
    triples = []
    for line in lines[3:]:
        day_str, inf_str, rec_str = line.split(",")
        triples.append((int(day_str), int(inf_str), int(rec_str)))
    
    # 4. Split into two lists
    before = [t for t in triples if t[0] < split_point]
    after  = [t for t in triples if t[0] >= split_point]
    
    return split_point, total_nodes, before, after

if analyze_quarantine_dynamics == True:
    # To run this, the configuration must have a split point
    # if analyze_quarantine_dynamics == True:
    samples = parse_sample_data("experiment_data/mfa_xy_data.txt")

    # k_0 = samples[0]['w1 Estimated']['y'][-1]
    k_0 = samples[0]['w1 True (Mean Node Degree)']['y'][-1]
    # m.n.d. after quarantining is observed
    # k_q = samples[1]['w1 Estimated']['y'][-1]
    k_q = samples[1]['w1 True (Mean Node Degree)']['y'][-1]
    split_point = int(samples[1]['w1 True (Mean Node Degree)']['x'][0])  # First time point of second sample
    adhering_proportion = samples[1]['Adhering proportion']

    population = samples[0]['Number of nodes']

    print("k_0:", k_0)
    print("k_q:", k_q)

    collection_k_eff_runs = []
    num_simulations = len(samples)
    assert num_simulations % 2 == 0, f"Number of samples should be even for split optimizations. Number: {len(samples)}"
    num_simulations = num_simulations / 2    # Only consider post-quarantining splits
    num_simulations = int(num_simulations)

    # List of informed and infected for each run
    inf_inf_lst = []

    for i in range(num_simulations):
        index = i * 2 + 1   # Only consider post-quarantining splits (odd indexed samples)

        # List of informed and infected over time for this sample
        informed_and_infected = samples[index]['Informed and Infected']['y']
        proportions = [x / population for x in informed_and_infected]
        inf_inf_lst.append(proportions)

    i_prime = np.mean(inf_inf_lst, axis=0)  # Average over all runs after quarantine

    # Calculate expected degree over time
    def k_expected(informed_and_infected, current_time, adherence=1.0):
        assert current_time < len(informed_and_infected) + 1, "Current time exceeds data length"

        # Observation:
        # High adherence -> True removal ~ k_0 * (1 - adherence * i')
        # Low adherence -> True removal ~ k_0 * (1 - 2 * adherence * i')
        k_expected_t = [k_0 * (1 - (2 * adherence * (informed_and_infected[time] / population))) for time in range(current_time)]

        # Expected <k> at each time point under full adherence
        expected_k = (1 / len(k_expected_t)) * sum(k_expected_t)

        return expected_k
    
    # Calculate expected degree under full quarantine
    def k_effective(infected):
        # Effective degree based on infected proportion
        k_eff_t = [k_0 * (1 - 2 * (infected[time] / population)) for time in range(len(infected))]

        k_eff = np.mean(k_eff_t)

        return k_eff

    #--------------
    #
    #  Naive estimate of adherence
    #  Show that only knowing <k_0>, <k_q>, k_eff is still insufficient in determining adherence
    #
    #-------------

    # We only want odd indexed samples (post-quarantine)
    odd_samples = [samples[i] for i in range(len(samples)) if i % 2 == 1]

    avg_inf = []
    for i, sample in enumerate(odd_samples):
        if 'SIR Infections' in sample:
            infected_over_time = sample['SIR Infections']['y']
            avg_inf.append(infected_over_time)
    inf = np.mean(avg_inf, axis=0) / population

    k_eff = k_effective(inf)
    adherence_naive = 1 - (k_eff - k_q) / (k_eff) if k_eff != 0 else 0

    print("Naive approximation for adherence:", adherence_naive)

    k_q_lst_avg = []
    for i in range(num_simulations):
        index = i * 2 + 1   # Only consider post-quarantining splits (odd indexed samples)

        # List of <k_q> over time for this sample
        k_q_lst = samples[index]['Dynamic degree']['y']
        k_q_lst_avg.append(k_q_lst)

    k_q_true = np.mean(k_q_lst_avg, axis=0)  # Average over all runs AFTER quarantine

    #-------------
    #
    #  Approximation: <k_q> ~ <k_0> * (1 - S * adherence * i'), where S is a scale factor to optimize
    #  Jointly optimize S and adherence
    #
    #-------------

    # Define loss (MSE) for given parameters
    def mse_loss(params, i_prime_cur):
        S, adherence_val = params
        # i_prime: a vector of informed and infected proportions over time after quarantine
        k_q_est = [k_0 * (1 - S * adherence_val * i_prime_cur[t]) for t in range(len(i_prime))]
        mse = np.mean((k_q_true - k_q_est) ** 2)

        return mse

    # Define parameter ranges
    S_RANGE = (1.0, 2.0)
    ADHERENCE_RANGE = (0.0, 1.0)

    # Keep track of best parameters across runs
    best_S_lst = []
    best_adh_lst = []

    for i in range(num_simulations):
        res = minimize(lambda params: mse_loss(params, inf_inf_lst[i]),
            # Initial guess
            x0=[1.5, 0.5],
            bounds=[S_RANGE, ADHERENCE_RANGE],
            method='L-BFGS-B',
            options={'gtol': 1e-6}
        )

        best_params = res.x
        best_S, best_adherence = best_params

        best_S_lst.append(best_S)
        best_adh_lst.append(best_adherence)

    # Average best parameters
    best_S = np.mean(best_S_lst)
    best_adherence = np.mean(best_adh_lst)

    print("Best S (avg over runs):", best_S)
    print("Best Adherence (avg over runs):", best_adherence)

    # Standard deviations
    S_std = np.std(best_S_lst)
    adherence_std = np.std(best_adh_lst)

    # Calculate estimated k_q using average best parameters
    k_q_est_best = [k_0 * (1 - best_S * best_adherence * i_prime[t]) for t in range(len(i_prime))]

    # Obtain std for estimated k_q
    k_q_est_runs = []
    for i in range(num_simulations):
        k_q_est_run = [k_0 * (1 - best_S_lst[i] * best_adh_lst[i] * inf_inf_lst[i][t]) for t in range(len(i_prime))]
        k_q_est_runs.append(k_q_est_run)

    k_q_est_runs = np.array(k_q_est_runs)
    k_q_est_mean = np.mean(k_q_est_runs, axis=0)
    k_q_est_std = np.std(k_q_est_runs, axis=0)
    
    #--------------
    #
    #  Plot <k_q>_opt vs <k_q>_true with std bands
    #
    #--------------

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(k_q_true)), k_q_true, label='True <$k_q$>', color='#1f77b4')
    plt.plot(range(len(k_q_est_best)), k_q_est_best, label='Estimated <$k_q$> (best params)', color='#ff7f0e', linestyle='--')
    plt.fill_between(range(len(k_q_est_mean)), k_q_est_mean - k_q_est_std, k_q_est_mean + k_q_est_std,
                        color='#ff7f0e', alpha=0.2, label='Estimate ± 1 std (all runs)')
    plt.xlabel('Time (in days)')
    plt.ylabel('<$k_q$>')
    plt.title('<$k_q$> Parameter Optimization vs Ground Truth')
    plt.legend()
    plt.grid(True)
    # plt.show()

    plt.figure(figsize=(6, 6))

    #----------------------
    #
    #  Plot true vs estimated adherence as stacked bar chart
    #
    #----------------------

    labels               = ['Adherence Proportion']
    true_adherence       = [adhering_proportion]      
    estimated_adherence  = [best_adherence]       
    error                = [adherence_std]        

    # Bottom bar = true value (full height)
    bars1 = plt.bar(labels, true_adherence, 
                    color='#2ca02c', label='True Adherence', 
                    edgecolor='black', linewidth=1.2)

    # Top bar = difference between estimate and true (stacked)
    difference = np.array(estimated_adherence) - np.array(true_adherence)
    bars2 = plt.bar(labels, difference, bottom=true_adherence,
                    yerr=error, error_kw={'capsize': 10, 'capthick': 2, 'ecolor': 'black'},
                    color='#d62728', label='Estimated Adherence', alpha=0.85,
                    edgecolor='black', linewidth=1.2)

    plt.text(0, estimated_adherence[0] + 0.02, f'{estimated_adherence[0]:.3f} ± {error[0]:.3f}',
            ha='center', va='bottom', fontweight='bold', color='#d62728')

    plt.ylabel('Adherence Proportion', fontsize=12)
    plt.title('True vs Estimated Adherence Proportion', fontsize=14)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2)

    # Remove x-tick labels since there's only one category
    plt.xticks([])

    plt.tight_layout()
    plt.show()