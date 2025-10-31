import matplotlib.pyplot as plt
import numpy as np
import re
from itertools import cycle, repeat
import ast
import os
import re

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
show_independence_k = False  # Show that <k> is independent of SIRS parameters when adherence = 0
analyze_quarantine_dynamics = True # Compare ideal vs actual quarantine dynamics from I(t), R(t), <k> estimated
new_function = False  # Experiment from 9/26/25 meeting. Add code at bottom of this file

#----------
#
#  Parse data written by mean_field_approx.py to experiment_data/mfa_xy_data.txt
#
#----------

def parse_sample_data(filename):
    """
    Parse the MFA data file into a list of samples.
    Handles:
      - Numeric datasets (floats)
      - 'w1 all runs' as a list of lists of floats
      - 'Informed Over Time' as a list of lists or nested data
      - 'Number of nodes' as an integer
    """
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
            if line.startswith(("SIR Infections", "Dynamic degree over time", "w1 True", "w1 Estimated",
                                "w1 all runs", "w2 True", "w2 Estimated",
                                "Informed and Infected Over Time")):
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
                elif current_dataset == 'Informed Over Time':
                    # Evaluate as Python object (list of lists)
                    try:
                        y_data = ast.literal_eval(line[2:].strip())
                    except Exception as e:
                        print(f"Error parsing Informed Over Time: {e}")
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
            plt.plot(sample['SIR Infections (Inset)']['x'], sample['SIR Infections (Inset)']['y'],
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
            plt.plot(sample['SIR Infections (Inset)']['x'], sample['SIR Infections (Inset)']['y'],
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
            if 'SIR Infections (Inset)' in sample:
                x = sample['SIR Infections (Inset)']['x']
                y = sample['SIR Infections (Inset)']['y']
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

# Plot w1 Estimated (and its standard deviation), w1 true (Mean Node Degree) for all samples. All in one figure
def plot_independence(samples):
    if not samples:
        print("No samples to plot.")
        return

    all_w1_runs = [sample['w1 all runs']['y'] for sample in samples if 'w1 all runs' in sample]

    plt.figure(figsize=(12, 8))
    colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])

    for i, sample in enumerate(samples):
        color = next(colors)

        # Plot w1 True
        if 'w1 True (Mean Node Degree)' in sample:
            x_true = sample['w1 True (Mean Node Degree)']['x']
            y_true = sample['w1 True (Mean Node Degree)']['y']
            plt.plot(x_true, y_true, label=f'Sample {i+1} w1 True', color=color, linestyle='-')

        # Plot w1 Estimated and std band
        if 'w1 Estimated' in sample:
            x_est = sample['w1 Estimated']['x']
            y_est = sample['w1 Estimated']['y']
            plt.plot(x_est, y_est, label=f'Sample {i+1} w1 Estimated', color=color, linestyle='--')

            # Plot std band if all_w1_runs exists for this sample
            if i < len(all_w1_runs):
                runs = np.array(all_w1_runs[i])  # shape: (num_runs, len(time_points))
                std_dev = np.std(runs, axis=0)
                plt.fill_between(x_est, np.array(y_est) - std_dev, np.array(y_est) + std_dev,
                                 color=color, alpha=0.3)

    plt.xlabel('Time')
    plt.ylabel('Mean Node Degree')
    plt.title('<k> results from optimizer')
    plt.grid(True)
    plt.legend()
    plt.show()

if show_independence_k == True:
    samples = parse_sample_data("experiment_data/mfa_xy_data.txt")
    plot_independence(samples)

if show_adherence == True:
    samples = parse_sample_data("experiment_data/mfa_xy_data.txt")

    for i, sample in enumerate(samples):
        if 'w1 Estimated' in sample and 'w1 True (Mean Node Degree)' in sample:
            w1_estimated = sample['w1 Estimated']['y'][-1]
            w1_true = sample['w1 True (Mean Node Degree)']['y'][-1]
            adherence = np.mean(np.abs(w1_estimated - w1_true) / w1_true)
            print(f"Sample {i}: Adherence = {adherence}")


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

# To run this, the configuration must have a split point
if analyze_quarantine_dynamics == True:
    samples = parse_sample_data("experiment_data/mfa_xy_data.txt")
    
    # k_0 = samples[0]['w1 Estimated']['y'][-1]
    k_0 = samples[0]['w1 True (Mean Node Degree)']['y'][-1]
    # m.n.d. after quarantining is observed
    # k_q = samples[1]['w1 Estimated']['y'][-1]
    k_q = samples[1]['w1 True (Mean Node Degree)']['y'][-1]
    adhering_proportion = samples[1]['Adhering proportion']

    # Informed over time for first sample (no quarantine) will be 0. This variable is for after quarantining begins
    # informed_and_infected = samples[1]['Informed and Infected Over Time']['y']
    # informed_and_infected = [samples[i]['Informed and Infected Over Time']['y'] for i in range(1, len(samples), 2)]
    # informed_and_infected = np.mean(informed_and_infected, axis=0)  # Average over all runs after quarantine
    population = samples[0]['Number of nodes']

    print("k_0:", k_0)
    print("k_q:", k_q)

    # Calculate expected degree over time
    def k_effective_calc(informed_and_infected, current_time, adherence=1.0):
        assert current_time < len(informed_and_infected) + 1, "Current time exceeds data length"

        # Correction term accounts for fact that i' -> i is common with adh. approx 1,
        #   but rare for adh. << 1
        correction_term = adherence + 1
        k_effective = [k_0 * (1 - (correction_term * adherence * (informed_and_infected[time] / population))) for time in range(current_time)]

        # Expected <k> at each time point under full adherence
        expected_k = (1 / len(k_effective)) * sum(k_effective)

        print("Expected <k> under full adherence: ", expected_k)

        return expected_k
    
    # k_q: learned value, a scalar
    # k_0: learned value, a scalar
    # i_prime: average informed and infected proportion over time
    def adherence_calc(k_q, k_0, i_prime):
        adherence = (1 / 2) * (-1 + np.sqrt(1 + 4 * (1 / i_prime) * (1 - (k_q / k_0))))

        return adherence

    #-------------
    #
    #  Show that <k_effective> (our estimate) converges to true <k_q> (from simulation) when adherence = 1
    #
    #-------------

    collection_k_eff_runs = []
    num_simulations = len(samples)
    assert num_simulations % 2 == 0, f"Number of samples should be even for split optimizations. Number: {len(samples)}"
    num_simulations = num_simulations / 2    # Only consider post-quarantining splits
    num_simulations = int(num_simulations)

    # Average dynamic degree over time after quarantining begins
    avg_dynamic_deg = np.mean([samples[i * 2 + 1]['Dynamic degree over time']['y'] for i in range(num_simulations)], axis=0)
    # List of running averages of dynamic degree over time
    running_avg_dynamic_degree = [sum(avg_dynamic_deg[:i+1]) / (i+1) for i in range(len(avg_dynamic_deg))]

    # Iterate over each sample (post-quarantining portions)
    for i in range(num_simulations):
        index = i * 2 + 1   # Only consider post-quarantining splits (odd indexed samples)

        # List of informed and infected over time for this sample
        informed_and_infected = samples[index]['Informed and Infected Over Time']['y']

        # List of <k_q> over time for this sample
        k_q_lst = samples[index]['Dynamic degree over time']['y']

        # k_effective calculated at each time step
        k_eff_over_time = []
        for t in range(1, len(informed_and_infected)):
            # Compute k_effective at time t
            k_eff_t = k_effective_calc(informed_and_infected, t, adherence=adhering_proportion)
            k_eff_over_time.append(k_eff_t)

        collection_k_eff_runs.append(k_eff_over_time)

    # Average k_effective over time, over all runs
    k_eff_curve = np.mean(collection_k_eff_runs, axis=0)

    plt.figure(figsize=(10, 6))
    # Plot <k_effective> over time
    plt.plot(range(1, len(k_eff_curve) + 1), k_eff_curve, label='<k_eff>', color='#1f77b4')
    # Plot average running average of dynamic degree over time
    plt.plot(range(1,len(running_avg_dynamic_degree) + 1), running_avg_dynamic_degree, color="orange")
    # Plot between with standard deviation band
    plt.fill_between(range(1, len(k_eff_curve) + 1),
                     np.array(k_eff_curve) - np.std(collection_k_eff_runs, axis=0),
                     np.array(k_eff_curve) + np.std(collection_k_eff_runs, axis=0),
                     color='#1f77b4', alpha=0.2, label='<k_eff> ± 1 std')
    plt.xlabel('Time')
    plt.ylabel('mean node degree')
    plt.title('<k_effective> vs <k_q> Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

    #-------------
    #
    #  Estimate adherence over time: should converge to true adherence
    #
    #-------------

    # running_avg_dynamic_degree is valid to use here too

    collection_adherence_over_time = []
    adherence_over_time = []

    for i in range(num_simulations):
        running_avg_k_q_lst = []
        index = i * 2 + 1   # Only consider post-quarantining splits (odd indexed samples)

        # List of informed and infected over time for this sample
        informed_and_infected = samples[index]['Informed and Infected Over Time']['y']
        # List of <k_q> over time for this sample
        k_q_lst = samples[index]['Dynamic degree over time']['y']

        assert len(informed_and_infected) == len(k_q_lst), f"Informed and Infected length must match Dynamic Degree length. {len(informed_and_infected)} != {len(k_q_lst)}"

        # Running average adherence over time: should converge to true adherence
        adherence_over_time = []
        for t in range(1, len(informed_and_infected)):
            # running_avg_kq = np.mean(k_q_lst[1:t+1])
            # running_avg_k_q_lst.append(running_avg_kq)

            k_eff_t = k_effective_calc(informed_and_infected, t)
            # adherence_t = abs(running_avg_dynamic_degree[t] - k_eff_t) / abs(k_0 - k_eff_t) if k_0 != k_eff_t else 0
            # adherence_t = (k_eff_t - running_avg_dynamic_degree[t]) / k_eff_t if k_eff_t != 0 else 0
            adherence_t = min((k_0 - running_avg_dynamic_degree[t]) / (k_0 - k_eff_t) if (k_0 - k_eff_t) != 0 else 0, 1)
            # adherence = 
            adherence_over_time.append(adherence_t)
    
        collection_adherence_over_time.append(adherence_over_time)

    adherence_over_time = np.mean(collection_adherence_over_time, axis=0)
    test_val = np.mean(adherence_over_time)
    print("Mean adherence: ", test_val)

    # Plot adherence over time
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(adherence_over_time) + 1), adherence_over_time, label='Adherence over time', color='#1f77b4')
    # Plot between with standard deviation band
    plt.fill_between(range(1, len(adherence_over_time) + 1),
                     np.array(adherence_over_time) - np.std(collection_adherence_over_time, axis=0),
                     np.array(adherence_over_time) + np.std(collection_adherence_over_time, axis=0),
                     color='#1f77b4', alpha=0.2, label='Adherence ± 1 std')
    plt.axhline(y=adhering_proportion, color='r', linestyle='--', label='True Adherence')
    plt.xlabel('Time')
    plt.ylabel('Calculated adherence')
    plt.ylim(-0.9, 1.1)
    plt.title('Adherence Estimation Over Time')
    plt.legend()
    plt.grid(True)
    # plt.show()


#-------------
#
#  Calculate adherence using the approximation: <k_q> = <k_0> * (1 - correction_term * adherence * i')
#    where i' is the informed and infected proportion
#
#-------------

# List of informed and infected for each run
inf_inf_lst = []

for i in range(num_simulations):
    index = i * 2 + 1   # Only consider post-quarantining splits (odd indexed samples)

    # List of informed and infected over time for this sample
    informed_and_infected = samples[index]['Informed and Infected Over Time']['y']
    proportions = [x / population for x in informed_and_infected]
    inf_inf_lst.append(proportions)

inf_inf_lst = np.mean(inf_inf_lst, axis=0)  # Average over all runs after quarantine
i_prime = np.mean(inf_inf_lst)
# i_prime = inf_inf_lst[-1]

adherence_est = adherence_calc(k_q, k_0, i_prime)
print("Estimated adherence (from formula): ", adherence_est)

if new_function == True:
    # This will parse your mean field approximation results from experiment_data/mfa_xy_data.txt
    # To access any of the lists (or lists of lists), you can do something like:
    #     samples[0]['w1 Estimated']['y'] for the FIRST sample's w1 (Mean Node Degree) estimated y values
    #     samples[1]['w1 Estimated']['y'] for the SECOND sample's w1 (Mean Node Degree) estimated y values
    samples = parse_sample_data("experiment_data/mfa_xy_data.txt")

    # Do something with the samples here. Remember, if there's a split point, there will be 2 times the number of configurations you run