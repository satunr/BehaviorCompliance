import matplotlib.pyplot as plt
import numpy as np
import re
from itertools import cycle, repeat
import ast
import os
import re

# plot_opt = True -> Plot optimization results for 1 sample (no splitting the optimization) 
plot_opt = True
plot_opt_real_world = False  # Plot optimization results for 1 sample, but with real-world data (Ex. Covid csv file)
show_adherence = False
plot_split_opt = False  #  Plot average results for 1 configuration, but with splitting the optimization
plot_many_non_split = False  # Plot many <k> estimates without splitting
show_independence_k = False  # Show that <k> is independent of SIRS parameters when adherence = 0
analyze_quarantine_dynamics = False # Compare ideal vs actual quarantine dynamics from I(t), R(t), <k> estimated

#----------
#
#  Parse data written by mean_field_approx.py to experiment_data/mfa_xy_data.txt
#
#----------

def parse_sample_data(filename):
    samples = []
    current_sample = {}
    current_dataset = None
    x_data = []
    y_data = []
    
    def parse_w1_all_runs_y_line(line):
        # Remove "y:" prefix
        y_str = line[2:].strip()

        # Remove np.float64(...) wrappers
        y_str_clean = re.sub(r'np\.float64\(([^)]+)\)', r'\1', y_str)

        # Now safely evaluate the list of lists string
        try:
            parsed = ast.literal_eval(y_str_clean)
            # parsed is a list of lists of floats
            return parsed
        except Exception as e:
            print(f"Error parsing w1 all runs y line: {e}")
            return []

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line == "==New Sample==":
                if current_dataset and x_data:
                    current_sample[current_dataset] = {'x': x_data, 'y': y_data}
                    x_data = []
                    y_data = []
                if current_sample:
                    samples.append(current_sample)
                    current_sample = {}
                current_dataset = None
                continue
            
            # Dataset names detection
            if line.startswith("SIR Infections") or line.startswith("w1 True") or line.startswith("w1 Estimated") or line.startswith('w1 all runs') or line.startswith("w2 True") or line.startswith("w2 Estimated"):
                if current_dataset and x_data:
                    current_sample[current_dataset] = {'x': x_data, 'y': y_data}
                    x_data = []
                    y_data = []
                current_dataset = line.split(':')[0].strip()
                continue
            
            if line.startswith("x:"):
                x_data = [float(x) for x in line[2:].split(',')]
            
            elif line.startswith("y:"):
                # Special case for w1 all runs
                if current_dataset == 'w1 all runs':
                    y_data = parse_w1_all_runs_y_line(line)
                else:
                    y_data = [float(y) for y in line[2:].split(',')]
        
        # Save last dataset of last sample
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

if analyze_quarantine_dynamics == True:
    num_runs = 5  # Number of runs for SIRS so we can average results for daily infections, daily recoveries
    samples = parse_sample_data("experiment_data/mfa_xy_data.txt")
    
    # m.n.d. before quarantining is observed
    # We say that our <k>'s are determined using the first run of SIRS
    #   We can't average this, as the k vector may be of different lengths for different runs 
    #     (recall sufficient I(t) to continue optimizer)
    # k_0 = samples[0]['w1 Estimated']['y'][-1]
    k_0 = samples[0]['w1 True (Mean Node Degree)']['y'][-1]
    # m.n.d. after quarantining is observed
    # k_q = samples[1]['w1 Estimated']['y'][-1]
    k_q = samples[1]['w1 True (Mean Node Degree)']['y'][-1]

    # Clear infected_recovered.txt
    with open("experiment_data/infected_recovered.txt", "w") as f:
        f.truncate(0)

    # Recall that the true split_point will be split_point + 1
    split_point = None
    total_nodes = None
    avg_before_lst = []
    avg_after_lst = []
    for i in range(num_runs):
        # Progress message
        print("Current run: ", i)

        # Run mean_field_approx.py to populate infected_recovered.txt
        os.system("python mean_field_approx.py")
        
        # Read from infected_recovered.txt
        split_point, total_nodes, before_list, after_list = split_triples_from_file()
        avg_before_lst.append(before_list)
        avg_after_lst.append(after_list)

        # Clear infected_recovered.txt for next iteration
        with open("experiment_data/infected_recovered.txt", "w") as f:
            f.truncate(0)

    # avg_before_lst has form: [[(day, infected, recovered), ...], ... ]
    # We need to keep days the same, and infected, recovered should be averaged along axis=0
    full_list = before_list + after_list

    def average_triple_lists(list_of_lists):
        # list_of_lists = [ [(t, inf, rec), (t, inf, rec), ...], ... ]
        avg_list = []
        for triples in zip(*list_of_lists):  # groups across runs
            times = [t for (t, _, _) in triples]
            assert len(set(times)) == 1  # sanity check: times match
            t = times[0]
            infs = [inf for (_, inf, _) in triples]
            recs = [rec for (_, _, rec) in triples]
            avg_list.append((t, np.mean(infs), np.mean(recs)))
        return avg_list

    full_list = average_triple_lists(avg_before_lst) + average_triple_lists(avg_after_lst)

    def plot_daily_infections():
        days = [t[0] for t in full_list]
        infections = [t[1] for t in full_list]
        plt.plot(days, infections, label='Daily Infections')
        plt.xlabel('Days')
        plt.ylabel('Number of Infections')
        plt.title('Daily Infections Over Time')
        plt.legend()
        plt.show()

    # plot_daily_infections()

    # print("Split point:", split_point)
    # print("Before:", before_list)
    # print("After:", after_list)
    print("k_0:", k_0)
    print("k_q:", k_q)
    print("Total nodes: ", total_nodes)

    # I_t: Sum of newly infected up to t, minus sum of newly recovered up to t
    # Approximate I(t) when only given daily data
    def I(time):
        return sum(full_list[i][1] for i in range(split_point, time)) - sum(full_list[i][2] for i in range(split_point, time))

    # Mean node degree if quarantine measures were followed 100%
    k_effective = [k_0 * (1 - I(time) / total_nodes) for time in range(split_point, len(full_list))]

    # Expected <k> at each time point under full adherence
    expected_k = (1 / len(k_effective)) * sum(k_effective)

    adherence = 1 - (k_q - expected_k) / k_q if k_q != 0 else 0
    # adherence = 1 - (expected_k - k_q) / expected_k if expected_k != 0 else 0

    print("Adherence, new formulation: ", adherence)

    with open("experiment_data/mfa_xy_data.txt", "w") as f:
        f.truncate(0)