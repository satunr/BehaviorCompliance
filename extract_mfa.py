import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.cm as cm
import numpy as np
import pickle
import re
from itertools import cycle

plot_opt = False
plot_new = False
plot_many = True

#----------
#
#  Parse data written by mean_field_approx.py to experiment_data/mfa_xy_data.txt
#
#----------

import matplotlib.pyplot as plt
import re

def parse_sample_data(filename):
    samples = []
    current_sample = {}
    current_dataset = None
    x_data = []
    y_data = []
    
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
            # Match dataset names, including those with parenthetical descriptions
            if line.startswith("SIR Infections") or line.startswith("w1 True") or line.startswith("w1 Estimated") or line.startswith("w2 True") or line.startswith("w2 Estimated"):
                if current_dataset and x_data:
                    current_sample[current_dataset] = {'x': x_data, 'y': y_data}
                    x_data = []
                    y_data = []
                # Capture the full dataset name, including parenthetical part
                current_dataset = line.split(':')[0].strip()
                continue
            if line.startswith("x:"):
                x_data = [float(x) for x in line[2:].split(',')]
            if line.startswith("y:"):
                y_data = [float(y) for y in line[2:].split(',')]
        
        if current_dataset and x_data:
            current_sample[current_dataset] = {'x': x_data, 'y': y_data}
        if current_sample:
            samples.append(current_sample)
    
    return samples

def plot_sample(sample, sample_num):
    # Plot w1 True vs w1 Estimated
    plt.figure(figsize=(10, 6))
    plt.plot(sample['w1 True (Mean Node Degree)']['x'], sample['w1 True (Mean Node Degree)']['y'], label='w1 True (Mean Node Degree)', color='#1f77b4', linestyle='-')
    plt.plot(sample['w1 Estimated']['x'], sample['w1 Estimated']['y'], label='w1 Estimated', color='#ff7f0e', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Mean Node Degree')
    plt.title(f'Sample {sample_num}: w1 True vs Estimated')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot w2 True vs w2 Estimated
    plt.figure(figsize=(10, 6))
    plt.plot(sample['w2 True (Recovered Fraction)']['x'], sample['w2 True (Recovered Fraction)']['y'], label='w2 True (Recovered Fraction)', color='#2ca02c', linestyle='-')
    plt.plot(sample['w2 Estimated']['x'], sample['w2 Estimated']['y'], label='w2 Estimated', color='#d62728', linestyle='--')
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

    # Plot for Sample 2
    plot_sample(samples[1], 2)

def plot_halves(samples):
    # Divide samples into two halves
    n_samples = len(samples)
    mid_point = (n_samples + 1) // 2  # Ceiling division to split as evenly as possible
    first_half = samples[:mid_point]
    second_half = samples[mid_point:]
    
    # Define color cycle for different samples
    colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])  # Matplotlib default colors
    
    # Plot for first half
    if first_half:
        # Plot all SIR curves
        plt.figure(figsize=(10, 6))
        for i, sample in enumerate(first_half, 1):
            color = next(colors)
            plt.plot(sample['SIR Infections (Inset)']['x'], sample['SIR Infections (Inset)']['y'], label=f'Sample {i}', color=color)
        plt.xlabel('Time')
        plt.ylabel('SIR Infections')
        plt.title('First Half: SIR Infections')
        plt.grid(True)
        plt.show()
        
        # Reset color cycle for w1 plot
        colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        
        # Plot all w1 True and Estimated
        plt.figure(figsize=(10, 6))
        for i, sample in enumerate(first_half, 1):
            color = next(colors)
            plt.plot(sample['w1 True (Mean Node Degree)']['x'], sample['w1 True (Mean Node Degree)']['y'], 
                     label=f'Sample {i} w1 True', color=color, linestyle='-')
            plt.plot(sample['w1 Estimated']['x'], sample['w1 Estimated']['y'], 
                     label=f'Sample {i} w1 Estimated', color=color, linestyle='--')
        plt.xlabel('Time')
        plt.ylabel('Mean Node Degree')
        plt.title('First Half: w1 True vs Estimated')
        plt.grid(True)
        plt.show()
    
    # Reset color cycle for second half
    colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    
    # Plot for second half
    if second_half:
        # Plot all SIR curves
        plt.figure(figsize=(10, 6))
        for i, sample in enumerate(second_half, mid_point + 1):
            color = next(colors)
            plt.plot(sample['SIR Infections (Inset)']['x'], sample['SIR Infections (Inset)']['y'], label=f'Sample {i}', color=color)
        plt.xlabel('Time')
        plt.ylabel('SIR Infections')
        plt.title('Second Half: SIR Infections')
        plt.grid(True)
        plt.show()
        
        # Reset color cycle for w1 plot
        colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        
        # Plot all w1 True and Estimated
        plt.figure(figsize=(10, 6))
        for i, sample in enumerate(second_half, mid_point + 1):
            color = next(colors)
            plt.plot(sample['w1 True (Mean Node Degree)']['x'], sample['w1 True (Mean Node Degree)']['y'], 
                     label=f'Sample {i} w1 True', color=color, linestyle='-')
            plt.plot(sample['w1 Estimated']['x'], sample['w1 Estimated']['y'], 
                     label=f'Sample {i} w1 Estimated', color=color, linestyle='--')
        plt.xlabel('Time')
        plt.ylabel('Mean Node Degree')
        plt.title('Second Half: w1 True vs Estimated')
        plt.grid(True)
        plt.show()

if plot_many == True:
    # Parse the average results from the file
    samples = parse_sample_data("experiment_data/mfa_xy_data.txt")

    plot_halves(samples)

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

    
