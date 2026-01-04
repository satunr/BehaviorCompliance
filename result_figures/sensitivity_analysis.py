import extract_mfa
import matplotlib.pyplot as plt
import numpy as np
import math

# Plot results from sensitivity analysis experiments

# Global matplotlib settings for publication-quality figures
plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "lines.linewidth": 2.5,
})

# Consistent figure size for all plots (suitable for Overleaf inclusion)
FIGURE_SIZE_LINE = (10, 6)
FIGURE_SIZE_BAR = (14, 8)

# parameter_name: string indicating which parameter was varied
# value_range: list of parameter values that were tested
def plot_sensitivity_results(file_path, parameter_name, value_range):
    samples = extract_mfa.parse_sample_data(file_path)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_LINE)

    max_length = 0
    true_k0 = None

    for i, sample in enumerate(samples):
        if 'w1 all runs' not in sample:
            continue

        w1_runs = sample['w1 all runs']['y']

        # Remove empty runs
        w1_runs = [run for run in w1_runs if len(run) > 0]
        if len(w1_runs) == 0:
            continue

        # Make all runs the same length
        max_length = max(max_length, max(len(run) for run in w1_runs))
        for run in w1_runs:
            while len(run) < max_length:
                run.append(run[-1])

        w1_array = np.array(w1_runs)
        mean_w1 = np.mean(w1_array, axis=0)
        std_w1 = np.std(w1_array, axis=0)
        time_points = np.arange(w1_array.shape[1])

        # --- Plot estimate and retrieve color ---
        line, = ax.plot(
            time_points,
            mean_w1,
            label=f'({parameter_name} = {value_range[i]})'
        )

        color = line.get_color()

        # Std band (same color, faded)
        ax.fill_between(
            time_points,
            mean_w1 - std_w1,
            mean_w1 + std_w1,
            color=color,
            alpha=0.15
        )

        # Save true value once (assumed constant across samples)
        if true_k0 is None and 'w1 True (Mean Node Degree)' in sample:
            true_k0 = sample['w1 True (Mean Node Degree)']['y'][-1]

    # True reference line
    if true_k0 is not None:
        ax.hlines(
            y=true_k0,
            xmin=0,
            xmax=max_length - 1,
            colors='black',
            linestyles='--',
            label=r"True $\langle k_0 \rangle$"
        )

    ax.set_xlabel("Time")
    ax.set_ylabel(r"Estimated $\langle k_0 \rangle$")
    ax.set_title(f"Sensitivity Analysis: {parameter_name}")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(f'result_figures/sensitivity_{parameter_name}.pdf', bbox_inches='tight')
    plt.close()

seed_range = [5, 10, 15]
seed_file = "experiment_data/sensitivity_num_seeds.txt"
beta_range = [0.05, 0.10, 0.20]
beta_file = "experiment_data/sensitivity_transmission_rate.txt"
edge_range = [1000, 1500, 2000]
density_range = [round(num_edge / (math.comb(200, 2)), 2) for num_edge in edge_range]
density_file = "experiment_data/sensitivity_density.txt"

plot_sensitivity_results(seed_file, "Number of Seeds", seed_range)
plot_sensitivity_results(beta_file, "Infection Rate (β)", beta_range)
plot_sensitivity_results(density_file, "Social Network Density", density_range)

