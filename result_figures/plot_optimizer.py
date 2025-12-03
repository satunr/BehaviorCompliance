import extract_mfa
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Plot w1_all_runs for specified half
# even_or_odd = 0: only plot first half (k_0) estimates
# even_or_odd = 1: only plot second half (k_q) estimates
# even_or_odd = 2: plot all samples
def plot_independence(samples, even_or_odd):
    plt.figure(figsize=(12, 7))

    results_list = []   # ← store results for pickle saving

    for i, sample in enumerate(samples):

        # Select based on mode:
        if not (even_or_odd == 2 or i % 2 == even_or_odd):
            continue

        if 'w1 all runs' not in sample:
            continue

        w1_runs = sample['w1 all runs']['y']

        # Remove empty runs
        w1_runs = [run for run in w1_runs if len(run) > 0]
        if len(w1_runs) == 0:
            continue

        # Make all runs the same length
        max_length = max(len(run) for run in w1_runs)
        for run in w1_runs:
            while len(run) < max_length:
                run.append(run[-1])

        w1_array = np.array(w1_runs)
        mean_w1 = np.mean(w1_array, axis=0)
        std_w1 = np.std(w1_array, axis=0)
        time_points = np.arange(w1_array.shape[1])

        # --- Plot estimate & retrieve its color ---
        line = plt.plot(
            time_points, mean_w1,
            label=f'Sample {i} Estimate (β = {0.15 - 0.03 * i})'
        )[0]

        color = line.get_color()

        # Std band uses same color but faded
        plt.fill_between(
            time_points, mean_w1 - std_w1, mean_w1 + std_w1,
            alpha=0.15, color=color
        )

        # Ground truth uses the same color
        true_val = None
        if 'w1 True (Mean Node Degree)' in sample:
            true_val = sample['w1 True (Mean Node Degree)']['y'][-1]
            plt.hlines(
                y=true_val, xmin=0, xmax=time_points[-1],
                linestyles='--',
                color=color,
                label=f'Sample {i} Ground Truth'
            )

        # --- Save to results list for pickling ---
        results_list.append({
            "sample_index": i,
            "time_points": time_points,
            "mean_w1": mean_w1,
            "std_w1": std_w1,
            "true_val": true_val,
        })

    plt.xlabel('Time')
    plt.ylabel('Degree Estimate from Optimizer')
    plt.title(r'$\langle k_q \rangle$ Estimates with Varying $\beta$')
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Save everything to a .pkl file ---
    with open("result_figures/plot_optimizer_data.pkl", "wb") as f:
        pickle.dump(results_list, f)

samples = extract_mfa.parse_sample_data('experiment_data/mfa_xy_data.txt')
plot_independence(samples, even_or_odd=2)
