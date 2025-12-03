import extract_mfa
import matplotlib.pyplot as plt
import numpy as np

#---------------
#
#  Plot Split Optimization
#
#---------------

sample = extract_mfa.parse_sample_data('experiment_data/mfa_xy_data.txt')[0]

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
plt.title(f'Recovered Proportion True vs Estimated')
plt.legend()
plt.grid(True)
plt.show()