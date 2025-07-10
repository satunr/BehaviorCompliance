import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.cm as cm
import numpy as np

#----------
#
#  Parse data written by mean_field_approx.py to experiment_data/mfa_xy_data.txt
#
#----------

def parse_data(filepath):
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

# === RUN THIS ===
if __name__ == "__main__":
    filepath = "experiment_data/mfa_xy_data.txt"
    samples = parse_data(filepath)

    # Group and plot
    plot_group(samples, 
               key_true="w1 True (Mean Node Degree)", 
               key_est="w1 Estimated", 
               title="True vs Estimated Mean Node Degree")

    plot_group(samples, 
               key_true="w2 True (Recovered Fraction)", 
               key_est="w2 Estimated", 
               title="True vs Estimated Recovered Fraction")
