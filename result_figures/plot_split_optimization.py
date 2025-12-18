import extract_mfa
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "figure.titlesize": 24,
    "lines.linewidth": 2.5,
})

samples = extract_mfa.parse_sample_data("experiment_data/mfa_xy_data.txt")
pre = samples[0]
post = samples[1]

# ============================
# Align TRUE to ESTIMATED
# ============================
def align_true_to_est(sample):
    x_est = np.array(sample["w1 Estimated"]["x"])
    y_est = np.array(sample["w1 Estimated"]["y"])

    x_true = np.array(sample["w1 True (Mean Node Degree)"]["x"])
    y_true = np.array(sample["w1 True (Mean Node Degree)"]["y"])

    # Keep only true values defined at estimated x's
    mask = np.isin(x_true, x_est)

    return x_est, y_est, y_true[mask]

# Extract aligned data
x_pre, y_pre_est, y_pre_true = align_true_to_est(pre)
x_post, y_post_est, y_post_true = align_true_to_est(post)

split_point = x_pre[-1] + 1

# ============================
# Std bands
# ============================
def mean_std(runs, x):
    runs = np.array(runs)
    return runs.mean(axis=0), runs.std(axis=0)

pre_mean, pre_std = mean_std(pre["w1 all runs"]["y"], x_pre)
post_mean, post_std = mean_std(post["w1 all runs"]["y"], x_post)

# ============================
# Plot
# ============================
fig, ax = plt.subplots(figsize=(14, 7))

# ---- True (clipped to halves) ----
ax.plot(
    x_pre,
    y_pre_true,
    color="black",
    linestyle="-",
    label="True ⟨k⟩ (pre-quarantine)",
)
ax.plot(
    x_post,
    y_post_true,
    color="black",
    linestyle="-",
    label="True ⟨k⟩ (post-quarantine)",
)

# ---- Estimated ----
ax.plot(
    x_pre,
    y_pre_est,
    color="#ff7f0e",
    linestyle="--",
    label="Estimated ⟨k⟩ (pre)",
)
ax.plot(
    x_post,
    y_post_est,
    color="#d62728",
    linestyle="--",
    label="Estimated ⟨k⟩ (post)",
)

# ---- Std bands ----
ax.fill_between(
    x_pre,
    pre_mean - pre_std,
    pre_mean + pre_std,
    color="#ff7f0e",
    alpha=0.25,
)
ax.fill_between(
    x_post,
    post_mean - post_std,
    post_mean + post_std,
    color="#d62728",
    alpha=0.25,
)

# ---- Vertical split ----
ax.axvline(
    split_point,
    color="black",
    linestyle="--",
    linewidth=2,
    label="Quarantine begins",
)

# ============================
# Formatting
# ============================
ax.set_xlabel("Time (days)")
ax.set_ylabel("Mean Node Degree ⟨k⟩")
ax.set_title("Mean Field Approximation Estimated Degree with Quarantine")

ax.legend(frameon=False)
ax.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig("mfa_degree_estimates.pdf", format="pdf", bbox_inches="tight")
plt.close(fig)
