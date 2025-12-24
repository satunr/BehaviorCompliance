import extract_mfa
import matplotlib.pyplot as plt
import numpy as np
import pickle

# ============================================================
# Global matplotlib settings
# ============================================================

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "lines.linewidth": 2.5,
})

# ============================================================
# Output files
# ============================================================

PKL_FILENAME = "result_figures/plot_infected_informed_data.pkl"

FIG_GROUPED_PDF = "experiment_data/quarantine_dynamics_by_adherence.pdf"
FIG_SINGLE_PDF  = "experiment_data/joint_informed_infected_dynamics.pdf"

# ============================================================
# Adherence data locations
# ============================================================

ADHERENCE_FILES = {
    0.2: "experiment_data/a_0.2",
    0.4: "experiment_data/a_0.4",
    0.6: "experiment_data/a_0.6",
    0.8: "experiment_data/a_0.8",
    0.9: "experiment_data/a_0.9",
    1.0: "experiment_data/a_1.0",
}

# ============================================================
# Grouped adherence plot
# ============================================================

def plot_groups_by_adherence(show=False):

    colors = {
        0.2: '#1f77b4',
        0.4: '#ff7f0e',
        0.6: '#2ca02c',
        0.8: '#d62728',
        0.9: '#9467bd',
        1.0: '#127b8f',
    }

    styles = {
        'Infected':            'solid',
        'Informed & Infected': 'dashed',
        'Informed':            'dotted'
    }

    fig, ax = plt.subplots(figsize=(14, 8))
    saved_output = {"grouped_results": {}}

    legend_handles = []
    legend_labels = []

    for a, filepath in ADHERENCE_FILES.items():

        samples = extract_mfa.parse_sample_data(filepath)
        samples = [samples[i] for i in range(len(samples)) if i % 2 == 1]

        if not samples:
            continue

        n = samples[0]['Number of nodes']
        T = len(samples[0]['SIR Infections']['y'])
        split_point = int(samples[0]['Informed']['x'][0])

        t = split_point + np.arange(T)

        infected_arr = np.vstack([s['SIR Infections']['y'] for s in samples]) / n
        informed_arr = np.vstack([s['Informed']['y'] for s in samples]) / n
        inf_inf_arr  = np.vstack([s['Informed and Infected']['y'] for s in samples]) / n

        curves = {
            'Infected': (infected_arr.mean(0), infected_arr.std(0)),
            'Informed': (informed_arr.mean(0), informed_arr.std(0)),
            'Informed & Infected': (inf_inf_arr.mean(0), inf_inf_arr.std(0)),
        }

        saved_output["grouped_results"][a] = {
            "n": n,
            "time": t.tolist()
        }

        for name, (mean, std) in curves.items():

            line, = ax.plot(
                t, mean,
                color=colors[a],
                linestyle=styles[name],
                label=f"{name} (a={a})"
            )

            ax.fill_between(
                t, mean - std,
                mean + std,
                color=colors[a],
                alpha=0.18
            )

            legend_handles.append(line)
            legend_labels.append(f"{name} (a={a})")

            # saved_output["grouped_results"][a][f"{name}_mean"] = mean.tolist()
            # saved_output["grouped_results"][a][f"{name}_std"]  = std.tolist()

    ax.set_xlabel("Time step")
    ax.set_ylabel("Fraction of population")
    ax.set_title("Quarantine dynamics by adherence level")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)

    # ax.legend(
    #     handles=legend_handles,
    #     labels=legend_labels,
    #     ncol=3,
    #     loc="lower center",
    #     bbox_to_anchor=(0.5, -0.28),
    #     frameon=True
    # )

    fig.tight_layout()
    fig.savefig(FIG_GROUPED_PDF, format="pdf", bbox_inches="tight")
    plt.close(fig)

    with open(PKL_FILENAME, "wb") as f:
        pickle.dump(saved_output, f)

    if show:
        plt.show()

    print(f"Saved grouped adherence figure → {FIG_GROUPED_PDF}")

# ============================================================
# Single adherence plot
# ============================================================

def plot_one_adherence_group(samples, show=False):

    if not samples:
        return

    n = samples[0]['Number of nodes']
    T = len(samples[0]['SIR Infections']['y'])
    split_point = int(samples[0]['Informed']['x'][0])
    t = split_point + np.arange(T)

    infected_arr = np.vstack([s['SIR Infections']['y'] for s in samples]) / n
    informed_arr = np.vstack([s['Informed']['y'] for s in samples]) / n
    inf_inf_arr  = np.vstack([s['Informed and Infected']['y'] for s in samples]) / n

    fig, ax = plt.subplots(figsize=(13, 8))

    # Title
    ax.set_title(f"Quarantine Dynamics Under Full Adherence")

    def plot_band(mean, std, label, color):
        ax.plot(t, mean, label=label, color=color)
        ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.22)

    # plot_band(informed_arr.mean(0), informed_arr.std(0), "Informed", "#2ca02c")
    # plot_band(inf_inf_arr.mean(0), inf_inf_arr.std(0), "Informed & Infected", "#ff7f0e")
    # plot_band(infected_arr.mean(0), infected_arr.std(0), "Infected", "#d62728")

    ax.set_xlabel("Time")
    ax.set_ylabel("Fraction of population")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)

    # ax.legend(
    #     loc="upper center",
    #     bbox_to_anchor=(0.5, -0.12),
    #     ncol=3,
    #     frameon=True
    # )

    fig.tight_layout()
    fig.savefig(FIG_SINGLE_PDF, format="pdf", bbox_inches="tight")
    plt.close(fig)

    if show:
        plt.show()

    print(f"Saved single adherence figure → {FIG_SINGLE_PDF}")

# ============================================================
# Pairwise standardized L2 distances
# ============================================================

def compute_pairwise_L2_informed_infected():

    with open(PKL_FILENAME, "rb") as f:
        data = pickle.load(f)

    grouped = data["grouped_results"]
    adherence_levels = sorted(grouped.keys())
    k = len(adherence_levels)

    means = {a: np.array(grouped[a]["Informed & Infected_mean"]) for a in adherence_levels}
    stds  = {a: np.array(grouped[a]["Informed & Infected_std"])  for a in adherence_levels}

    D = np.zeros((k, k))

    for i, ai in enumerate(adherence_levels):
        for j, aj in enumerate(adherence_levels):

            if i == j:
                continue

            pooled = np.sqrt(stds[ai]**2 + stds[aj]**2)
            valid = pooled > 0

            z = np.zeros_like(means[ai])
            z[valid] = (means[ai][valid] - means[aj][valid]) / pooled[valid]

            D[i, j] = np.sqrt(np.mean(z**2))

    return adherence_levels, D

# if __name__ == "__main__":
    # read_file = "experiment_data/a_1.0"
    # samples = extract_mfa.parse_sample_data(read_file)
    # samples = [samples[i] for i in range(len(samples)) if i % 2 == 1]
    
    # # Shorten every sample to first 70 time steps
    # for sample in samples:
    #     for key in sample:
    #         if isinstance(sample[key], dict) and 'y' in sample[key]:
    #             sample[key]['y'] = sample[key]['y'][:70]
    #             sample[key]['x'] = sample[key]['x'][:70]

    # plot_one_adherence_group(samples, show=True)

    # plot_groups_by_adherence(show=True)