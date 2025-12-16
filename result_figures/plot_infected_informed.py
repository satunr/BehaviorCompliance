import extract_mfa
import matplotlib.pyplot as plt
import numpy as np
import pickle

# -------------------------------------------
# Save file name
# -------------------------------------------
PKL_FILENAME = "result_figures/plot_infected_informed_data.pkl"

# -------------------------------------------
# Adherence files
# -------------------------------------------
ADHERENCE_FILES = {
    0.0: "experiment_data/a_0.0",
    0.2: "experiment_data/a_0.2",
    0.5: "experiment_data/a_0.5",
    0.6: "experiment_data/a_0.6",
    0.7: "experiment_data/a_0.7",
    1.0: "experiment_data/a_1.0",
}

# -------------------------------------------
# Plot grouped adherence curves with std bands
# -------------------------------------------
def plot_groups_by_adherence():

    colors = {
        0.0: '#1f77b4',
        0.2: '#ff7f0e',
        0.5: '#2ca02c',
        0.6: '#d62728',
        0.7: '#9467bd',
        1.0: '#8c564b',
    }

    styles = {
        'Infected':            ('solid',  2.5),
        'Informed & Infected': ('dashed', 2.2),
        'Informed':            ('dotted', 2.5)
    }

    plt.figure(figsize=(14, 8))

    saved_output = {"grouped_results": {}}
    legend_handles = []
    legend_labels = []

    for a, filepath in ADHERENCE_FILES.items():

        samples = extract_mfa.parse_sample_data(filepath)

        # --- odd indexed samples only ---
        samples = [samples[i] for i in range(len(samples)) if i % 2 == 1]

        if len(samples) == 0:
            print(f"No usable samples for adherence = {a}")
            continue

        n = samples[0]['Number of nodes']
        T = len(samples[0]['SIR Infections']['y'])
        t = np.arange(T)

        # --- collect curves ---
        infected_arr = np.vstack(
            [s['SIR Infections']['y'] for s in samples]
        ) / n

        informed_arr = np.vstack(
            [s['Informed']['y'] for s in samples]
        ) / n

        inf_inf_arr = np.vstack(
            [s['Informed and Infected']['y'] for s in samples]
        ) / n

        # --- mean + std ---
        curves = {
            'Infected': (infected_arr.mean(axis=0), infected_arr.std(axis=0)),
            'Informed': (informed_arr.mean(axis=0), informed_arr.std(axis=0)),
            'Informed & Infected': (inf_inf_arr.mean(axis=0), inf_inf_arr.std(axis=0)),
        }

        saved_output["grouped_results"][a] = {
            "n": n,
            "time": t.tolist()
        }

        # --- plotting ---
        for name, (mean, std) in curves.items():
            linestyle, lw = styles[name]

            line = plt.plot(
                t,
                mean,
                color=colors[a],
                linestyle=linestyle,
                linewidth=lw,
                label=f'{name} (a = {a})'
            )[0]

            plt.fill_between(
                t,
                mean - std,
                mean + std,
                color=colors[a],
                alpha=0.18
            )

            legend_handles.append(line)
            legend_labels.append(f'{name} (a = {a})')

            saved_output["grouped_results"][a][f"{name}_mean"] = mean.tolist()
            saved_output["grouped_results"][a][f"{name}_std"] = std.tolist()

    plt.xlabel('Time Step', fontsize=13)
    plt.ylabel('Fraction of Population', fontsize=13)
    plt.title('Quarantine Dynamics by Adherence Level', fontsize=15)

    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.02)

    plt.legend(
        handles=legend_handles,
        labels=legend_labels,
        ncol=3,
        fontsize=10.5,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.25),
        frameon=True,
        fancybox=True,
        shadow=True
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.30)
    plt.show()

    # --- Save to PKL ---
    with open(PKL_FILENAME, "wb") as f:
        pickle.dump(saved_output, f)

    print(f"Saved grouped adherence curves → {PKL_FILENAME}")

# -------------------------------------------
# Plot single-adherence dynamics (two y-axes)
# -------------------------------------------
def plot_one_adherence_group(samples):

    if len(samples) == 0:
        print("No samples provided.")
        return

    n = samples[0]['Number of nodes']
    T = len(samples[0]['SIR Infections']['y'])
    t = np.arange(T)

    infected_arr = np.vstack(
        [s['SIR Infections']['y'] for s in samples]
    ) / n

    informed_arr = np.vstack(
        [s['Informed']['y'] for s in samples]
    ) / n

    inf_inf_arr = np.vstack(
        [s['Informed and Infected']['y'] for s in samples]
    ) / n

    infected_mean = infected_arr.mean(axis=0)
    informed_mean = informed_arr.mean(axis=0)
    inf_inf_mean = inf_inf_arr.mean(axis=0)

    infected_std = infected_arr.std(axis=0)
    informed_std = informed_arr.std(axis=0)
    inf_inf_std = inf_inf_arr.std(axis=0)

    plt.figure(figsize=(13, 8))
    plt.title(
        "Joint Evolution of Informed and SIRS Epidemic Model",
        fontsize=20,
        pad=20
    )

    def plot_with_band(mean, std, label, color):
        plt.plot(t, mean, color=color, linewidth=3.0, label=label)
        plt.fill_between(t, mean - std, mean + std, color=color, alpha=0.22)

    plot_with_band(informed_mean, informed_std, "Informed", "#2ca02c")
    plot_with_band(inf_inf_mean, inf_inf_std, "Informed & Infected", "#ff7f0e")
    plot_with_band(infected_mean, infected_std, "Infected", "#d62728")

    plt.xlabel("Time (in days)", fontsize=18)
    plt.ylabel("Fraction of Population", fontsize=18)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.3)

    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        frameon=True,
        fontsize=15,
    )

    plt.tight_layout()
    plt.show()

    # --- Save results to PKL ---
    with open(PKL_FILENAME, "rb") as f:
        data = pickle.load(f)

    data["single_adherence_group"] = {
        "n": n,
        "time": t.tolist(),
        "infected_mean": infected_mean.tolist(),
        "infected_std": infected_std.tolist(),
        "informed_mean": informed_mean.tolist(),
        "informed_std": informed_std.tolist(),
        "inf_and_inf_mean": inf_inf_mean.tolist(),
        "inf_and_inf_std": inf_inf_std.tolist(),
    }

    with open(PKL_FILENAME, "wb") as f:
        pickle.dump(data, f)

# -------------------------------------------
# Example usage
# -------------------------------------------

# Plot grouped adherence curves
plot_groups_by_adherence()

# Plot single-adherence group if desired
# samples = extract_mfa.parse_sample_data("experiment_data/a_0.5")
# samples = [samples[i] for i in range(len(samples)) if i % 2 == 1]
# plot_one_adherence_group(samples)


# -----------------------
#
#  Compute pairwise L^2 distances between 'Informed & Infected' curves
#    to see if there's confusion between adherence levels in our approximation for <k_q>
#
#------------------------

PKL_FILENAME = "result_figures/plot_infected_informed_data.pkl"

def compute_pairwise_L2_informed_infected():
    """
    Compute pairwise standardized L^2 distances between
    'Informed & Infected' curves for each adherence level.
    """

    with open(PKL_FILENAME, "rb") as f:
        data = pickle.load(f)

    grouped = data["grouped_results"]

    adherence_levels = sorted(grouped.keys())
    k = len(adherence_levels)

    # --- extract curves ---
    means = {}
    stds = {}

    for a in adherence_levels:
        means[a] = np.array(grouped[a]["Informed & Infected_mean"])
        stds[a]  = np.array(grouped[a]["Informed & Infected_std"])

    # --- distance matrix ---
    D = np.zeros((k, k))

    for i, a_i in enumerate(adherence_levels):
        for j, a_j in enumerate(adherence_levels):

            if i == j:
                continue

            mu_i = means[a_i]
            mu_j = means[a_j]

            sigma_i = stds[a_i]
            sigma_j = stds[a_j]

            pooled_std = np.sqrt(sigma_i**2 + sigma_j**2)

            # avoid divide-by-zero
            valid = pooled_std > 0

            z = np.zeros_like(mu_i)
            z[valid] = (mu_i[valid] - mu_j[valid]) / pooled_std[valid]

            D[i, j] = np.sqrt(np.mean(z**2))

    return adherence_levels, D

levels, D = compute_pairwise_L2_informed_infected()

print("Adherence levels:", levels)
print("Pairwise standardized L2 matrix:")
print(np.round(D, 3))

from sklearn.linear_model import LinearRegression

def compute_curve_nonlinearity(curves):
    """
    curves: 2D array of shape (num_samples, T)
    Returns: array of nonlinearity values for each sample
    """
    T = curves.shape[1]
    t = np.arange(T).reshape(-1, 1)

    nonlinearity = []
    for y in curves:
        model = LinearRegression().fit(t, y)
        y_pred = model.predict(t)
        resid = y - y_pred
        nonlinearity.append(np.sqrt(np.mean(resid**2)))  # RMS residual
    return np.array(nonlinearity)

# Store results per adherence level
dynamic_degree_nonlinearity = {}

for a, filepath in ADHERENCE_FILES.items():
    samples = extract_mfa.parse_sample_data(filepath)
    samples = [samples[i] for i in range(len(samples)) if i % 2 == 1]

    if len(samples) == 0:
        print(f"No usable samples for adherence = {a}")
        continue

    # Stack Dynamic degree curves
    kq_arr = np.vstack([s['Dynamic degree']['y'] for s in samples])

    # Compute nonlinearity per sample and average
    nl = compute_curve_nonlinearity(kq_arr)
    dynamic_degree_nonlinearity[a] = np.mean(nl)

# Print results
for a, nl in dynamic_degree_nonlinearity.items():
    print(f"Adherence {a:.1f}: Nonlinearity = {nl:.3f}")