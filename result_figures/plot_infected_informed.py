import extract_mfa
import matplotlib.pyplot as plt
import numpy as np
import pickle

#-------------------------------------------
#  Save file name
#-------------------------------------------
PKL_FILENAME = "result_figures/plot_infected_informed_data.pkl"

#-------------------------------------------
#  Plot Informed, Infected, and Informed & Infected grouped by adherence
#-------------------------------------------
def plot_groups_by_adherence(odd_samples):

    # Truncate to a multiple of 5 (one full set for each adherence level)
    num_full_groups = len(odd_samples) // 5
    odd_samples = odd_samples[:num_full_groups * 5]

    saved_output = {}   # ← data to be saved into PKL

    if len(odd_samples) == 0:
        print("No post-quarantine samples found.")
        return

    n = odd_samples[0]['Number of nodes']

    # Reshape: 5 adherence groups
    grouped = np.array(odd_samples).reshape(5, -1)

    adherence_levels = [0.2, 0.4, 0.6, 0.8, 1.0]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    styles = {
        'Infected':            ('solid',  2.5),
        'Informed & Infected': ('dashed', 2.2),
        'Informed':            ('dotted', 2.5)
    }

    plt.figure(figsize=(14, 8))

    legend_handles = []
    legend_labels  = []

    saved_output["grouped_results"] = {}

    for idx, (a, color) in enumerate(zip(adherence_levels, colors)):
        group = grouped[idx]

        # Extract lists of curves
        inf_data = [s['SIR Infections']['y'] for s in group]
        ii_data  = [s['Informed and Infected']['y'] for s in group]
        i_data   = [s['Informed']['y'] for s in group]

        t = np.arange(200)

        # Save raw curves
        saved_output["grouped_results"][a] = {
            "infected_raw": inf_data,
            "informed_raw": i_data,
            "inf_and_inf_raw": ii_data,
            "n": n,
            "time": t.tolist()
        }

        # Plot each type
        for name, data_list in [
            ('Infected', inf_data),
            ('Informed & Infected', ii_data),
            ('Informed', i_data)
        ]:
            mean_curve = np.mean(data_list, axis=0) / n
            linestyle, lw = styles[name]

            line = plt.plot(
                t[:len(mean_curve)], mean_curve,
                color=color, linestyle=linestyle, linewidth=lw,
                label=f'{name} (a = {a})'
            )[0]

            legend_handles.append(line)
            legend_labels.append(f'{name} (a = {a})')

            # Save mean curves
            saved_output["grouped_results"][a][f"{name}_mean"] = mean_curve.tolist()

    plt.xlabel('Time Step', fontsize=13)
    plt.ylabel('Fraction of Population', fontsize=13)
    plt.title('Post-Quarantine Dynamics by Adherence Level', fontsize=15)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.02)

    plt.legend(handles=legend_handles, labels=legend_labels,
               ncol=3, fontsize=10.8,
               loc='lower center', bbox_to_anchor=(0.5, -0.20),
               frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.24)
    plt.show()

    #-------------------------------------------
    # Save to PKL
    #-------------------------------------------
    with open(PKL_FILENAME, "wb") as f:
        pickle.dump(saved_output, f)

    print(f"Saved grouped adherence curves → {PKL_FILENAME}")


#-------------------------------------------
#  Plot single-sample infected/informed dynamics (two y-axes)
#-------------------------------------------

def plot_one_adherence_group(samples):
    if len(samples) == 0:
        print("No samples provided.")
        return

    n = samples[0]['Number of nodes']
    T = len(samples[0]['SIR Infections']['y'])
    t = np.arange(T)

    # --- Collect y-curves from all samples ---
    infected_list = [np.array(s['SIR Infections']['y'], dtype=float) for s in samples]
    informed_list = [np.array(s['Informed']['y'], dtype=float) for s in samples]
    inf_inf_list  = [np.array(s['Informed and Infected']['y'], dtype=float) for s in samples]

    infected_arr = np.vstack(infected_list) / n
    informed_arr = np.vstack(informed_list) / n
    inf_inf_arr  = np.vstack(inf_inf_list) / n

    # --- Mean curves ---
    infected_mean = infected_arr.mean(axis=0)
    informed_mean = informed_arr.mean(axis=0)
    inf_inf_mean  = inf_inf_arr.mean(axis=0)

    # --- Std curves ---
    infected_std = infected_arr.std(axis=0)
    informed_std = informed_arr.std(axis=0)
    inf_inf_std  = inf_inf_arr.std(axis=0)

    # --- Plotting ---
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
        borderpad=0.8,
        labelspacing=0.7,
        handlelength=2.5,
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


samples = extract_mfa.parse_sample_data("experiment_data/mfa_xy_data.txt")

# Only odd samples (post-quarantine)
odd_samples = [samples[i] for i in range(len(samples)) if i % 2 == 1]

# plot_groups_by_adherence(odd_samples)
plot_one_adherence_group(samples)
