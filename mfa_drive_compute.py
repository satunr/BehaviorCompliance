# Use this file to run the mean field approximation several times
import subprocess
import sys
import matplotlib.pyplot as plt

repeat = 10  # Number of times to run the MFA
run_simulation = True
run_real_world = False

write_to = ("experiment_data/a_0.0", "experiment_data/a_0.2", "experiment_data/a_0.5",
            "experiment_data/a_0.6", "experiment_data/a_0.7", "experiment_data/a_1.0")
adherence_level = (0.0, 0.2, 0.5, 0.6, 0.7, 1.0)

# Run mean field approximation several times (same configuration) and extract results
if run_simulation == True:
    # Clear previous data files
    for file in write_to:
        open(file, "w").close()

    # Iteratively run MFA for each adherence level
    for i in range(len(write_to)):
        for _ in range(repeat):
            subprocess.run([sys.executable, "mean_field_approx.py", write_to[i], str(adherence_level[i])])
    # subprocess.run([sys.executable, "result_figures/analyze_quarantine_dynamics.py"])

# Run mean field approximation on real-world time-series infection data
if run_real_world == True:
    #  Split point is in mfa_real_world.py
    subprocess.run([sys.executable, "mfa_real_world.py", "--first_half"])
    subprocess.run([sys.executable, "mfa_real_world.py", "--second_half"])