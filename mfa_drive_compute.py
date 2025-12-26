# Use this file to run the mean field approximation several times
import subprocess
import sys

repeat = 25  # Number of times to run the MFA

# Run MFA with different adherence levels
run_simulation = False

# Run MFA on real-world COVID-19 dataset
run_real_world = False

# Run MFA several times with same configuration
simple_repeat = False

# Run MFA on YJMob dataset
yjmob_mode = True

# Run mean field approximation with varying adherence levels
if run_simulation == True:
    write_to = ("experiment_data/a_0.2", "experiment_data/a_0.4", "experiment_data/a_0.6", 
                "experiment_data/a_0.8", "experiment_data/a_0.9", "experiment_data/a_1.0")
    adherence_level = (0.2, 0.4, 0.6, 0.8, 0.9, 1.0)

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

if simple_repeat == True:
    for _ in range(repeat):
        subprocess.run([sys.executable, "mean_field_approx.py"])

if yjmob_mode == True:
    files_to_clear = ["experiment_data/yjmob0_runs.txt", "experiment_data/yjmob1_runs.txt",
                      "experiment_data/yjmob2_runs.txt", "experiment_data/yjmob3_runs.txt",
                      "experiment_data/yjmob4_runs.txt"]
    
    for file in files_to_clear:
        open(file, "w").close()

    for _ in range(repeat):
        # File index: correspond to different time intervals in YJMob dataset
        for file_index in range(5):
            subprocess.run([sys.executable, "mean_field_approx.py", 
                            str(file_index)])