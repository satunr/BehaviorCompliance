# Use this file to run the mean field approximation several times and average the results
import subprocess
import sys

#----------
#
#  File architecture: mfa_compute temporarily holds data from runs. Extracts averages and saves them to mfa_avgs.pkl. 
#    Cleared when a new SIR configuration is created.
#
#  Our averaging abstracts here away from the true values, so we're going off of just the optimizer's outputs.
#
#----------

repeat = 10
run_simulation = True
run_real_world = False

if run_simulation == True:
    for _ in range(repeat):
        subprocess.run([sys.executable, "mean_field_approx.py", "--subprocess"])

if run_real_world == True:
    #  Split point is in mfa_real_world.py
    subprocess.run([sys.executable, "mfa_real_world.py", "--first_half"])
    subprocess.run([sys.executable, "mfa_real_world.py", "--second_half"])
