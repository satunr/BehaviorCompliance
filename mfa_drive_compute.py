# Use this file to run the mean field approximation several times
import subprocess
import sys

repeat = 5
run_simulation = True
run_real_world = False

if run_simulation == True:
    for _ in range(repeat):
        subprocess.run([sys.executable, "mean_field_approx.py", "--subprocess"])
    subprocess.run([sys.executable, "extract_mfa.py"])

if run_real_world == True:
    #  Split point is in mfa_real_world.py
    subprocess.run([sys.executable, "mfa_real_world.py", "--first_half"])
    subprocess.run([sys.executable, "mfa_real_world.py", "--second_half"])
