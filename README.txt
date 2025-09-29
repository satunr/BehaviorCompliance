Compile and run:
IDE: VSCode (should be optional)
Language: Python
Version: 3.11.7
Libraries: matplotlib, networkx, p4c (Python for Cytoscape), numpy, scipy

To run the optimization problem, use mean_field_approx.py, where you can set all your parameters
    If you are trying a new network configuration, make sure to set clear to True
    Seed nodes, initial infected, social graph density can all be controlled here
experiment_data/mfa_xy_data.py will show you all the data collected from the optimization (and some extras)
To extract this data into usable lists, use extract_mfa.py
    Notes on how to extract these lists at the bottom of that file (extract_mfa.py)