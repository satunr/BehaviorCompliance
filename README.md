<p align="justify">This project provides a suite of Python scripts for analyzing and visualizing network-based quarantine simulations. The core functionality revolves around parsing outputs from Mean-Field Approximation (MFA) simulations, estimating quarantine adherence, and generating publication-quality figures for both infection and information spread dynamics. The scripts support post-quarantine analysis, cumulative adherence estimation, and optimizer-based evaluation of network metrics such as mean node degree. Key workflows include loading simulation data, computing summary statistics, fitting model parameters, and producing visualizations with mean and variability bands. Results are stored in pickle files for reproducibility and downstream analysis. The project is designed to facilitate quantitative evaluation of quarantine effects and network adherence dynamics in computational epidemiology studies.

analyze_quarantine_dynamics.py — Analyzes post-quarantine MFA simulation data to estimate adherence, optimize parameters, track dynamic adherence over time, and generate comparison plots. Outputs include PDF figures and processed data in pickle files.

analyze_yjmob_quarantine.py — Processes YJMob100k network simulations, estimates quarantine adherence using MFA degree dynamics, aggregates time-series data, and produces publication-quality plots with cumulative adherence metrics. Results are saved in a pickle file.

extract_mfa.py — Provides tools to parse MFA simulation output files into structured Python objects, handling time series of infections, informed individuals, and network metrics. Supports downstream analysis and averaging over simulation runs.

plot_infected_informed.py — Visualizes the spread of infection and information across different quarantine adherence levels. Produces PDF figures comparing grouped and individual adherence scenarios, with optional serialized data for reproducibility.

plot_informed.py — Tracks the informed fraction in a population over time. Generates mean ± standard deviation plots and saves processed data for further analysis.

plot_optimizer.py — Compares optimizer estimates of mean node degrees to ground truth values. Generates figures and saves processed data, highlighting pre- and post-quarantine dynamics with variability bands.

plot_split_optimization.py — Visualizes MFA-estimated mean node degrees across pre- and post-quarantine periods, including variability across multiple runs. Outputs publication-quality PDFs and serialized processed data.

sensitivity_analysis.py — Performs sensitivity analyses of the optimizer’s mean node degree estimates under varying simulation parameters. Generates plots with mean ± standard deviation bands and exports PDF figures for each parameter.

mfa_degree_estimates.pdf / .pkl — Figures and data comparing optimizer-estimated mean node degrees to ground truth across pre- and post-quarantine periods.

cumulative_adherence_results.pkl — Pickled results containing cumulative adherence estimates, true and estimated mean node degrees, and related metrics from MFA simulations.

Grouped adherence figure PDFs — Compare infection/information dynamics across multiple adherence levels.
</p>

Single adherence PDFs — Show mean dynamics and variability for a specific adherence scenario.

Sensitivity analysis PDFs — Visualize how estimates of initial mean node degree respond to changes in simulation parameters (e.g., number of seeds, initial infection proportion, network density).
}
