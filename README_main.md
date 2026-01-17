# When knowledge meets infection, ties break: a multiplex approach to behavioral compliance

### correlated_graphs.py.
<p align="justify">
This code implements several methods for generating and analyzing directed graphs with structured correlation and similarity properties. It includes a core routine, create_correlated_digraph, which constructs a directed graph whose edge existence is correlated with an underlying undirected base graph, enabling controlled sparsity through a tunable correlation factor. In addition, the code provides create_w_k_hop_correlation, which generates directed edges based on Jaccard similarity between k-hop neighborhoods, offering a mechanism to capture proximity and local connectivity patterns in graphs. Finally, the create_social_graph function models social network structure by ranking node pairs using Jaccard similarity and sampling directed edges accordingly, allowing the creation of directed social graphs with flexible and interpretable edge density profiles.

### extract_loss_calc.py.
<p align="justify">
This code loads a real-world social network, computes loss matrices for Independent Cascade (IC) and Linear Threshold (LT) diffusion models, and manages their structured storage and retrieval. It parses a Facebook network dataset, relabels nodes to ensure consistent indexing, and generates multiple correlated network instances to support robust loss evaluation. The resulting IC and LT loss matrices are written to a well-defined text format with validation checks that ensure safe and reliable re-parsing. For downstream analysis and comparison, the parsed matrices are further simplified by retaining only their final rows, enabling efficient inspection and reuse without recomputing the full diffusion processes.
</p>

### find_seeds.py.
<p align="justify">
This code provides a probabilistic framework for selecting seed nodes in a social network based on node degrees and a configurable exponent that controls selection bias. The find_seed_set function computes selection probabilities for each node, enabling the construction of a diverse yet influential seed set rather than relying on purely deterministic choices. The initialize_social_IM function then initializes a social influence model—either Independent Cascade or Linear Threshold—using the selected seeds and specified influence parameters, and runs simulations to estimate the resulting influence spread. Together, these components support flexible seed selection and influence maximization experiments across different network structures and diffusion dynamics.
</p>

### IM.py.
<p align="justify">
This code models influence spread in social networks using the Independent Cascade (IC) and Linear Threshold (LT) diffusion frameworks. The IC_prob_matrix function estimates the probability of each node becoming informed through repeated Monte Carlo simulations, while the IC routine simulates realized influence propagation paths. A greedy selection strategy is implemented to identify seed nodes that maximize expected influence under the IC model, and a corresponding greedy_for_lt function extends this approach to the LT model by explicitly accounting for node-specific activation thresholds. Both diffusion models incorporate quarantine constraints, ensuring that quarantined nodes are not activated during simulations, and define influence in terms of the activation states of neighboring nodes, enabling realistic modeling of constrained information or disease spread.
</p>

### informingcontact.py.
<p align="justify">
This code simulates and visualizes influence spread in social and contact networks using the Independent Cascade (IC) and Linear Threshold (LT) diffusion models. It constructs a contact network from input files, correlates it with an underlying social network, and runs an SIR epidemic simulation on the social graph to capture disease dynamics. The resulting social structure is then used for Influence Maximization (IM) to identify key seed nodes whose selection aims to mitigate infection spread by strategically removing edges from the contact network. The implementation supports rich visualization workflows, including integration with Cytoscape via py4cytoscape, exporting networks to GML, and inline plotting with matplotlib. Finally, it evaluates the effectiveness of the greedy IM strategy against random seed selection by comparing influence spread and plotting edge-removal efficiency as a function of k, the number of selected seed nodes.
</p>

### lt_ic_loss_function.py.
<p align="justify">
This code calculates the loss between observed and inferred quarantine data under the Independent Cascade (IC) and Linear Threshold (LT) diffusion models, with a particular focus on how varying threshold parameters affect model discrepancy. It generates subgraphs from an underlying social network using random walks, applies random node removal and edge existence probabilities, and computes LT loss relative to IC using mean squared error (MSE) as the evaluation metric. The implementation supports generating multiple network realizations to assess variability in loss behavior across different graph instances. Results are visualized through a 3D bar plot that compares losses across networks, along with additional plots that illustrate how loss changes as a function of the LT threshold, enabling systematic sensitivity analysis.
</p>

### mean_field_approx.py.
<p align="justify">
This code simulates a Susceptible–Infected–Recovered (SIR) model on a contact network and optimizes key epidemiological parameters, including the mean node degree and recovered fraction, using a mean-field approximation. It supports multiple execution modes—such as standard simulation, adherence-based dynamics, YJMOB scenarios, and sensitivity analysis—allowing flexible exploration of intervention and behavioral effects. Model parameters are estimated via L-BFGS-B optimization by minimizing the error between predicted and observed infection and recovery trajectories. The simulation tracks newly infected and recovered populations over time and saves these results to disk for downstream analysis, validation, and comparison across scenarios.
</p>

### mfa_drive_compute.py.
<p align="justify">
This code repeatedly runs a Mean Field Approximation (MFA) model under varying parameter settings to explore a range of epidemic and behavioral scenarios. It includes an adherence mode that evaluates system dynamics across multiple adherence levels, saving results for comparative analysis. The simple repeat mode executes the MFA multiple times with identical configurations to assess stability and variability. The YJMob mode applies MFA to the YJMob dataset across different time intervals, enabling temporal sensitivity studies. Additionally, a sensitivity analysis mode systematically varies key parameters such as the number of seed nodes, the initial infected proportion, and network density, iterating multiple times to generate sufficient data for robust evaluation.
</p>

### parse.py.
<p align="justify">
This code defines a utility function that parses an edge list from a text file and constructs a corresponding NetworkX graph. The parse function reads each line of the input file as a pair of node indices, adds an edge between the nodes using G.add_edge(i, j), and tracks the minimum node index encountered to ensure consistent labeling if needed. It returns the fully constructed graph object. An optional parse_example flag allows the function to be tested directly by supplying a file containing edge data, providing a simple way to validate the parsing and graph construction logic.
</p>

### SIR_examples.py
<p align="justify">
This code simulates and visualizes a suite of Susceptible–Infected–Recovered (SIR) epidemic models under diverse quarantine strategies and behavioral conditions to study their impact on infection dynamics. It compares infection trajectories with and without quarantine, evaluates constant-duration, Gaussian-distributed, recovery-based, and permanent quarantine policies, and analyzes the effects of full versus partial adherence. The implementation also contrasts random versus degree-based seed selection, computes and visualizes Jaccard similarity between social and contact networks, and examines how information awareness influences spread. All strategies are executed through a unified simulation pipeline, with results plotted as infection curves and saved as publication-ready PDF figures. Simulation outputs and parameters are serialized via pickle for reproducibility and later analysis, and can be reloaded to inspect or extend prior results.
</p>

### SIR.py. 
<p align="justify">
This code simulates an extended SIR (Susceptible-Infected-Recovered) model that incorporates quarantine measures, information diffusion, and dynamic social network effects. It manages state transitions using sirs_step, accounting for infection spread (beta), recovery (gamma), and immunity loss (mu), while integrating dynamic quarantines through quarantine_edge_removal and restore_edges, which remove and later restore edges based on node adherence and quarantine duration. The simulation models the spread of information about quarantine measures across a social network using Independent Cascade (IC) or Linear Threshold (LT) models, allowing initially uninformed nodes to become informed and adhere to quarantine rules. Nodes may remain quarantined for fixed or variable periods, with network edges dynamically updated to reflect active quarantines and recoveries. Throughout the simulation, metrics are tracked, including infection counts over time, quarantine statuses (all_quaratines), informed individuals, and dynamic node degrees to capture connectivity changes. Key steps include initializing the social network and node states, propagating infections via sirs_step, dynamically removing and restoring edges during quarantine, and modeling information spread through selected seed nodes. This framework enables detailed observation of how quarantine policies, adherence levels, and information dissemination influence infection dynamics and network structure over time.
</p>

### yjmob_network_creation.py.
<p align="justify">
This code processes movement data to construct time-resolved contact networks and a global social network based on user interactions. It begins by parsing a CSV file (yjmob_sample.csv) and organizing the data into day_data_initial and day_data_final dictionaries, grouping users by 15-minute intervals and storing their location and timestamp information. Collisions, defined as two users occupying the same location within a 2-time-unit window, are detected using Cartesian distance calculations, and each collision is stored as an edge in the corresponding contact network for that interval. For each time slice, these edges form an undirected contact network (contact_networks), while a directed social network is generated from global collision frequencies, retaining only edges above the 75th percentile to focus on significant interactions. The code ensures consistency across all networks by computing the union of nodes and relabeling user IDs to consecutive integers, which is crucial for integration with external tools like Cytoscape. Networks for each interval are exported in GML format, and if ping_cytoscape is enabled, the social network is converted to an undirected format, relabeled to prevent node overlaps, and sent to Cytoscape using py4cytoscape for visualization.
</p>
