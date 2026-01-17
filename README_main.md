This README covers purpose and functionality of main folder .py files.

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




SIR.py:
Simulation of an SIR (Susceptible, Infected, Recovered) model with additional features, including quarantine measures, information spread, and social network dynamics. 

Main Features::
SIRS Model with State Transitions:
The sirs_step function handles the transition of nodes through the SIR states, considering infection spread (via the beta parameter), recovery (via the gamma parameter), and immunity loss (via the mu parameter).

Quarantine Mechanism:
quarantine_edge_removal quarantines nodes based on certain conditions, removing edges to prevent further spread. It checks if a node is "informed" and "adhering" to quarantine measures before quarantining it.

The restore_edges function restores previously removed edges when the quarantine ends, ensuring a proper simulation of the dynamics.

Node State and Information Spread:
The simulation includes the spread of information about quarantining measures through a social network (social_network), using either Independent Cascade (IC) or Linear Threshold (LT) models.

Initially uninformed nodes can become "informed" during the simulation (at the begin_q time). Once informed, they may follow quarantine measures if adhering to them.

Dynamic Quarantine and Social Network:
The simulation adjusts quarantine statuses dynamically, with users possibly remaining in quarantine for a fixed or variable period. The state of the network (edges, nodes) can change depending on which nodes are quarantining and whether information about quarantining spreads.

Infection Dynamics:
Infection spread is managed using sirs_step, where infected nodes try to infect susceptible neighbors with a probability determined by beta.

Additionally, nodes have a recovery probability (gamma) and possibly lose immunity (mu).

Simulation Data Collection:
During the simulation, the model keeps track of various metrics, including:

Infection count over time (Inf list).

Quarantine status for each node (all_quaratines).

The degree distribution (dynamic_degree), tracking changes in node connectivity.

Quarantine Duration and Edge Restoration:
For nodes that are quarantining, the length of their quarantine period is either fixed (q is an integer) or variable (using a normal distribution). The edges are restored after the quarantine period ends (in q=True or "r" mode).

Key Simulation Steps:
Initialize the social network and set the initial node states.

Propagate infections using the sirs_step method, which incorporates the likelihood of infection spread, recovery, and immunity loss.

Model the quarantine process with dynamic updates:
Edge removals when nodes are quarantined.

Edge restorations either at the end of a fixed quarantine period or immediately upon recovery.

Information spread is modeled via the find_seeds method for determining initial informed nodes, which then spreads through the social network via either the Independent Cascade (IC) or Linear Threshold (LT) model.

Track key metrics for each time step:
Infection dynamics (Inf list).

The state of quarantining (all_quaratines).

The number of informed individuals and quarantined individuals at each time step.

Node degrees, to assess how quarantine and network changes affect the structure over time.





yjmob_network_creation.py:
Processes movement data, computes contact networks, and generates social networks based on user interactions.

Main Steps and Features:
Data Parsing and Preparation:
The code starts by reading movement data from a CSV file (yjmob_sample.csv).

It processes the data into time intervals and stores it in the day_data_initial and day_data_final dictionaries.

day_data_initial: Stores user data grouped by time intervals (15-minute intervals).

day_data_final: Transforms the initial data into a dictionary where each interval has users with their time and location data.

Collisions Detection (for Contact Networks):
A collision occurs when two users are in the same location within a 2-time unit window (checking current and next time point).

The distance between users is computed using the Cartesian distance formula.

If two users’ coordinates match (within the defined time window), a collision is counted, and the pair is stored as an edge in a graph.

Creating Contact Networks:
For each time interval, the collision data is used to create an undirected contact network (contact_network), where nodes represent users and edges represent interactions (collisions).

The contact network for each interval is stored in contact_networks.

Social Network Creation:
A social network is built from the global collision frequency.

The 75th percentile of the collision count is calculated and used to filter out edges (i.e., only keep edges with a collision count greater than or equal to the 75th percentile).

The social network is directed (social_network), and this directed graph is then relabeled for consistency.

Ensuring Consistency Across Networks:
The union of all nodes across all time intervals is computed to ensure that all contact and social networks have the same set of nodes.

The user IDs are relabeled to consecutive integers for consistency across all networks (this is important when integrating these networks with external tools like Cytoscape).

Relabeling and Exporting Networks:
For each interval, the contact and social networks are saved in .gml format using nx.write_gml.

Cytoscape Integration: If ping_cytoscape is set to True, the social network is converted to an undirected graph and relabeled to avoid node overlaps. This union of the contact and social networks is sent to Cytoscape using py4cytoscape's create_network_from_networkx function.
