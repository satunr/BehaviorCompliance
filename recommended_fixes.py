import networkx as nx
import json
import py4cytoscape as p4c

num_nodes = 10

# Create two graphs
# Social network (G1)
G1 = nx.erdos_renyi_graph(n=num_nodes, p=0.1)

# Contact network (G2) - Relabel nodes to 10-19 upfront
G2 = nx.erdos_renyi_graph(n=num_nodes, p=0.3)
mapping_g2 = {i: i + num_nodes for i in range(num_nodes)}  # Shift G2 nodes to 10-19
nx.relabel_nodes(G2, mapping_g2, copy=False)

# Compute positions
pos1 = nx.spring_layout(G1)  # Positions for G1 (0-9)
pos2 = nx.spring_layout(G2)  # Positions for G2 (10-19 initially)

# Shift G2 downward (using pos2 directly)
vertical_shift = -10
for node in pos2:
    pos2[node][1] += vertical_shift  # Shift G2 positions down

# Combine into one graph
G_combined = nx.Graph()
G_combined.add_edges_from(G1.edges())  # G1 edges (0-9)
G_combined.add_edges_from(G2.edges())  # G2 edges (10-19)

# Add inter-layer edges (connect G1 nodes 0-9 to G2 nodes 10-19)
inter_layer_edges = [(i, i + num_nodes) for i in range(num_nodes)]
G_combined.add_edges_from(inter_layer_edges)

# Define a relabeling mapping for G2 nodes (10-19 → "0'"-"9'")
mapping = {i: f"z {i - num_nodes}" for i in range(num_nodes, 2 * num_nodes)}
nx.relabel_nodes(G_combined, mapping, copy=False)

# Assign positions as node attributes
combined_pos = {**pos1, **pos2}
for node in G_combined.nodes():  # Use all nodes to avoid missing any
    if node in combined_pos:  # Only assign if position exists
        G_combined.nodes[node]['x'] = float(combined_pos[node][0])
        G_combined.nodes[node]['y'] = float(combined_pos[node][1])

# Convert to Cytoscape JSON
cyto_data = nx.cytoscape_data(G_combined)

# Save to file with error handling
try:
    with open("multiplex_network.json", "w") as f:
        json.dump(cyto_data, f)
    print("File 'multiplex_network.json' saved successfully!")
except Exception as e:
    print(f"Error saving file: {e}")

# Check connection to Cytoscape
p4c.cytoscape_ping()

# Import the network
p4c.create_network_from_networkx(G_combined)

# Apply the preset layout using the x, y attributes
p4c.layout_network('preset')

print("Network imported and laid out in Cytoscape!")