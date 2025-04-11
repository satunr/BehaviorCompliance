import networkx as nx
import json

num_nodes = 10

# Create two graphs
# Call this the social network
G1 = nx.erdos_renyi_graph(n = num_nodes, p = 0.1)

# Call this the contact network
G2 = nx.erdos_renyi_graph(n = num_nodes, p = 0.3)

# Compute positions
pos1 = nx.spring_layout(G1)
pos2 = nx.spring_layout(G2)

# Shift G2 downward
vertical_shift = -10
for node in pos1:
    pos2[node][0] = pos1[node][0]
    pos2[node][1] = pos1[node][1] - vertical_shift

# Combine into one graph
G_combined = nx.Graph()
G_combined.add_edges_from(G1.edges())
G_combined.add_edges_from(G2.edges())

# Add edges from G1 nodes to G2 nodes (example: connect each G1 node to one G2 node)
inter_layer_edges = []
for i in range(0, num_nodes):
    inter_layer_edges.append((i, i+num_nodes))
G_combined.add_edges_from(inter_layer_edges)

# Define a relabeling mapping
mapping = {}
for i in range(num_nodes, 2 * num_nodes):
    mapping[i] = f"{i - num_nodes}'"

# Relabel nodes in place
nx.relabel_nodes(G_combined, mapping, copy=False)

# Assign positions as node attributes
combined_pos = {**pos1, **pos2}
for node, coords in combined_pos.items():
    G_combined.nodes[node]['x'] = float(coords[0])
    G_combined.nodes[node]['y'] = float(coords[1])

# Convert to Cytoscape JSON
cyto_data = nx.cytoscape_data(G_combined)

# Save to file with error handling
try:
    with open("multiplex_network.json", "w") as f:
        json.dump(cyto_data, f)
    print("File 'multiplex_network.json' saved successfully!")
except Exception as e:
    print(f"Error saving file: {e}")


import py4cytoscape as p4c

# Check connection to Cytoscape
p4c.cytoscape_ping()

# Import the network
p4c.create_network_from_networkx(G_combined)

# Apply the preset layout using the x, y attributes
# p4c.set_node_position_mapping('x', 'y')
p4c.layout_network('preset')