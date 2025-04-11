import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
G.add_node(1, label='1_0')
G.add_node(2, label='2_0')
G.add_node(3, label='1_1')
G.add_edge(1, 2)

# Try to obtain position of a node
pos = nx.spring_layout(G, seed=42)  # Seed for reproducibility

# Adjust position of node 4 to be 0.5 units below node 1
x1, y1 = pos[1]  # Get position of node 1
pos[3] = (x1, y1 - 0.5)  # Place node 2 below node 1

# Draw the graph
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500)
plt.show()

nx.write_gml(G, "test_graph.gml")

# # Step 2: Define custom positions
# pos = {}
# pos['A'] = (0, 0)    # Place 'A' at the origin
# distance = 1         # Distance to place 'B' below 'A'
# pos['B'] = (0, -distance)  # Same x, y decreased by 1

# # Step 3: Draw the graph with custom positions
# nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=12)