import networkx as nx
import matplotlib.pyplot as plt


# Create a Graph
#	◦	Create an undirected graph and add 5 nodes labeled 1 to 5.
#	◦	Add edges to make it a cycle graph (1-2-3-4-5-1).
G = nx.Graph()
G.add_nodes_from([1, 2, 3, 4, 5])
G.add_edges_from([(1,2), (2,3), (3,4), (4,5), (5,1)])


#In the same graph, add and Remove Nodes/Edges
#	◦	Add a new node 6 and connect it to nodes 2 and 4.
#	◦	Remove the edge between nodes 3 and 4.
G.add_node(6)
G.add_edges_from([(2,6), (4,6)])
G.remove_edges_from([(3,4)])


#In the same graph, check graph properties
#	◦	Print the number of nodes and edges.
#	◦	Check whether the graph is connected.
nodes = 0
for i in G.nodes():
    nodes = nodes + 1

edges = 0
for i in G.edges():
    edges = edges + 1

print("Number of nodes: ", nodes, "Number of edges: ", edges)

if nx.is_connected(G) == True:
    print("G is connected")


#Compute node degrees
#	◦	Print the degree of each node.
for i in G.nodes():
    print(f"Degree of node {i}: ", G.degree(i))


#Find Shortest Paths
#	◦	Compute the shortest path from node 1 to 5 using Dijkstra’s algorithm.
shortest_path = nx.dijkstra_path(G, 1, 5, weight='weight')
print(shortest_path)


#Find Connected Components
#	◦	If the graph is disconnected, print the number of connected components.
connected_components = nx.number_connected_components(G)
print(f"Number of connected components: {connected_components}")


#Convert Between Graph Types
#	◦	Convert the graph to a directed graph and print its adjacency list.
directed_G = nx.DiGraph(G)

print("Adjacency list:")
for node in G.adj:
    neighbors = list(G.adj[node].keys())
    print(f"{node}: {neighbors}")


#Visualize the Graph
#	◦	Use matplotlib to draw the graph with node labels.
nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=12)
plt.show()


#Create a Random Graph
#	◦	Generate an Erdős–Rényi random graph with 10 nodes and an edge probability of 0.3.
erdos = nx.erdos_renyi_graph(10, 0.3, seed=None, directed=False)

nx.draw(erdos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=12)
plt.show()


#Find Clustering Coefficients
#	•	Compute the clustering coefficient of each node and the average clustering coefficient.
clustering = nx.clustering(erdos)

print("Local clustering coefficients:")
for node, coeff in clustering.items():
    print(f"Node {node}: {coeff}")

print(f"Clustering of whole graph: {clustering}")