import networkx as nx
import correlated_graphs
import matplotlib.pyplot as plt
import random
import SIR

# Function to read edges from a text file and create a NetworkX graph
def parse(filename):
    # Initialize an empty graph
    G = nx.Graph()
    
    # Open and read the file
    try:
        with open(filename, 'r') as file:
            for line in file:
                # Skip empty lines
                if not line.strip():
                    continue
                
                # Split the line into two integers (i, j)
                i, j = map(int, line.strip().split())
                
                # Add the edge to the graph
                G.add_edge(i, j)
    
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return None
    except ValueError:
        print("Error: Each line in the file must contain exactly two integers separated by a space.")
        return None
    
    return G

# Specify the filename
filename = 'contact_network_text.txt'

# Create the graph from the file
base_graph = parse(filename)

if base_graph is not None:

    random.seed(42)

    # Print basic info about the base graph
    print("Base Graph Nodes:", len(base_graph.nodes()))
    print("Base Graph Edges:", len(base_graph.edges()))

    # Compute the correlated graph with k=1 (1-hop)
    correlated_graph_1hop, similarity_matrix_1hop = correlated_graphs.create_w_k_hop_correlation(base_graph, k=1)

    # Compute the correlated graph with k=2 (2-hop)
    correlated_graph_2hop, similarity_matrix_2hop = correlated_graphs.create_w_k_hop_correlation(base_graph, k=2)

    # Print basic info about the correlated graphs
    print("\nCorrelated Graph (1-hop) Nodes:", len(correlated_graph_1hop.nodes()))
    print("Correlated Graph (1-hop) Edges:", len(correlated_graph_1hop.edges()))
    print("Correlated Graph (2-hop) Nodes:", len(correlated_graph_2hop.nodes()))
    print("Correlated Graph (2-hop) Edges:", len(correlated_graph_2hop.edges()))

    # Print a sample of the similarity matrices (to avoid flooding the output)
    print("\nSimilarity Matrix (1-hop, Sample):")
    sample_pairs_1hop = list(similarity_matrix_1hop.items())[:5]  # Show first 5 pairs
    for (i, j), similarity in sample_pairs_1hop:
        print(f"Nodes ({i}, {j}): Jaccard Similarity = {similarity:.3f}")

    print("\nSimilarity Matrix (2-hop, Sample):")
    sample_pairs_2hop = list(similarity_matrix_2hop.items())[:5]  # Show first 5 pairs
    for (i, j), similarity in sample_pairs_2hop:
        print(f"Nodes ({i}, {j}): Jaccard Similarity = {similarity:.3f}")

    # Compute positions for nodes using a spring layout (same for all graphs for consistency)
    pos = nx.spring_layout(base_graph, seed=42)  # Seed for reproducibility

    # First comparison: Base Graph vs. 1-hop Correlated Graph
    plt.figure(figsize=(12, 5))

    # Plot the base graph
    plt.subplot(121)
    nx.draw(base_graph, pos, with_labels=False, node_size=50, node_color='lightblue', edge_color='gray', width=1)
    plt.title("Base Graph")

    # Plot the 1-hop correlated graph
    plt.subplot(122)
    nx.draw(correlated_graph_1hop, pos, with_labels=False, node_size=50, node_color='salmon', edge_color='gray', width=1)
    plt.title("Correlated Graph (1-hop)")

    plt.tight_layout()
    plt.show()

    # Second comparison: Base Graph vs. 2-hop Correlated Graph
    plt.figure(figsize=(12, 5))

    # Plot the base graph again
    plt.subplot(121)
    nx.draw(base_graph, pos, with_labels=False, node_size=50, node_color='lightblue', edge_color='gray', width=1)
    plt.title("Base Graph")

    # Plot the 2-hop correlated graph
    plt.subplot(122)
    nx.draw(correlated_graph_2hop, pos, with_labels=False, node_size=50, node_color='lightgreen', edge_color='gray', width=1)
    plt.title("Correlated Graph (2-hop)")

    plt.tight_layout()
    plt.show()

def simulate_real_network_SIR(contact_network):
    # n = 100
    T = 200
    Repeat = 1

    beta = 0.07  #infection rate
    gamma = 0.04  # recovery rate
    mu = 0.05   # immunity loss
    init = 0.05
    # SIR.Simulate_SIR(len(contact_network.nodes()), T=T, Repeat=Repeat,beta=beta,gamma=gamma,mu=mu,init=init)


