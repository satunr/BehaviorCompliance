import networkx as nx
import correlated_graphs
import matplotlib.pyplot as plt
import random

#---------------
#
#  Parse a network from a text file
#
#---------------

# F = open('../Social/Freemans_EIES-1_n48.txt', 'r')
# G = nx.Graph()
# Th = 1

# for l in F.readlines():

#     # if len(l.split(' ')) != 3:
#     #     continue

#     u, v, w = l.split(' ')
#     # u, v = l.split(' ')

#     if u == v:
#         continue

#     if not G.has_edge(v, u) and int(w) >= Th:
#         G.add_edge(u, v)

# G = nx.convert_node_labels_to_integers(G, first_label = 0)
# print (G)



parse_example = False

# Function to read edges from a text file and create a NetworkX graph
# NOTE: This ignores the weights for now. The commented code above considers weights
def parse(filename):
    # Initialize an empty graph
    G = nx.Graph()
    
    # Open and read the file
    try:
        with open(filename, 'r') as file:
            # Find minimum node index
            min_node_index = float('inf')
            for line in file:
                # Skip empty lines
                if not line.strip():
                    continue
                
                # Split the line into two integers (i, j)
                i, j = map(int, line.strip().split())

                # Update the minimum node index
                min_node_index = min(min_node_index, i, j)
                
                # Add the edge to the graph
                G.add_edge(i, j)
    
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return None
    except ValueError:
        print("Error: Each line in the file must contain exactly two integers separated by a space.")
        return None
    
    return G
        