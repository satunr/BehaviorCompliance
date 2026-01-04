import networkx as nx

parse_example = False

# Function to read edges from a text file and create a NetworkX graph
def parse(filename):
    # Initialize an empty graph
    G = nx.Graph()
    
    # Open and read the file
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

    return G
        