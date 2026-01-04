import lt_ic_loss_function as lc
import numpy as np
import parse
import correlated_graphs
import networkx as nx

# Real-world network data
# Specify the filename
filename = 'experiment_data/facebook_network0.txt'
# Create the graph from the file
social_graph = parse.parse(filename)

mapping = {node: node - 1 for node in social_graph.nodes()}

# Relabel the nodes
social_graph = nx.relabel_nodes(social_graph, mapping)

# A list of lists: First of each is I.C., second is L.T.
def write_matrices_to_file(matrix_groups, filename):
    with open(filename, 'w') as f:
        # Clear the file content
        f.truncate(0)
        
        for group_idx, matrices in enumerate(matrix_groups):
            f.write(f"Group {group_idx + 1}\n")
            for i, matrix in enumerate(matrices):
                # Write header
                if i == 0:
                    f.write("I.C. matrix\n")
                else:
                    f.write(f"L.T. matrix (threshold of {i - 1})\n")
                
                # Convert matrix to NumPy array if it isn't already
                matrix = np.array(matrix)
                
                # Write matrix content
                np.savetxt(f, matrix, fmt='%.4f', delimiter=' ')
                f.write("\n")  # Add blank line after each matrix
            f.write("\n\n\n")  # Add blank lines between groups

def parse_matrices():
    """
    Returns:
    List of tuples: [(I.C. matrix, [L.T. matrices])]
    Indexed like this: result[group_index][IC or LT][threshold (if applicable)]

    Example usage: If you want to use I.C. matrix from first graph, it would be result[0][0]
                   If you want to use L.T. matrix (threshold 2) from first graph, it would be result[0][1][2]
    """
    result = []
    current_group = None
    ic_matrix_rows = []
    lt_matrices = []
    current_lt_matrix_rows = []
    is_ic_matrix = False
    is_lt_matrix = False
    expected_columns = None

    try:
        with open("experiment_data/loss_matrices.txt", "r") as file:
            lines = [line.strip() for line in file if line.strip()]

        i = 0
        while i < len(lines):
            line = lines[i]

            # Check for group header
            if line.startswith("Group"):
                # Save previous group's data if exists
                if current_group is not None and ic_matrix_rows:
                    ic_matrix = np.array(ic_matrix_rows)
                    if current_lt_matrix_rows:  # Save any pending L.T. matrix
                        lt_matrix = np.array(current_lt_matrix_rows)
                        if expected_columns is not None and lt_matrix.shape[1] != expected_columns:
                            print(f"Error: L.T. matrix at line {i+1} has {lt_matrix.shape[1]} columns, expected {expected_columns}")
                            return []
                        lt_matrices.append(current_lt_matrix_rows)
                    result.append((ic_matrix, [np.array(m) for m in lt_matrices]))
                
                # Start new group
                current_group = line
                ic_matrix_rows = []
                lt_matrices = []
                current_lt_matrix_rows = []
                is_ic_matrix = False
                is_lt_matrix = False
                expected_columns = None
                i += 1
                continue

            # Check for I.C. matrix label
            if line == "I.C. matrix":
                is_ic_matrix = True
                is_lt_matrix = False
                i += 1
                continue

            # Check for L.T. matrix label
            if line.startswith("L.T. matrix"):
                if is_ic_matrix and ic_matrix_rows:
                    # Finalize I.C. matrix
                    ic_matrix = np.array(ic_matrix_rows)
                    is_ic_matrix = False
                if current_lt_matrix_rows:
                    # Save previous L.T. matrix
                    lt_matrix = np.array(current_lt_matrix_rows)
                    if expected_columns is not None and lt_matrix.shape[1] != expected_columns:
                        print(f"Error: L.T. matrix at line {i+1} has {lt_matrix.shape[1]} columns, expected {expected_columns}")
                        return []
                    lt_matrices.append(current_lt_matrix_rows)
                    current_lt_matrix_rows = []
                is_lt_matrix = True
                i += 1
                continue

            # Parse numerical row
            try:
                values = [float(x) for x in line.split()]
                if expected_columns is None:
                    expected_columns = len(values)
                elif len(values) != expected_columns:
                    print(f"Error: Row at line {i+1} has {len(values)} columns, expected {expected_columns}")
                    return []

                if is_ic_matrix:
                    ic_matrix_rows.append(values)
                elif is_lt_matrix:
                    current_lt_matrix_rows.append(values)
            except ValueError:
                print(f"Warning: Skipping non-numeric line {i+1}: {line}")
                i += 1
                continue

            i += 1

        # Append the last group's data
        if current_group is not None and ic_matrix_rows:
            ic_matrix = np.array(ic_matrix_rows)
            if current_lt_matrix_rows:
                lt_matrix = np.array(current_lt_matrix_rows)
                if lt_matrix.shape[1] != expected_columns:
                    print(f"Error: Last L.T. matrix has {lt_matrix.shape[1]} columns, expected {expected_columns}")
                    return []
                lt_matrices.append(current_lt_matrix_rows)
            result.append((ic_matrix, [np.array(m) for m in lt_matrices]))

    except FileNotFoundError:
        print("Error: 'loss_matrices.txt' not found in the current directory.")
        return []
    except ValueError as e:
        print(f"Error parsing file at line {i+1}: {lines[i] if i < len(lines) else 'EOF'} - {str(e)}")
        return []

    return simplify_matrices(result)  # Return only last rows of matrices for now.

def simplify_matrices(matrices):
    # Only keep the last row for every I.C., L.T. matrix
    simplified = []
    for ic_mat, lt_mats in matrices:
        simplified_ic = ic_mat[-1:]  # Keep only the last row of I.C. matrix
        simplified_lt = [lt_mat[-1:] for lt_mat in lt_mats]  # Keep only the last row of each L.T. matrix
        simplified.append((simplified_ic, simplified_lt))

    return simplified

# Example usage
if parse_matrices():
    matrices = lc.calculate_loss_on_many_networks(social_graph=social_graph, num_networks=10, si=150)
    write_matrices_to_file(matrices, "experiment_data/loss_matrices.txt")

    # # Parse the matrices from the file
    matrices = parse_matrices()

    # Print results for verification
    for idx, (ic_mat, lt_mats) in enumerate(matrices, 1):
        print(f"Group {idx}:")
        # print("I.C. Matrix shape:", ic_mat.shape)
        print("Number of L.T. Matrices:", len(lt_mats))
        print("I.C. Matrix:\n", ic_mat)
        for j, lt_mat in enumerate(lt_mats, 1):
            # print(f"L.T. Matrix with threshold {j - 1} shape:", lt_mat.shape)
            print(f"L.T. Matrix with threshold {j - 1}:\n", lt_mat)
        print()