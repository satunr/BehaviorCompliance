import numpy as np
import time
import socket
import os

# Set random seed for reproducibility
np.random.seed(42)

# Parameters for the computation
matrix_size = 1000  # Size of square matrices (1000x1000)
num_iterations = 1000  # Number of matrix multiplications

# Function to perform intensive matrix multiplication
def compute_matrix_sum():
    total_sum = 0.0
    for _ in range(num_iterations):
        # Generate two random matrices
        A = np.random.rand(matrix_size, matrix_size)
        B = np.random.rand(matrix_size, matrix_size)
        # Perform matrix multiplication
        C = np.dot(A, B)
        # Sum the squares of all elements in the result
        total_sum += np.sum(C ** 2)
    return total_sum

# Start timing
start_time = time.time()

# Run the computation
result = compute_matrix_sum()

# End timing
end_time = time.time()
execution_time = end_time - start_time

# Print results and system info
print(f"Hostname: {socket.gethostname()}")
print(f"Current Working Directory: {os.getcwd()}")
print(f"Python Executable: {os.sys.executable}")
print(f"Execution Time: {execution_time:.2f} seconds")
print(f"Result (sum of squares): {result:.2e}")