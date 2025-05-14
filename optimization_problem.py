import networkx as nx
import matplotlib.pyplot as plt
import parse
import correlated_graphs
import SIR
import numpy as np
import py4cytoscape as p4c
from scipy.optimize import minimize  # Import scipy.optimize
import random
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import time


ping_cytoscape = False

# Takes in list of arrays from SIR w/ I.C. simulations and returns the average array
def average_and_normalize(arrays):    
    # Convert list of arrays to a single NumPy array and compute mean along axis 0
    try:
        stacked = np.stack(arrays)
    except ValueError:
        raise ValueError("All arrays must have the same shape")
    
    avg = np.mean(stacked, axis=0)
    
    # Find the two smallest values (unique values)
    flat_avg = avg.flatten()
    unique_vals = np.unique(flat_avg)
    if len(unique_vals) < 2:
        return "Not enough unique values to determine average simulation"
    else:
        smallest_vals = np.sort(unique_vals)[:2]  # Take the two smallest
    
    # Set all instances of the two smallest values to 0
    for val in smallest_vals:
        avg[avg == val] = 0
    
    # Normalize: 0 -> 0, non-zero -> 1
    normalized = np.where(avg != 0, 1, 0)
    
    return normalized

# Lazy way of parsing the loss data from the HPC, if used
def parse_loss_data(file_path):
    thresholds = []
    losses = []
    
    with open(file_path, 'r') as file:
        for line in file:
            # Remove leading/trailing whitespace
            line = line.strip()
            if line.startswith('loss with threshold of'):
                # Split the line into parts
                parts = line.split(':')
                if len(parts) == 2:
                    # Extract threshold (between 'of' and ':')
                    threshold_part = parts[0].split('of')[1].strip()
                    # Extract loss
                    loss_part = parts[1].strip()
                    try:
                        threshold = int(threshold_part)
                        loss = float(loss_part)
                        thresholds.append(threshold)
                        losses.append(loss)
                    except ValueError:
                        continue  # Skip lines that can't be converted to numbers
    
    return thresholds, losses


#---------
#
#  Optimization problem: Minimize loss between I.C., L.T. models, 
#    to arrive at a deterministic construction of unknown social network
#    Assumption: Threshold value is the same for all nodes
#    f: Z -> Z, and we want f(X*) approx A in Z, where A is the observed I.C. results
#      Learnable parameter: Tau - Threshold for L.T. simulation.
#
#---------


# Parameters
T = 100
Repeat = 8
beta = 0.09
gamma = 0.07
mu = 0.11
init = 0.03
num_graphs = 10
tau_range = range(2, 11)

# Loss function for dataset creation
def loss_function(tau, contact_network, results):
    tau = int(round(float(tau)))
    target = SIR.Simulate_SIR(
        contact_network=contact_network, social_network=None,
        T=T, Repeat=Repeat, beta=beta, gamma=gamma, mu=mu, init=init,
        average_data=False, q=True, allow_restoration=True, save_all=True, lt_threshold=tau
    )[3]
    target = np.array(target, dtype=float)
    target = np.where(target != 0, 1, 0)

    results = np.array(results, dtype=float)
    results = np.where(results != 0, 1, 0)

    loss = np.sum(np.abs(target - results))
    return loss

# Generate dataset
dataset = []
for i in range(num_graphs):
    G = nx.erdos_renyi_graph(n=random.randint(25, 100), p=random.uniform(0.1, 0.5))

    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    degrees = np.array([d for _, d in G.degree()])
    node_features = torch.FloatTensor(degrees / degrees.max()).reshape(-1, 1)
    # Simulate SIR w/ I.C. model on the graph
    cur_result = SIR.Simulate_SIR(
        contact_network=G, social_network=None, T=T, Repeat=Repeat, beta=beta, gamma=gamma, mu=mu, init=init,
        average_data=False, q=True, allow_restoration=True, save_all=True, lt_threshold=None
    )[3]
    losses = []
    for tau in tau_range:
        loss = loss_function(tau, G, cur_result)
        losses.append(loss)
    optimal_tau = 2 + np.argmin(losses)
    data = Data(x=node_features, edge_index=edge_index, y=torch.tensor([optimal_tau], dtype=torch.float))
    dataset.append(data)

# Split dataset
train_dataset = dataset[:8]
test_dataset = dataset[8:]
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

#----------
#
#  Define GNN model
#
#----------

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(1, 16)  # Input: 1 feature (degree), output: 16
        self.conv2 = GCNConv(16, 16)
        self.fc = torch.nn.Linear(16, 1)  # Output: 1 value (tau)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = x.mean(dim=0)  # Global mean pooling
        x = self.fc(x)
        return x

# Initialize model, optimizer, and loss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

# Training loop
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Evaluation
def test(loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y)
            total_loss += loss.item()
    return total_loss / len(loader)

train_losses = []
test_losses = []

# Train for some epochs
for epoch in range(100):
    train_loss = train()
    test_loss = test(test_loader)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

# Plotting the losses
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.show()

#----------
#
#  Use Case: Predict tau for a real-world graph
#
#----------

# Specify the filename
filename = 'contact_network_text.txt'

# Start timer
start_time = time.time()

# Create the graph from the file
real_graph = parse.parse(filename)
# Relabel the nodes
mapping = {node: node - 1 for node in real_graph.nodes()}
# Relabel the nodes
real_graph = nx.relabel_nodes(real_graph, mapping)

#----------
#
#  Use the trained model to predict tau for a real-world graph
#
#----------

# Use neural network to predict tau for the real graph
real_edge_index = torch.tensor(list(real_graph.edges)).t().contiguous()
real_degrees = np.array([d for _, d in real_graph.degree()])
real_node_features = torch.FloatTensor(real_degrees / real_degrees.max()).reshape(-1, 1)
real_data = Data(x=real_node_features, edge_index=real_edge_index)
real_data = real_data.to(device)
model.eval()
with torch.no_grad():
    predicted_tau = model(real_data).item()
predicted_tau = int(round(predicted_tau))
print(f"Predicted tau for the real graph: {predicted_tau}")

# End timer
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.2f} seconds")

#----------
#
#  Find true tau that minimizes loss using iterative method
#
#----------

# Iterative method for minimizing loss
def iterative_loss_minimization(contact_network, results, initial_tau=2, max_iterations=10):
    losses = []
    for _ in range(initial_tau, max_iterations):
        loss = loss_function(tau, contact_network, results)
        if loss == 0:
            break
        losses.append(loss)
    # Return tau that minimizes loss
    return losses, initial_tau + np.argmin(losses)

# Start timer for iterative method
start_time_iterative = time.time()

avg_ic_results = []
print("Running I.C. average simulation on real graph...")
for i in range(Repeat):
# Simulate SIR model on the real graph
    real_ic = SIR.Simulate_SIR(
        contact_network=real_graph, social_network=None, T=T, Repeat=Repeat, beta=beta, gamma=gamma, mu=mu, init=init,
        average_data=False, q=True, allow_restoration=True, save_all=True, lt_threshold=None
    )[3]
    avg_ic_results.append(real_ic)
real_ic = average_and_normalize(avg_ic_results)

# Use the iterative method to find tau on real graph
iterative_results = iterative_loss_minimization(contact_network=real_graph, results=real_ic)
losses = iterative_results[0]
actual_tau = iterative_results[1]  

# End timer for iterative method
end_time_iterative = time.time()
execution_time_iterative = end_time_iterative - start_time_iterative
print(f"Execution time for iterative method: {execution_time_iterative:.2f} seconds")
print(f"Actual tau using iterative method: {actual_tau}")

x_range = [i for i in range(2, 15)]
# Plotting the loss function
plt.plot(x_range[:len(losses)], losses, marker='o')
plt.xlabel('Tau')
plt.ylabel('Loss')
plt.title('Loss Function for Different Tau Values')
plt.grid()
plt.show()