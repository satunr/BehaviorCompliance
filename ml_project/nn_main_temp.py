import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load and prepare data (64% train, 16% val, 20% test)
df = pd.read_csv("ml_project/spambase.data", header=None)

X = df.iloc[:, :-1].values.astype('float32')
y = df.iloc[:, -1].values.astype('float32')

# Separate test set (20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Split remaining into train (64%) and validation (16%)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
)

# Scale features using training statistics
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# Convert to torch tensors
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val_t   = torch.tensor(X_val_scaled,   dtype=torch.float32)
y_val_t   = torch.tensor(y_val,   dtype=torch.float32).unsqueeze(1)
X_test_t  = torch.tensor(X_test_scaled,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1)

# DataLoaders
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_t,   y_val_t),   batch_size=64, shuffle=False)
test_loader  = DataLoader(TensorDataset(X_test_t,  y_test_t),  batch_size=64, shuffle=False)

# Loss function shared by both neural models
criterion = nn.BCELoss()

# Approach 1: Gaussian Naive Bayes
class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes, class_counts = np.unique(y, return_counts=True)
        self.prior = class_counts / len(y)
        self.mean  = np.array([X[y == c].mean(axis=0) for c in self.classes])
        self.var   = np.array([X[y == c].var(axis=0) + 1e-9 for c in self.classes])

    def predict(self, X):
        log_posteriors = []
        for i, c in enumerate(self.classes):
            log_prior      = np.log(self.prior[i])
            log_likelihood = -0.5 * np.sum(np.log(2. * np.pi * self.var[i])) \
                           - 0.5 * np.sum(((X - self.mean[i]) ** 2) / self.var[i], axis=1)
            log_posteriors.append(log_prior + log_likelihood)
        log_posteriors = np.vstack(log_posteriors).T          # (n_samples, n_classes)
        return self.classes[np.argmax(log_posteriors, axis=1)]

print("\n=== Gaussian Naive Bayes ===")
gnb = GaussianNaiveBayes()
gnb.fit(X_train_scaled, y_train)
y_pred_gnb = gnb.predict(X_test_scaled)

accuracy_gnb = np.mean(y_pred_gnb == y_test)
tp_gnb = np.sum((y_pred_gnb == 1) & (y_test == 1))
tn_gnb = np.sum((y_pred_gnb == 0) & (y_test == 0))
fp_gnb = np.sum((y_pred_gnb == 1) & (y_test == 0))
fn_gnb = np.sum((y_pred_gnb == 0) & (y_test == 1))

print(f"GNB Test Accuracy: {accuracy_gnb:.4f}")
print(f"GNB - TP:{tp_gnb:3d}  TN:{tn_gnb:3d}  FP:{fp_gnb:3d}  FN:{fn_gnb:3d}")

# Approach 2: Logistic Regression
class LogisticReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(57, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

print("\n=== Training Logistic Regression (100 epochs, lr=0.005) ===")
model_log = LogisticReg()
optimizer_log = torch.optim.Adam(model_log.parameters(), lr=0.005)

train_losses_log = []
val_losses_log   = []

for epoch in range(100):
    # ---- Train ----
    model_log.train()
    train_loss = 0.0
    for Xb, yb in train_loader:
        optimizer_log.zero_grad()
        preds = model_log(Xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer_log.step()
        train_loss += loss.item()
    train_losses_log.append(train_loss / len(train_loader))

    # ---- Validation ----
    model_log.eval()
    val_loss = 0.0
    with torch.no_grad():
        for Xb, yb in val_loader:
            preds = model_log(Xb)
            val_loss += criterion(preds, yb).item()
    val_losses_log.append(val_loss / len(val_loader))

    if (epoch + 1) % 20 == 0 or epoch == 99:
        print(f"Epoch {epoch+1:3d}/100 - Train Loss: {train_losses_log[-1]:.4f}  Val Loss: {val_losses_log[-1]:.4f}")

# Plot convergence
plt.figure(figsize=(8, 5))
plt.plot(train_losses_log, label="Training Loss")
plt.plot(val_losses_log,   label="Validation Loss")
plt.title("Logistic Regression - Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.legend()
plt.grid(True)
plt.show()

# Test evaluation
model_log.eval()
tp = tn = fp = fn = correct = total = 0
with torch.no_grad():
    for Xb, yb in test_loader:
        preds = (model_log(Xb) > 0.5).float()
        correct += (preds == yb).sum().item()
        total   += yb.size(0)
        tp += ((preds == 1) & (yb == 1)).sum().item()
        tn += ((preds == 0) & (yb == 0)).sum().item()
        fp += ((preds == 1) & (yb == 0)).sum().item()
        fn += ((preds == 0) & (yb == 1)).sum().item()

print(f"Logistic Regression Test Accuracy: {correct/total:.4f}")
print(f"Logistic - TP:{tp:3d}  TN:{tn:3d}  FP:{fp:3d}  FN:{fn:3d}")

# Approach 3: Multi-Layer Perceptron
class SpamClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(57, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

print("\n=== Training Multi-Layer Perceptron (50 epochs, lr=0.001) ===")
model_mlp = SpamClassifier()
optimizer_mlp = torch.optim.Adam(model_mlp.parameters(), lr=0.001)

train_losses_mlp = []
val_losses_mlp   = []

for epoch in range(50):
    # ---- Train ----
    model_mlp.train()
    train_loss = 0.0
    for Xb, yb in train_loader:
        optimizer_mlp.zero_grad()
        preds = model_mlp(Xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer_mlp.step()
        train_loss += loss.item()
    train_losses_mlp.append(train_loss / len(train_loader))

    # ---- Validation ----
    model_mlp.eval()
    val_loss = 0.0
    with torch.no_grad():
        for Xb, yb in val_loader:
            preds = model_mlp(Xb)
            val_loss += criterion(preds, yb).item()
    val_losses_mlp.append(val_loss / len(val_loader))

    if (epoch + 1) % 10 == 0 or epoch == 49:
        print(f"Epoch {epoch+1:2d}/50 - Train Loss: {train_losses_mlp[-1]:.4f}  Val Loss: {val_losses_mlp[-1]:.4f}")

# Plot convergence
plt.figure(figsize=(8, 5))
plt.plot(train_losses_mlp, label="Training Loss")
plt.plot(val_losses_mlp,   label="Validation Loss")
plt.title("MLP - Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.legend()
plt.grid(True)
plt.show()

# Test evaluation
model_mlp.eval()
tp = tn = fp = fn = correct = total = 0
with torch.no_grad():
    for Xb, yb in test_loader:
        preds = (model_mlp(Xb) > 0.5).float()
        correct += (preds == yb).sum().item()
        total   += yb.size(0)
        tp += ((preds == 1) & (yb == 1)).sum().item()
        tn += ((preds == 0) & (yb == 0)).sum().item()
        fp += ((preds == 1) & (yb == 0)).sum().item()
        fn += ((preds == 0) & (yb == 1)).sum().item()

print(f"MLP Test Accuracy: {correct/total:.4f}")
print(f"MLP - TP:{tp:3d}  TN:{tn:3d}  FP:{fp:3d}  FN:{fn:3d}")