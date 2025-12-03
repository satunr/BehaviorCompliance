import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ---------------------------------------
# Load and prepare data
# ---------------------------------------
df = pd.read_csv("ml_project/spambase.data", header=None)

# 57 features, last column = label
X = df.iloc[:, :-1].values.astype('float32')
y = df.iloc[:, -1].values.astype('float32')

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# DataLoaders
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)


# ---------------------------------------
# Neural Network Model
# ---------------------------------------
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
            nn.Sigmoid()  # binary probability
        )

    def forward(self, x):
        return self.net(x)


# Initialize model, loss, optimizer
model = SpamClassifier()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ---------------------------------------
# Training Loop with Loss Tracking
# ---------------------------------------
epochs = 20
train_losses = []
val_losses = []

for epoch in range(epochs):
    # ---- TRAINING ----
    model.train()
    total_loss = 0.0

    for Xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(Xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # ---- VALIDATION ----
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for Xb, yb in test_loader:
            preds = model(Xb)
            loss = criterion(preds, yb)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(test_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")


# ---------------------------------------
# Plot Training vs Validation Loss Curve
# ---------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss Curve")
plt.legend()
plt.grid(True)
plt.show()


# ---------------------------------------
# Final Evaluation
# ---------------------------------------
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for Xb, yb in test_loader:
        preds = (model(Xb) > 0.5).float()
        correct += (preds == yb).sum().item()
        total += yb.size(0)

print(f"Test Accuracy: {correct/total:.4f}")


# ---------------------------------------
# Confusion Components
# ---------------------------------------
tp = tn = fp = fn = 0

with torch.no_grad():
    for Xb, yb in test_loader:
        preds = (model(Xb) > 0.5).float()

        tp += ((preds == 1) & (yb == 1)).sum().item()
        tn += ((preds == 0) & (yb == 0)).sum().item()
        fp += ((preds == 1) & (yb == 0)).sum().item()
        fn += ((preds == 0) & (yb == 1)).sum().item()

print(f"True Positives: {tp}, True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}")

total_conf = tp + tn + fp + fn
print(f"TP%: {tp/total_conf:.4f}, TN%: {tn/total_conf:.4f}, FP%: {fp/total_conf:.4f}, FN%: {fn/total_conf:.4f}")
