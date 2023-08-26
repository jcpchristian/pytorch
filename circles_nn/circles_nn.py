from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# Device agnostic code
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 2
hidden_size = 16
output_size = 1
epochs = 1000
learning_rate = 0.3

# Create dataset using sklearn
X, y = (make_circles(1000,
                     noise=0.03,
                     random_state=42))

''' Visualise, visualise, visualise '''

# Create pd dataframe to visualise data
circles_pd = pd.DataFrame({"X1": X[:,0],
                           "X2": X[:, 1],
                           "Y": y})
print(circles_pd.head(10),"\n")

# Plot scatter chart to visualise data
plt.scatter(x=X[:,0],
            y=X[:,1],
            c=y,
            cmap=plt.cm.RdYlBu)
plt.savefig("circles_nn")

# Check shapes match
print(f"X shape: {X.shape}")
print(f"Y shape: {y.shape}\n")

# Turn our data into Tensors
X_train = torch.from_numpy(X.astype(np.float32))
y_train = torch.from_numpy(y.astype(np.float32))

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}\n")

# Create model class
class build_model(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.layer3(out)

        return out

# Accuracy function
def accuray_fn(y_train, y_pred ):
    correct = (torch.eq(y_train, y_pred).sum().item()) / len(y_pred) * 100
    return correct

# Build model
model = build_model(input_size, output_size, hidden_size).to(device)
print(model)

# Loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs+1):
    # Forward pass & loss
    y_logits = model(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    loss = loss_fn(y_logits, y_train)

    # Backward pass and update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Show loss
    if (epoch) % 100 == 0:
        accuracy = accuray_fn(y_train, y_pred)
        print(f"Epoch: {epoch} | Train loss: {loss:.5f} | Train accuracy: {accuracy}%")