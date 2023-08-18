import torch
import torch.nn as nn
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 1
hidden_size = 50
output_size = 1
num_epochs = 10000
batch_size = 16
learning_rate = 0.0001


# dummy dataset
X_list = np.array([i for i in range(100)])
y_list = np.array([i*3 for i in X_list])

X = torch.from_numpy(X_list.astype(np.float32))
y = torch.from_numpy(y_list.astype(np.float32))

X = torch.unsqueeze(X,1)
y = torch.unsqueeze(y,1)

class build_model(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        output = self.layer1(x)
        output = self.layer2(output)


        return output

model = build_model(input_size,output_size).to(device)

# loss function and optimizer
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# training loop
for epoch in range(num_epochs):
    # forward pass & loss
    y_pred = model(X)
    loss = loss_fn(y_pred, y)

    # Backward pass and update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Train loss: {loss:.5f}")

# plot dataset
fig, ax = plt.subplots()
fig.tight_layout()
ax.plot(X, y, X, y_pred.detach().numpy())
# ax.set_xticks([i*25 for i in range(9)])
plt.show()