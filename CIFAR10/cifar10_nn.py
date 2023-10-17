import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Grayscale
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
epochs = 3
batch_size = 64
input_size = 1*32*32
output_size = 10
hidden_size = 128
learning_rate = 0.01

# Get train & test datasets
train_set = CIFAR10(root='data',
                  train=True,
                  download=True,
                  transform=ToTensor()
                  )

test_set = CIFAR10(root='data',
                   train=False,
                   download=True,
                   transform=ToTensor()
                   )

len(train_set.data)

# Get class names
class_names = train_set.classes
print(class_names)

# Visualise data
figure = plt.figure(figsize=(6, 6))
cols, rows = 4, 4
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_set), size=(1,)).item()
    img, label = train_set[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(class_names[label])
    plt.axis("off")
    plt.imshow(img.permute(1, 2, 0), cmap="gray")
plt.savefig("cifar10.png")

image, label = train_set[0]
print(image.shape)

# Prepare train & test dataloader
train_dataloader = DataLoader(dataset=train_set,
                              batch_size=batch_size,
                              shuffle=False)

test_dataloader = DataLoader(dataset=train_set,
                             batch_size=batch_size,
                             shuffle=False)

print(f"Dataloaders: {train_dataloader, test_dataloader}")
print(f"Length of train dataloader: {len(train_dataloader)} batches of {batch_size}")
print(f"Length of test dataloader: {len(test_dataloader)} batches of {batch_size}")

train_features_batch, train_labels_batch = next(iter(train_dataloader))
print(train_features_batch.shape)
print(train_labels_batch.shape)

# Build model class
class BuildModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.flatten(x)
        out = self.layer1(out)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.relu2(out)
        out = self.layer3(out)
        return out

model = BuildModel(input_size, output_size).to(device)
print(model)

# Loss function & optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Traind & test loop
train_loss_values = []
test_loss_values = []
for epoch in range(epochs):
    train_loss = 0
    for batch, (X_train, y_train) in enumerate(train_dataloader, 0):
        model.train()
        optimizer.zero_grad()
        X_train = Grayscale()(X_train)
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        train_loss += loss
        loss.backward()
        optimizer.step()
        if batch % (batch_size*5) == 0:
            print(f"Looked at {batch * len(X_train)}/{len(train_dataloader.dataset)} samples")

    train_loss /= len(train_dataloader)

    model.eval()
    test_loss=0
    with torch.inference_mode():
        for X_train, y_train in test_dataloader:
            X_train = Grayscale()(X_train)
            test_pred = model(X_train)
            test_loss += loss_fn(test_pred, y_train)

        test_loss /= len(test_dataloader)

    print(f"Train loss: {train_loss:.5f} | Test loss: {test_loss:.5f}")
    train_loss_values.append(train_loss)
    test_loss_values.append(test_loss)

# Plot loss
plt.plot(test_loss_values)
# plt.show(test_loss_values)