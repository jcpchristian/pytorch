import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
epochs = 30
batch_size = 32
output_size = 10
learning_rate = 0.001

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
    def __init__(self, output_size):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, padding=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=1)
        # self.conv3 = nn.Conv2d(16, 32, kernel_size=5, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(576, 64)
        self.linear2 = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.maxpool(self.relu(self.conv2(x)))
        # x = self.maxpool(self.relu(self.conv3(x)))
        # x = self.relu(x)
        # x = self.maxpool(x)
        # x = self.conv2(x)
        # x = self.relu(x)
        # x = self.maxpool(x)
        # x = self.conv3(x)
        # x = self.relu(x)
        # x = self.maxpool(x)
        x = self.linear2(self.linear1(self.flatten(x)))
        # x = self.linear1(x)
        # x = self.linear2(x)
        return x


model = BuildModel(output_size).to(device)
print(model)

# Loss function & optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

start_time = time.time()
# Train & test loop
train_loss_values = []
test_loss_values = []
for epoch in range(epochs):
    train_loss = 0
    for batch, (X_train, y_train) in enumerate(train_dataloader, 0):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        train_loss += loss
        loss.backward()
        optimizer.step()

    train_loss /= len(train_dataloader)

    model.eval()
    test_loss = 0
    with torch.inference_mode():
        for X_train, y_train in test_dataloader:
            test_pred = model(X_train)
            test_loss += loss_fn(test_pred, y_train)

        test_loss /= len(test_dataloader)

    print(f"Epoch: {epoch}\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}")
    train_loss_values.append(train_loss)
    test_loss_values.append(test_loss)

finish_time = time.time()
print(finish_time - start_time)
