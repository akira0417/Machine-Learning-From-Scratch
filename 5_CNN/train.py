import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import SimpleCNN


# Dataset
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


# DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)


# Model
model = SimpleCNN()


# Loss
criterion = nn.CrossEntropyLoss()


# Optimizer
optimizer = optim.SGD(
    model.parameters(),
    lr=0.1
)


# Training
epochs = 5

train_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(epochs):

    # Training mode
    model.train()

    running_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        train_correct += (
            predictions == labels
        ).sum().item()

        train_total += labels.size(0)

    train_loss = (
        running_loss / len(train_loader)
    )

    train_accuracy = (
        train_correct / train_total
    )


    # Evaluation mode
    model.eval()

    test_correct = 0
    test_total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            test_correct += (
                predictions == labels
            ).sum().item()

            test_total += labels.size(0)

    test_accuracy = (
        test_correct / test_total
    )

    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    print(
        f"Epoch {epoch + 1}/{epochs}, "
        f"Loss: {train_loss:.4f}, "
        f"Train Acc: {train_accuracy:.4f}, "
        f"Test Acc: {test_accuracy:.4f}"
    )

plt.plot(
    range(1, epochs + 1),
    train_losses
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")

plt.show()

plt.plot(
    range(1, epochs + 1),
    train_accuracies,
    label="Train"
)

plt.plot(
    range(1, epochs + 1),
    test_accuracies,
    label="Test"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Train vs Test Accuracy")

plt.legend()

plt.show()