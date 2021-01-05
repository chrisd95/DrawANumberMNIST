# Christopher Dip
# January 4th 2020
# MIT License


import time
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision.transforms as transforms
import torch
import torchvision
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


class MNISTDeepNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, output_size)

    def forward(self, xb):
        # Contrary to the logistic regression model
        # The forward needs to take in the previous layer and convert the rest
        # To a linear matrix
        xb = xb.view(xb.size(0), -1)
        # Forward pass to the hidden layer
        out = self.input_layer(xb)
        # Apply activation function
        out = F.relu(out)
        # Forward pass to the output layer
        out = self.hidden_layer(out)
        # Apply a softmax to get probabilities
        out = F.softmax(out, dim=1)
        return out

    def training_step(self, batch):
        # Get the images and labels from the batch
        images, labels = batch
        # Apply the model to the images
        out = self(images)
        # Get the loss which is the probability
        # Of the negative log (predicted number - 1)
        # i.e. softmax
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        # Get the images and labels from the batch
        images, labels = batch
        # Apply the model to the images and get output
        out = self(images)
        # Calculate the loss
        loss = F.cross_entropy(out, labels)

        # The prediction is the max probability index
        _, preds = torch.max(out, dim=1)
        # Sum the number of times the prediction is right
        # And divide it by the total number of predictions
        # To get the accuracy
        acc = torch.tensor(torch.sum(preds == labels).item()/len(preds))

        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['val_loss'], result['val_acc']))


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


device = get_default_device()


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


def evaluate(model, val_loader):
    """Outputs are generated as a list comprehension after getting
    evaluated iteratively by validation_step"""
    outputs = [model.validation_step(batch) for batch in val_loader]
    # We then pass it to validation_epoch_end
    # Which returns the average of loss, acc for the epoch
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    """Train the model using gradient descent"""
    history = []
    # Use SGD (weight -= gradient descent * lr)
    optimizer = opt_func(model.parameters(), lr)

    # For every epoch
    for epoch in range(epochs):
        # For every batch
        for batch in train_loader:
            # Calculate the loss
            loss = model.training_step(batch)
            # Find gradient
            loss.backward()
            # Remove SGD (weight -= gradient descent * lr)
            optimizer.step()
            # Reset the gradients
            optimizer.zero_grad()

        # Evaluate the model's weights with the validation set
        result = evaluate(model, val_loader)
        # Print the epoch's results
        model.epoch_end(epoch, result)
        history.append(result)
    return history


# Use MNIST dataset
dataset = MNIST(root='../data/', download=True)

dataset = MNIST(root='../data/', train=True, transform=transforms.ToTensor())

# dataset = MNIST(root='../data/', train=True,
#                 transform=transforms.Compose([
#                     transforms.RandomAffine(
#                         degrees=30, translate=(0.5, 0.5), scale=(0.25, 1),
#                         shear=(-30, 30, -30, 30)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
#                 ]))

# Split the dataset randomly between training and validation sets
train_ds, val_ds = random_split(dataset, [50000, 10000])

# Create the dataloaders
batch_size = 512
train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size, shuffle=True)
# Send to GPU
train_loader = DeviceDataLoader(train_loader, device)
val_loader = DeviceDataLoader(val_loader, device)


# Create the model
deepModel = MNISTDeepNN(784, 32, 10)
# Send to GPU
to_device(deepModel, device)

# deepModel.load_state_dict(torch.load("pytorch_model_deep_neural_network.pt"))

history = fit(45, 1, deepModel, train_loader, val_loader)

torch.save(deepModel.state_dict(), "pytorch_model_deep_neural_network.pt")
