# Christopher Dip
# Date: January 1st 2021
# This was based on Aakash N S's course on Deep learning with pytorch
# This script creates an image classifier with the MNIST dataset
# It uses a linear regression model to train the model and stores the weights
# and biases in a .pt file which gets exported to an onnx model

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


dataset = MNIST(root='../../../data/', download=True)

dataset = MNIST(root='../../../data/', train=True,
                transform=transforms.ToTensor())

train_ds, val_ds = random_split(dataset, [50000, 10000])

# Use a dataloader to split the data into batches which are iterable

batch_size = 128

train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item()/len(preds))


class MNISTRegressionModel(nn.Module):
    # Constructor
    def __init__(self):
        # Inherit from nn.Module
        super().__init__()
        # Instantiate with nn.Linear(784, 10)
        self.linear = nn.Linear(784, 10)

    # Forward manipulates every batch
    # reshapes it into the proper size [n-dim, 28*28]
    def forward(self, xb):
        xb = xb.reshape(-1, 784)
        out = self.linear(xb)
        out = F.softmax(out, dim=1)
        return out

    # Training_step finds the loss
    # for every batch
    def training_step(self, batch):
        images, labels = batch
        preds = self(images)
        loss = F.cross_entropy(preds, labels)
        return loss

    # validation_step finds the loss
    # and accuracy for every batch in the
    # validation sample
    def validation_step(self, batch):
        images, labels = batch
        preds = self(images)
        loss = F.cross_entropy(preds, labels)
        acc = accuracy(preds, labels)
        return {'val_loss': loss, 'val_acc': acc}

    # validation_epoch_end merges all the losses
    # and accuracies

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['val_loss'], result['val_acc']))


def evaluate(model, val_loader):
    # list comprehension is simpler
    # For every element batch in val_loader,
    # every element of outputs is the output of model.validation_step(batch)
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    optimizer = opt_func(model.parameters(), lr)
    history = []

    # For every epoch
    for epoch in range(epochs):
        # For every batch in the iterable train_loader
        for batch in train_loader:
            # Generate predictions and calculate loss
            # We need to redefine the class to add training_step method
            loss = model.training_step(batch)
            # Calculate gradient
            loss.backward()
            # Update weights
            optimizer.step()
            # Reset gradients
            optimizer.zero_grad()

        # Validation phase
        # for every batch in val_loader
        # Generate prediction
        # Calculate loss
        # Calculate metrics(accuracy, etc.)
        result = evaluate(model, val_loader)
        # Calculate average validation loss & metrics
        model.epoch_end(epoch, result)

        # Log epoch, loss & metrics for inspection
        history.append(result)

    return history


model = MNISTRegressionModel()

# Load the previous weights, train, and save model

# model.load_state_dict(torch.load("pytorch_model_logistic_regression.pt"))

# history1 = fit(10, 0.025, model, train_loader, val_loader)

# torch.save(model.state_dict(), "pytorch_model_logistic_regression.pt")
