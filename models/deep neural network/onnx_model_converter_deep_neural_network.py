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


class MNISTDeepNNModifiedInput(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, output_size)

    def forward(self, xb):
        xb = xb.reshape(280, 280, 4)
        xb = torch.narrow(xb, dim=2, start=3, length=1)
        xb = xb.reshape(1, 1, 280, 280)
        xb = F.avg_pool2d(xb, 10, stride=10)
        xb = xb / 255
        xb = xb.reshape(-1, 784)
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


deepModel = MNISTDeepNNModifiedInput(784, 32, 10)

deepModel.load_state_dict(torch.load("pytorch_model_deep_neural_network.pt"))

dummy_input = torch.zeros([280*280*4])

deepModel(dummy_input)
torch.onnx.export(deepModel, dummy_input,
                  'onnx_model_deep_neural_network.onnx', verbose=True)
