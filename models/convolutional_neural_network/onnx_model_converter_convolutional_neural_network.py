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


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class Cifar10CnnModel(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, xb):
        xb = xb.reshape(280, 280, 4)
        xb = torch.narrow(xb, dim=2, start=3, length=1)
        xb = xb.reshape(1, 1, 280, 280)
        xb = F.avg_pool2d(xb, 10, stride=10)
        xb = xb / 255
        xb = self.conv1(xb)
        xb = F.relu(xb)
        xb = self.conv2(xb)
        xb = F.relu(xb)
        xb = F.max_pool2d(xb, 2)
        xb = self.dropout1(xb)
        xb = torch.flatten(xb, 1)
        xb = self.fc1(xb)
        xb = F.relu(xb)
        xb = self.dropout2(xb)
        xb = self.fc2(xb)
        output = F.softmax(xb, dim=1)
        return output


CNNModel = Cifar10CnnModel()

CNNModel.load_state_dict(torch.load(
    "pytorch_model_convolutional_neural_network.pt"))

dummy_input = torch.zeros([280*280*4])

CNNModel(dummy_input)
torch.onnx.export(CNNModel, dummy_input,
                  'onnx_model_convolutional_neural_network.onnx', verbose=True)
