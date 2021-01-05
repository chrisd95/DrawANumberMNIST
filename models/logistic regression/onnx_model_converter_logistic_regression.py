# Written by Christopher Dip
# January 4th 2020
# MIT License


from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
import torch.nn as nn


class MNISTRegressionModelModifiedInput(nn.Module):
    # Constructor
    def __init__(self):
        # Inherit from nn.Module
        super().__init__()
        # Instantiate with nn.Linear(784, 10)
        self.linear = nn.Linear(784, 10)

    # Forward manipulates every batch
    # reshapes it into the proper size [n-dim, 28*28]
    def forward(self, xb):
        xb = xb.reshape(280, 280, 4)
        xb = torch.narrow(xb, dim=2, start=3, length=1)
        xb = xb.reshape(1, 1, 280, 280)
        xb = F.avg_pool2d(xb, 10, stride=10)
        xb = xb / 255
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


dataset = MNIST(root='data/', train=True, transform=transforms.ToTensor())

model = MNISTRegressionModelModifiedInput()

model.load_state_dict(torch.load('pytorch_model_logistic_regression.pt'))

dummy_input = torch.zeros([280*280*4])

# This is the export to onnx model
# torch.onnx.export(pytorch_model, dummy_input, 'onnx_model.onnx', verbose=True)


# This is the value
print(torch.max(model.forward(dummy_input), dim=1).values.item())


# This is the indice
print(torch.max(model.forward(dummy_input), dim=1).indices.item())
