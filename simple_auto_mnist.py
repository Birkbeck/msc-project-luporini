import sys
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import torch
from torch import nn

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Subset

class AutoEncoder(nn.Module):


    def __init__(self, input, latent, nonlinearity=nn.ReLU):
        super().__init__()
        self._input = input
        self._latent = latent
        self._nonl = nonlinearity

        self.encoder = nn.Sequential(
            nn.Linear(self._input, self._latent),
            self._nonl()
        )

        self.decoder = nn.Sequential(
            nn.Linear(self._latent, self._input),
            nn.Sigmoid()
        )

    def forward(self, data):
        output = self.encoder(data)
        output = self.decoder(output)
        return output
    

mnist_train = MNIST(root="./datasets", download=True, train=True, transform=ToTensor())
mnist_test = MNIST(root="./datasets", download=True, train=False, transform=ToTensor())

dataloader_train = DataLoader(mnist_train, batch_size=30, shuffle=True)
dataloader_test = DataLoader(mnist_test, batch_size=30, shuffle=True)

print("\nunderstanding torch.squeeze/unsqueeze")
for X, _ in dataloader_train:
    print(X.shape)
    X = torch.squeeze(X)
    print(X.shape)
    X = torch.unsqueeze(X, dim=1)
    print(X.shape)
    break
print("\nnow flattening with tensor.reshape")
for X, _ in dataloader_train:
    print(X.shape)
    X = X.reshape((-1, 784))
    print(X.shape)
    break

# model = AutoEncoder(input=784, latent=50)  #MNIST images (28, 28) –> flatten!!!
# loss_fn = nn.MSELoss()
# optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

# epochs = 10
# losses = []
# for e in range(epochs):
#     loss_sum = 0
#     for X, _ in dataloader_train:
#         X = X.reshape((-1, 784))  #MNIST images (28, 28) –> flatten!!!

#         optimiser.zero_grad()
#         pred = model(X)
#         loss = loss_fn(pred, X)
#         loss.backward()
#         optimiser.step()
#         loss_sum += loss.item()

#     losses.append(loss_sum)
#     print(f"loss at {e}th epoch: {loss_sum}")