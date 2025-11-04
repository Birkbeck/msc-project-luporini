import sys
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import torch
from torch import nn

from architectures import TiniestAutoEncoder
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Subset


mnist_train = MNIST(root="./datasets", download=True, train=True, transform=ToTensor())
mnist_test = MNIST(root="./datasets", download=True, train=False, transform=ToTensor())

dataloader_train = DataLoader(mnist_train, batch_size=30, shuffle=True)
dataloader_test = DataLoader(mnist_test, batch_size=30, shuffle=True)

# print("\nunderstanding torch.squeeze/unsqueeze")
# for X, _ in dataloader_train:
#     print(X.shape)
#     X = torch.squeeze(X)
#     print(X.shape)
#     X = torch.unsqueeze(X, dim=1)
#     print(X.shape)
#     break
# print("\nnow flattening with tensor.reshape")
# for X, _ in dataloader_train:
#     print(X.shape)
#     X = X.reshape((-1, 784))
#     print(X.shape)
#     break

model = TiniestAutoEncoder(input=784, latent=50)  #MNIST images (28, 28) –> flatten!!!
loss_fn = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 10
noise = 0.4
losses = []
for e in range(epochs):
    loss_sum = 0
    i = 0
    for i, (X, _) in enumerate(dataloader_train):
        X = X.reshape((-1, 784))  #MNIST images (28, 28) –> flatten!!!
        noisy = X + noise * torch.randn_like(X)

        optimiser.zero_grad()
        pred = model(noisy)
        loss = loss_fn(pred, X)
        loss.backward()
        optimiser.step()
        loss_sum += loss.item()

    losses.append(loss_sum/(i+1))
    print(f"Avg.loss at {e}th epoch: {loss_sum/(i+1)}")


model.eval()
with torch.no_grad():
    tot_loss = 0
    for i, (X, _) in enumerate(dataloader_test):
        X = X.reshape((-1, 784))
        noisy = X + noise * torch.randn_like(X)
        pred = model(noisy)
        loss = loss_fn(pred, X)
        tot_loss += loss.item()
    
    print(f"\nAvg. test loss: {tot_loss/(i+1)}")

    # test_subset = Subset(mnist_test, indices=range(5))
    # small_loader = DataLoader(test_subset, batch_size=5)
    # for X, _ in small_loader:
    #     X = X.reshape((-1, 784))
    #     pred = model(X)
    #     # print(pred.shape)


    test_subset = Subset(mnist_test, indices=range(5))
    loader_subset = DataLoader(test_subset, batch_size=5)
    for X, _ in loader_subset:
        X = X.reshape((-1, 784))
        noisy = X + noise * torch.randn_like(X)
        pred = model(noisy)
        pred = pred.view(-1, 1, 28, 28)  #does not changes elements in memory!!!

        fig, axes = plt.subplots(3, 5, figsize=(7, 7))

        for i in range(5):
            axes[0, i].imshow(X[i].view(28, 28), cmap="gray")
            axes[0, i].axis("off")
            axes[0, 1].set_title(f"original")

            axes[1, i].imshow(noisy[i].view(28, 28), cmap="gray")
            axes[1, i].axis("off")
            axes[1, i].set_title(f"noisy")

            axes[2, i].imshow(pred[i].view(28, 28), cmap="gray")
            axes[2, i].axis("off")
            axes[2, i].set_title(f"predictions")
        
        plt.show()
