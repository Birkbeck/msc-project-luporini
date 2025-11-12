import time
import sys

from matplotlib import pyplot as plt

import torch
from torch import nn
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Subset

data_train = FashionMNIST(root="./datasets", download=True, train=True, transform=ToTensor())
data_test = FashionMNIST(root="./datasets", download=True, train=False, transform=ToTensor())

# dataloader_train = DataLoader(data_train, batch_size=30)
# dataloader_test = DataLoader(data_test, batch_size=30)

number = 5

subset_train = Subset(data_train, indices=range(number)) # expects an iterable
# subset_train[i] >>> tuple (img, label)
# print(subset_train[1][0].squeeze().shape)

fig, axes = plt.subplots(nrows=1, ncols=number) # nrows=1 gives a 1D array
for i in range(number):
    axes[i].imshow(subset_train[i][0].squeeze(), cmap="gray")
    axes[i].axis("off")

plt.show()