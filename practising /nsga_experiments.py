import time
import sys

from matplotlib import pyplot as plt

import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Subset
from architectures import TinyFlexyConvAE, FlexyConvAE
import nsga2

mytransform = transforms.ToTensor()

# help(MNIST) # PIL images

train_data = MNIST("./datasets", download=False, train=True, transform=mytransform)
test_data = MNIST("./datasets", download=False, train=False, transform=mytransform)

train_loader = DataLoader(train_data, batch_size=30)
test_loader = DataLoader(test_data, batch_size=30)


evolution = nsga2.NSGA2(
    pop_size=50,
    model=TinyFlexyConvAE,
    interval=[1, 10],
    data=train_data
)


start = time.time()
evolution.evolve(
    generations=5,
    m_prob=0.3
)
finish = time.time()


print(
    f"it took || {round(finish - start, 4)} sec to go through a run"
)