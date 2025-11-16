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

# exploratory runs for empirical min/max
evolution.evolve(
    generations=6,
    bound_estimation=True,
    m_prob=0.3
)
# actual evolution
start = time.time()
evolution.evolve(
    generations=6,
    bound_estimation=False,
    m_prob=0.3
)
finish = time.time()

# plot
go = input("wanna plot?")
if go:
    evolution.plot_evolution()