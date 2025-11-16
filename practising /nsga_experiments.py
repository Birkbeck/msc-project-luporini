import time
import sys
import random

import numpy as np
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

experiments = 20
seed = 42
avg_convergence = []
for e in range(experiments):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
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
    evolution.evolve(
        generations=6,
        bound_estimation=False,
        m_prob=0.3
    )

    avg_convergence.append(evolution.get_avg_convergence())
    seed += 2



fig, ax = plt.subplots(figsize=(10, 10))
ax.hist()
plt.show()


# # plot
# go = input("wanna plot?")
# if go:
#     evolution.plot_evolution()