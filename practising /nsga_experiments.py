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
from rainclouds import rainclouds

mytransform = transforms.ToTensor()

# help(MNIST) # PIL images

train_data = MNIST("./datasets", download=False, train=True, transform=mytransform)
test_data = MNIST("./datasets", download=False, train=False, transform=mytransform)

train_loader = DataLoader(train_data, batch_size=30)
test_loader = DataLoader(test_data, batch_size=30)

experiments = 2
seed = 42
convergences = []
for e in range(experiments):
    print(f"\nBeginning experiment {e+1}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    evolver = nsga2.NSGA2(
        pop_size=10,
        model=TinyFlexyConvAE,
        interval=[1, 10],
        data=train_data
    )

    # exploratory runs for empirical min/max
    print("\n- estimating the bounds..")
    evolver.evolve(
        generations=2,
        bound_estimation=True,
        m_prob=0.3
    )

    b1, b2 = evolver.get_bounds()
    # this and just reinitialise evolver!!! will need to
    
    evolver = nsga2.NSGA2(
        pop_size=10,
        model=TinyFlexyConvAE,
        bound1= b1,
        bound2=b2,
        interval=[1, 10],
        data=train_data
    )
    # actual evolution
    print("- actual evolution..")
    evolver.evolve(
        generations=2,
        bound_estimation=False,
        m_prob=0.3
    )

    avg_convergence = evolver.get_avg_convergence()
    convergences.append(avg_convergence)
    print(f"Avg population convergence: {round(avg_convergence, 2)}")
    
    seed += 2


fig, ax = plt.subplots(figsize=(10, 10))
ax.hist(convergences)
plt.show()


# # plot
# go = input("wanna plot?")
# if go:
#     evolution.plot_evolution()