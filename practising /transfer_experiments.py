# for m in self._population:
#     weights = m.encoder.state_dict()
#     new = Model(m, m._stride)
#     new.encoder.load_state_dict(weights)

import time
import sys
import random

import numpy as np
from matplotlib import pyplot as plt

import torch
from torch import nn
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
# test_loader = DataLoader(test_data, batch_size=30)

    
evolver = nsga2.NSGA2(
    pop_size=40,
    model=TinyFlexyConvAE,
    interval=[1, 8],
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
evolver.reset(
    TinyFlexyConvAE,
    pop_size=40,
    interval=[1, 8],
    bound1=b1,
    bound2=b2
)

# actual evolution
print("- actual evolution..")
evolver.evolve(
    generations=6,
    bound_estimation=False,
    m_prob=0.3
)


conv = evolver.conv_in_time()
conv_final = evolver.avg_convergence()

print(f"Avg population convergence: {round(conv_final, 2)}")
    
_, ax = plt.subplots(figsize=(10, 10))
ax.plot(range(len(conv)), conv, alpha=0.4, color="lightblue")
    
ax.set_xlabel("generations")
ax.set_ylabel("avg. distance from ideal solution")

plt.show()


images = 4
idxs = range(images)
test_subset = Subset(test_data, indices=idxs)
test_loader = DataLoader(test_subset, batch_size=len(idxs))

model, _ = evolver.get_best()

model.eval()
with torch.no_grad():
    for X, _ in test_loader:
        pred = model(X)
        break

_, ax = plt.subplots(nrows=2, ncols=images)
for i in range(images):
    ax[0, i].imshow(X[i].permute(1, 2, 0), cmap="grey")
    ax[0, i].set_title("originals")
    ax[0, i].axis("off")

    ax[1, i].imshow(pred[i].permute(1, 2, 0), cmap="grey")
    ax[1, i].set_title("predictions")
    ax[1, i].axis("off")

plt.show()
    


