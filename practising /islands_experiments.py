import time
import sys

from matplotlib import pyplot as plt

import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Subset
from architectures import FlexyConvAE
import islands

mytransform = transforms.ToTensor()

# help(MNIST) # PIL images

train_data = MNIST("./datasets", download=False, train=True, transform=mytransform)
test_data = MNIST("./datasets", download=False, train=False, transform=mytransform)

train_loader = DataLoader(train_data, batch_size=30)
test_loader = DataLoader(test_data, batch_size=30)

########################################
########################################
# print(len(train_data))
# print(train_data[0])

# images = 5
# fig, axes = plt.subplots(nrows=1, ncols=images, figsize=(5, 5))

# for i in range(images):
#     img, _ = train_data[i]
#     axes[i].imshow(img, cmap="gray")

# plt.show()
########################################
########################################

start = time.time()
evolution = islands.Islands(
    pop_size=20,
    model=FlexyConvAE,
    data=train_data
)
finish = time.time()
print(
    f"it took || {round(finish - start, 4)} sec || {round((finish - start)/60)} min"
)

print("defining islands...")
evolution._initialise_islands()
print(f"there are {len(evolution._islands.keys())} islands")
for key, value in evolution._islands.items():
    print(f"for stride {key} there are {len(value)} models")
print(evolution._islands.keys())

start = time.time()
print("initialising fitnesses")
evolution._initialise_fitness
print("fitnesses were initialised!!!")
finish = time.time()
print(
    f"it took || {round(finish - start, 4)} sec || {round((finish - start)/60)} min to initialise fitnesses"
)

start = time.time()
evolution.evolve(
    generations=1
)
finish = time.time()

print(
    f"it took || {round(finish - start, 4)} sec || {round((finish - start)/60)} min to go through a generation"
)