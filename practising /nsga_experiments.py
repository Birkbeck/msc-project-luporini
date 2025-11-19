import time
import sys
import json

import random
import numpy as np
from matplotlib import pyplot as plt
from rainclouds import rainclouds

import torch
from torchvision import transforms
from torchvision.datasets import MNIST, KMNIST, CIFAR10, FashionMNIST
from torch.utils.data import DataLoader, Subset
from architectures import TinyFlexyConvAE, TinyConvClassifier
import all


###############################################
######## get the data ########################
###############################################
mytransform = transforms.ToTensor()

train_data = MNIST("./datasets", download=False, train=True, transform=mytransform)
test_data = MNIST("./datasets", download=False, train=False, transform=mytransform)

# train_data = CIFAR10("./datasets", download=False, train=True, transform=mytransform)
# test_data = CIFAR10("./datasets", download=False, train=False, transform=mytransform)

train_loader = DataLoader(train_data, batch_size=30)
test_loader = DataLoader(test_data, batch_size=30)

mnist = (1, 28, 28)
cifar = (3, 32, 32)

################################################
######### set the experiments #################
################################################
m = TinyConvClassifier
pop = 40
exps = 30
probl = "classification"
seed = 42
avg_convs = []
convs_in_time = []

_, ax = plt.subplots(figsize=(10, 10))
for e in range(exps):
    print(f"\nBeginning experiment {e}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    evolver = all.NSGA2(
        pop_size=pop,
        model=m,
        input_shape=mnist,
        interval=[1, 7],
        data=train_data,
        problem=probl
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
        m, 10, interval=[1, 7], bound1=b1, bound2=b2
    )

    # actual evolution
    print("- actual evolution..")
    evolver.evolve(
        generations=5,
        bound_estimation=False,
        m_prob=0.3
    )


    conv = evolver.conv_in_time()
    conv_final = evolver.avg_convergence()
    convs_in_time.append(conv)
    avg_convs.append(conv_final)
    print(f"Avg population convergence: {round(conv_final, 2)}")
    
    seed += 2

    ax.plot(range(len(conv)), conv, alpha=0.4, color="lightblue")

################################################
######## save results #########################
################################################
with open(f"MNIST_{probl}_{pop}_{exps}.json", "w") as f:
    json.dump({"avg_convs": avg_convs, "convs_in_time": convs_in_time})

# with open(f"MNIST_{pop}_{exps}.json", "r") as f:
#     data = json.load(f)
#     avg_convs = data["avg_convs"]
#     convs_in_time = data["convs_in_time"]

################################################
######## plot convergence #####################
################################################
# group convergence across experiments by generation
# avg_conv_per_gen = [sum(i)/len(i) for i in zip(*convs_in_time)]

# ax.plot(range(len(avg_conv_per_gen)), avg_conv_per_gen, color="tomato")
# ax.set_xlabel("generations")
# ax.set_ylabel("avg. distance from ideal solution")

# plt.show()


# # plot
# go = input("wanna plot?")
# if go:
#     evolution.plot_evolution()

