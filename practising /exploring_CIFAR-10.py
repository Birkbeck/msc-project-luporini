import time
import sys

from matplotlib import pyplot as plt

import torch
from torch import nn
import architectures as archi

from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Subset

data_train = CIFAR10(root="./0 datasets", download=True, train=True, transform=ToTensor())
data_test = CIFAR10(root="./0 datasets", download=True, train=False, transform=ToTensor())


dataloader_train = DataLoader(data_train, batch_size=30)
dataloader_test = DataLoader(data_test, batch_size=30)

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

#################################################################
##### convolutional auto encoder ################################
#################################################################
# model = archi.FlexyConvAE(input=32, channels=3, stride=1, kernel=3)
# loss_fn = nn.MSELoss()
# optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
# noise = 0.3

# epochs = 4
# print("\nTraining is starting now:")
# for e in range(epochs):
#     tot_loss = 0
#     for i, (X, _) in enumerate(dataloader_train):
#         optimiser.zero_grad()
#         noisy = X + noise*torch.randn_like(X)
#         pred = model(noisy)

#         loss = loss_fn(pred, X)
#         loss.backward()
#         optimiser.step()

#         tot_loss += loss.item()
    
#     print(f"-Avg.loss per batch at {e+1}th epoch: {tot_loss/(i+1)}")

# fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(10, 10))

# model.eval()
# with torch.no_grad():
#     tot_loss = 0
#     for i, (X, y) in enumerate(dataloader_test):
#         noisy = X + noise*torch.randn_like(X)
#         pred = model(noisy)
#         loss = loss_fn(pred, X)

#         tot_loss += loss.item()

#         if i == 1:
#             for j in range(5):
#                 axes[0, j].imshow(X[j].permute(1, 2, 0))
#                 axes[0, 1].set_ylabel("original")
#                 axes[0, j].axis("off")
#                 axes[0, j].set_title(class_names[y[j].item()])

#                 axes[1, j].imshow(noisy[j].permute(1, 2, 0))
#                 axes[1, 0].set_ylabel("noisy")
#                 axes[1, j].axis("off")
#                 axes[1, j].set_title(class_names[y[j].item()])

#                 axes[2, j].imshow(pred[j].permute(1, 2, 0))
#                 axes[2, 0].set_ylabel("reconstructions")
#                 axes[2, j].axis("off")
#                 axes[2, j].set_title(class_names[y[j].item()])

    
#     print(f"\nAvg.loss at testing time: {tot_loss/(i+1)}")

# plt.show()


#################################################################
##### classification ############################################
#################################################################
model = archi.TinyConvClassifier(input=32, channels=3, stride=1, kernel=3)
loss_fn = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
noise = 0.3

epochs = 4
training_losses = []
print("\nTraining is starting now:")
for e in range(epochs):
    tot_loss = 0
    for i, (X, y) in enumerate(dataloader_train):
        optimiser.zero_grad()
        noisy = X + noise*torch.randn_like(X)
        pred = model(noisy)

        loss = loss_fn(pred, X)
        loss.backward()
        optimiser.step()

        tot_loss += loss.item()
    
    avg_loss = tot_loss/(i+1)
    
    print(f"-Avg.loss per batch at {e+1}th epoch: {tot_loss/(i+1)}")


model.eval()
with torch.no_grad():
    tot_loss = 0
    for i, (X, y) in enumerate(dataloader_test):
        noisy = X + noise*torch.randn_like(X)
        pred = model(noisy)
        loss = loss_fn(pred, X)

        tot_loss += loss.item()

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 7))
axes[0].plot(training_losses)
axes[1].plot()
plt.show()
    