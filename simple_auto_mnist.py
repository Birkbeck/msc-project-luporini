import sys
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import torch
from torch import nn

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Subset

class AutoEncoder():


    def __init__(self, input, latent, nonlinearity=nn.ReLU):
        self._input = input
        self._latent = latent
        self._nonl = nonlinearity

        self.encoder = nn.Sequential(
            nn.Linear(self._input, self._latent),
            self._nonl
        )

        self.decoder = nn.Sequential(
            nn.Linear(self._latent, self._input),
            nn.Sigmoid()
        )

    def forward(self, data):
        output = self.encoder(data)
        output = self.decoder(output)
        return output