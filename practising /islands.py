import random
from copy import deepcopy

import torch
from torch import nn
from torch.utils.data import DataLoader
from genalgo import flatten, mutate, crossover, model_fitness, group_fitness

def embed(model, biggest):
    flat = flatten(model)
    mu = flat.mean().item()
    sigma = flat.std().item()

    difference = biggest - flat.numel()
    if difference % 2 == 0:
        lx_pad_size = difference // 2
        rx_pad_size = lx_pad_size
    else:
        lx_pad_size = difference // 2
        rx_pad_size = difference - lx_pad_size
    
    lx_padding = torch.normal(mu, sigma, (lx_pad_size,), dtype=flat.dtype)
    rx_padding = torch.normal(mu, sigma, (rx_pad_size,), dtype=flat.dtype)

    return torch.cat([lx_padding, flat, rx_padding])

class Islands():


    def __init__(
            self,
            model,
            pop_size,
            data: DataLoader,
            interval=[1, 4], # small interval compared to pop_size? ⛔️ representativeness
            fit_fn = model_fitness,
            problem = "AE"
    ):
        self._model = model
        self._fit_fn = fit_fn(data, problem=problem) #model_fitness is HIGHER ORDER
        self._data = data
        self._population = [deepcopy(model(stride=random.randint(interval[0], interval[1]))) for i in range(pop_size)]
        self._fitnesses = [None for i in range(pop_size)]
        self._biggest = max(
            sum(param for param in model.parameters())
            for model in self._population
        )