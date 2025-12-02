import sys
import random
from copy import deepcopy
from collections import defaultdict

import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from genalgo import flatten, mutate, model_fitness, group_fitness, normalise_fitness

def embed(model, biggest): # seed=42???????
    flat = flatten(model)
    mu = flat.mean().item()
    sigma = flat.std().item()

    size = flat.numel()
    difference = biggest - size
    if difference % 2 == 0:
        lx_pad_size = difference // 2
        rx_pad_size = lx_pad_size
    else:
        lx_pad_size = difference // 2
        rx_pad_size = difference - lx_pad_size
    
    # if seed is not None:
    #     torch.manual_seed(seed)
    
    # careful here⛔️: every time embed() is called, torch.random introduces randomness
    lx_padding = torch.normal(mu, sigma, (lx_pad_size,), dtype=flat.dtype)
    rx_padding = torch.normal(mu, sigma, (rx_pad_size,), dtype=flat.dtype)

    return (
        torch.cat([lx_padding, flat, rx_padding]), size, model
    ) # (flat, size, archi) (f, s, a)


def remodel(embedded, original_size, model, biggest):
    difference = biggest - original_size
    lx = difference // 2
    flat = embedded[lx:lx + original_size]

    index = 0
    for param in model.parameters():
        with torch.no_grad():
            n = param.numel()
            param.copy_(flat[index:index+n].view_as(param))
            index += n
    return model


# rename to distinguish from crossover() in genalgo?
def crossover(parent1: torch.Tensor, parent2: torch.Tensor)-> tuple[torch.Tensor, torch.Tensor]:
    """
    masked crossover between two flat models:
    Args:
       parent1: flat model
       parent2: flat model
    """
    flat1, s, a = parent1[0], parent1[1], parent1[2]
    flat2 = parent2[0]
    
    mask = torch.randint(0, 2, flat1.shape, dtype=torch.bool) # mask with zeroes and ones
    child1 = (torch.where(mask, flat1, flat2), s, a)
    child2 = (torch.where(mask, flat2, flat1), s, a)

    return child1, child2

class Islands():
    """
    remember:
        - initialise_islands????
        - initialise_fitness????
        - need the right problem when initialising the Island class!!!

    class attributes:
        pop_size: integer for pop size
        model: model class
        data: raw image data... getting random subset every generation
        interval: integer for the interval within which the varying model parameter can fall
    """
    def _initialise_islands(self):
        self._islands = defaultdict(list)
        for m in self._population:
            key = m.get_stride()
            self._islands[key].append(m)
    
    def _initialise_fitness(self):
        self._fitness = group_fitness(self._population, self._fit_fn)

    def __init__(
            self,
            pop_size,
            model,
            data,
            interval=[1, 4], # small interval compared to pop_size? ⛔️ representativeness
            # fit_fn = , # just pass the class
            problem = "AE"
    ):
        self._islands = None
        self._pop_size = pop_size
        self._model = model # needs to be a class, not an istance!
        self._fit_fn = model_fitness#(data, problem=problem) #instancing the class
        self._problem = problem
        
        self._data = data
        self._population = [deepcopy(model(stride=random.randint(interval[0], interval[1]))) for i in range(pop_size)]
        self._fitnesses = None
        self._biggest = max(
            sum(param.numel() for param in m.parameters()) # ⛔️ will change mid run????
            for m in self._population
        )
    
    def evolve(self, generations=10, subset_fraction=0.1, report_jump=2, m_prob=0.3):

        for gen in range(generations):
            
            # BUT NEEDS STRATIFYING ⛔️ TRAIN_TEST_SPLIT 🔥
            full_idxs = list(range(len(self._data)))
            labels = self._data.targets.numpy()
            random_indices, _ = train_test_split(full_idxs, train_size=subset_fraction, stratify=labels)
            subset = Subset(self._data, indices=random_indices)

            train_loader = DataLoader(subset, batch_size=30)
            fit_fn = self._fit_fn(train_loader, self._problem)

            # initialise self._fitnesses coz can't add list + None later
            if gen == 0:
                self._fitnesses = group_fitness(self._population, fit_fn)

            # checking topologies.. changing through generations ⁉️
            for key, value in self._islands.items():
                print(f"{key}: {len(value)} models")
            print(f"\npopulation size: {self._pop_size}")

            # mating events 🔥
            # what if island with only 1 individual???? CANT SAMPLE 2!!!!
            children = [] # TOURNAMENT 🔥
            for _ in range(self._pop_size//2):
                if random.random() < 0.1: # unlikely cross-species crossover 🔥
                    random_keys = random.sample(list(self._islands.keys()), k=2)
                    key1, key2 = random_keys[0], random_keys[1]
                    pool1, pool2 = self._islands[key1], self._islands[key2]
                    parent1, parent2 = random.choice(pool1), random.choice(pool2)
                    parent1, parent2 = embed(parent1, self._biggest), embed(parent2, self._biggest)
                else: # regular intraspecies crossover 🔥
                    key = random.choice(list(self._islands.keys()))
                    pool = self._islands[key]
                    if len(pool) == 1:
                        # parent1, parent2 = pool[0], deepcopy(pool[0])
                        continue
                    else:
                        parents = random.sample(pool, k=2)
                        parent1, parent2 = parents[0], parents[1]
                        parent1, parent2 = embed(parent1, self._biggest), embed(parent2, self._biggest)
                
                child1, child2 = crossover(parent1, parent2)
                child1 = (mutate(child1[0]), child1[1], child1[2])
                child2 = (mutate(child2[0]), child2[1], child2[2])
                
                children.extend([child1, child2])
            
            ##############################################################
            remodelled_children = [remodel(f, s, a, self._biggest) for f, s, a in children]
            children_fitnesses = group_fitness(remodelled_children, fit_fn)
            all_solutions = self._population + remodelled_children
            all_fitnesses = self._fitnesses + children_fitnesses
            whole = list(zip(all_solutions, all_fitnesses))
            sorted_whole = sorted(whole, key=lambda x: x[1], reverse=True)

            self._population = [s for s, _ in sorted_whole[:self._pop_size]]
            self._fitnesses = [f for _, f in sorted_whole[:self._pop_size]]

            self._initialise_islands()

            current_biggest = max(
                sum(param.numel() for param in m.parameters()) 
                for m in self._population
            )
            if current_biggest != self._biggest:
                self._biggest = current_biggest
            
            # if (i+1) % report_jump == 0: #⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️
            print(f"{gen+1}th gen | avg. population finess: {self.avg_fitness()}")
            

    def avg_fitness(self):
        fitnesses = [i for i in self._fitnesses if i is not None]
        if not fitnesses:
            return None
        return torch.mean(torch.tensor(fitnesses)).item()              
