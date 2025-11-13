import random
from copy import deepcopy
from collections import defaultdict

import torch
from torch import nn
from torch.utils.data import DataLoader
from genalgo import flatten, mutate, model_fitness, group_fitness

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
    
    # careful here: every time embed() is called, torch.random introduces randomness
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
    add description...
    """
    def __init__(
            self,
            model,
            pop_size,
            data: DataLoader,
            interval=[1, 4], # small interval compared to pop_size? ⛔️ representativeness
            fit_fn = model_fitness,
            problem = "AE"
    ):
        self._pop_size = pop_size
        self._model = model # needs to be a class, not an istance!
        self._fit_fn = fit_fn(data, problem=problem) #model_fitness is HIGHER ORDER
        self._data = data
        self._population = [deepcopy(model(stride=random.randint(interval[0], interval[1]))) for i in range(pop_size)]
        self._fitnesses = group_fitness(self._population, self._fit_fn)
        self._biggest = max(
            sum(param.numel() for param in m.parameters()) # ⛔️ will change mid run????
            for m in self._population
        )
    
    def evolve(self, generations=10, report_jump=2, m_prob=0.3):
        for gen in range(generations):
            embedded_parents = [embed(m, biggest=self._biggest) for m in self._population]
            
            islands = defaultdict(list)
            for i in embedded_parents:
                islands[i[1]].append(i)

            for i, group in enumerate(islands.values()): # just to check⛔️
                print(f"{i} island: {len(group)} models")

            # mating events, either within(more likely) or between(less likely)
            children = [] # TOURNAMENT 🔥
            for _ in range(self._pop_size//2):
                if random.random() < 0.1: # then cross-island crossover
                    random_keys = random.sample(list(islands.keys()), k=2)
                    key1, key2 = random_keys[0], random_keys[1]
                    pool1, pool2 = islands[key1], islands[key2]
                    parent1, parent2 = random.choice(pool1), random.choice(pool2)
                else:
                    key = random.choice(list(islands.keys()))
                    pool = islands[key]
                    parents = random.sample(pool, k=2)
                    parent1, parent2 = parents[0], parents[1]
                
                child1, child2 = crossover(parent1, parent2)
                child1 = (mutate(child1[0]), child1[1], child1[2])
                child2 = (mutate(child2[0]), child2[1], child2[2])
                
                children.extend([child1, child2])

            remodelled_children = [remodel(f, s, a, self._biggest) for f, s, a in children]
            children_fitnesses = group_fitness(remodelled_children, self._fit_fn)
            all_solutions = self._population + remodelled_children
            all_fitnesses = self._fitnesses + children_fitnesses
            whole = list(zip(all_solutions, all_fitnesses))
            sorted_whole = sorted(whole, key=lambda x: x[1], reverse=True)

            self._population = [s for s, _ in sorted_whole[:self._pop_size]]
            self._fitnesses = [f for _, f in sorted_whole[:self._pop_size]]

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
