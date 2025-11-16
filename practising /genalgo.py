from copy import deepcopy
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

def flatten(model):
    """
     flatten each parameter into a 1D tensor and concatenate
    """
    return torch.cat([param.view(-1) for param in model.parameters()])
    # return torch.cat([param.data.view(-1) for param in model.parameters()]) # NO!!!!
                            # BREAKS COMPUTATION GRAPH.. don't risk it
                            # might need autograd later


def remodel(flat, model):
    """
    remap a flat model into a non flat torch.architecture
    """
    index = 0
    for param in model.parameters():
        with torch.no_grad():
            n = param.numel()
            param.copy_(flat[index:index+n].view_as(param))
            index += n
    return model


def mutate(guy:torch.Tensor, m_chance=0.05, mode="small", m_rate=0.3) -> torch.Tensor:
    """
    randomly mutate parameters in a 1D tensor

    args:
        guy: flat tensor
        m_chance: chance of mutation in "small" mode - mutatate if rand below m_chance
        mode: either "50/50", where half of the genes mutate on avg. or else, where #mutation depends on m_chance
        m_rate: scaling factor for mutation strength
    """
    if mode == "50/50":
        mask = torch.randint_like(guy, 2) # 0-1s mask.. which params are mutated
        strength = torch.randn_like(guy) # raw mutation effect
        # mutation = mask * noise #?? effect too great on 1s, needs scaling
        mutation = m_rate * mask * strength
        #⛔️half of the genes mutated on avg!!! AGGRESSIVE?
    else:
        mask = torch.rand_like(guy) < m_chance
        strength = torch.randn_like(guy)
        mutation = m_rate * mask * strength
    
    return guy + mutation


def crossover(parent1, parent2, type="uniform"):
    """
    uniform crossover between two flat 1D tensors by default. If type != "uniform", then one-point.
    """
    if type == "uniform":
        mask = torch.randint_like(parent1, 2)
        child1 = torch.where(mask, parent1, parent2)
        child2 = torch.where(mask, parent2, parent1)
    else:
        rip = torch.randint(0, parent1.numel()-1, size=(1,)).item() #for fun..
        child1 = torch.cat([parent1[:rip], parent2[rip:]])
        child2 = torch.cat([parent2[:rip], parent1[rip:]])
    return child1, child2


def model_fitness(data: DataLoader, problem="AE"):
    """
    returns a fitness function that computes 1/avg_loss = avg_fitness
    across batches given a model

    Args:
        problem: either "regression", "classification" or default (AE).
    """
    if problem == "regression":
        loss_fn = nn.MSELoss()
        out = lambda X, y: y
    elif problem == "classification":
        loss_fn = nn.CrossEntropyLoss() # expect logits, y.shape((batch,))
        out = lambda X, y: y        # if working with encodings, will break!
    else:
        loss_fn = nn.MSELoss()
        out = lambda X, y: X
    
    def fitness(model):
        model.eval()
        with torch.no_grad():
            tot_loss = 0
            for X, y in data:
                X, y = X, out(X, y)
                pred = model(X)
                loss = loss_fn(pred, y)
                tot_loss += loss.item()   # ⛔️SPIKING FITNESS if avg_loss very small
            avg_loss = tot_loss / len(data) # enumerate starts from 0
            avg_fitness = 1 / (avg_loss + 1e-8) # if avg_loss–>0, avg_fit–>inf!!
        return avg_fitness
    
    return fitness


def group_fitness(pop:list, fn):
    """
    given a model pop and a fitness function, return list of fitnesses for each model
    """
    return [fn(i) for i in pop]


def normalise_fitness(fitnesses: list, mino, maxo):
    """
    normalises fitnesses between 0 and 1
    normalised_f = (f - min)/(max - min) 
    """
    deno = (maxo - mino + 1e-8)
    normalised_fitnesses = [
        (f - mino) / deno
        for f in fitnesses
    ]
    return normalised_fitnesses
    

class GeneticAlgorithm():


    def __init__(
            self,
            model,
            pop_size,
            data:DataLoader,
            fit_fn=model_fitness, # returns a function when instantiated
            problem="AE"
    ):
        self._model = model
        self._pop_size = pop_size # ⁉️need it⁉️
        self._fit_fn = fit_fn(data, problem=problem) #model_fitness is HIGHER ORDER
        self._data = data
        self._population = [deepcopy(model) for i in range(self._pop_size)]
        self._fitnesses = [None for i in range(self._pop_size)]
        
    def evolve(self, generations=10, report_jump=2, m_prob=0.3):
        """
        evolution method👍

        careful: do you want access to mutation parameters????

        args:
            generations: number of generations
            report_jump: integer n, with report given every n generations
        """
        for i in range(generations):
            parent_fitnesses = group_fitness(self._population, self._fit_fn)
            parents = list(zip(self._population, parent_fitnesses))
            
            children = []  # too big tournament??? pop_size//3??? just 3-4???
            for tournament in range(self._pop_size//2):
                pool = random.sample(parents, k=self._pop_size//2)
                sorted_pool = sorted(pool, key=lambda x: x[1], reverse=True)
                parent1, parent2 = sorted_pool[0][0], sorted_pool[1][0]
                flat1, flat2 = flatten(parent1), flatten(parent2)
                child1, child2 = crossover(flat1, flat2)

                # if random.random() < m_prob:
                #     child1 = mutate(child1)
                # if random.random() < m_prob:
                #     child2 = mutate(child2)
                
                # do I want children to always mutate⁉️
                child1 = mutate(child1)
                child2 = mutate(child2)

                child1 = remodel(child1, deepcopy(self._model))
                child2 = remodel(child2, deepcopy(self._model))
                
                children.extend([child1, child2])

            children_fitnesses = group_fitness(children, self._fit_fn)
            children = list(zip(children, children_fitnesses))
            whole = parents + children
            sorted_whole = sorted(whole, key=lambda x: x[1], reverse=True)

            self._population = [m for m, _ in sorted_whole[:self._pop_size]]
            self._fitnesses = [f for _, f in sorted_whole[:self._pop_size]]

            if (i+1) % report_jump == 0:
                print(f"{i+1}th gen | avg. population finess: {self.avg_fitness()}")


    def avg_fitness(self):
        fitnesses = [i for i in self._fitnesses if i is not None]
        if not fitnesses:
            return None
        return torch.mean(torch.tensor(fitnesses)).item()
    
    def extract_best(self, k=1):
        zipped = list(zip(self._population, self._fitnesses))
        sorted_population = sorted(zipped, key=lambda x: x[1], reverse=True)
        return sorted_population[:k]





