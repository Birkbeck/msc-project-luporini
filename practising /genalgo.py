from copy import deepcopy
import random
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

def flatten(model):
    """
     flatten each parameter into a 1D tensor and concatenate
    """
    device = next(model.parameters()).device
    return torch.cat([param.view(-1) for param in model.parameters()]).to(device)
    # return torch.cat([param.view(-1) for param in model.parameters()])
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


def mutate(guy:torch.Tensor, m_rate=0.2, m_strength=0.3, mode="small") -> torch.Tensor:
    """
    randomly mutate parameters in a 1D tensor

    args:
        guy: flat tensor
        m_chance: chance of mutation in "small" mode - mutatate if rand below m_chance
        mode: either "50/50", where half of the genes mutate on avg. or else, where #mutation depends on m_chance
        m_rate: scaling factor for mutation strength
    """
    device = guy.device
    if mode == "50/50":
        mask = torch.randint_like(guy, 2, device=device) # 0-1s mask.. which params are mutated
        # mutation = mask * noise #?? effect too great on 1s, needs scaling
        #⛔️half of the genes mutated on avg!!! AGGRESSIVE?
    else:
        mask = torch.rand_like(guy, device=device) < m_rate
    
    noise = torch.randn_like(guy, device=device)
    mutation = m_strength * mask * noise
    
    return guy + mutation


def crossover(parent1, parent2, type="uniform"):
    """
    uniform crossover between two flat 1D tensors by default. If type != "uniform", then one-point.
    """
    device = parent1.device
    if type == "uniform":
        mask = torch.randint_like(parent1, 2, device=device).bool()
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
        out = lambda X, y: y      # if working with encodings, will break!
    elif problem == "AE":
        loss_fn = nn.MSELoss()
        out = lambda X, y: X
    
    def fitness(model):
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            tot_loss = 0
            if problem == "classification":
                correct = 0
                tot = 0
            for X, y in data:
                X = X.to(device, non_blocking=True)
                y = out(X, y).to(device, non_blocking=True)
                
                pred = model(X)
                loss = loss_fn(pred, y)
                tot_loss += loss.item()   # ⛔️SPIKING FITNESS if avg_loss very small
                
                if problem == "classification":
                    y_pred_class = torch.argmax(pred, dim=1)
                    correct += (y_pred_class == y).sum().item()
                    tot += y.size(0)
            
            if problem == "classification":
                return correct / tot
        
            avg_loss = tot_loss / len(data) # enumerate starts from 0
            # avg_fitness = 1 / (avg_loss + 1e-8) # if avg_loss–>0, avg_fit–>inf!!
            avg_fitness = -avg_loss
        
        return avg_fitness
    
    return fitness


def group_fitness(pop:list, fn):
    """
    given a model pop and a fitness function, return list of fitnesses for each model
    """
    return [fn(i) for i in pop]


def normalise_fitness(fitnesses: list, bound):
    """
    normalises fitnesses between 0 and 1
    normalised_f = (f - min)/(max - min) 
    """
    mino, maxo = bound[0], bound[1]
    deno = (maxo - mino + 1e-8)
    normalised_fitnesses = [
        (f - mino) / deno
        for f in fitnesses
    ]
    return normalised_fitnesses




class GeneticAlgorithmV2():


    def __init__(
            self,
            model,
            pop_size,
            data,
            problem="AE"
    ):
        self._model = model
        self._pop_size = pop_size # ⁉️need it⁉️
        self._fit_fn = None
        self._data = data
        self._population = [deepcopy(model) for i in range(self._pop_size)]
        self._fitnesses = [None for i in range(self._pop_size)]
        self._problem = problem
    
    def _trainval_loaders(self, fraction):
        """
        produces train and val loaders
        val_loader_size < train_loader_size
        """
        full_idxs = list(range(len(self._data)))
        if isinstance(self._data.targets, torch.Tensor):
            labels = self._data.targets.numpy()
        elif isinstance(self._data.targets, list):
            labels = np.array(self._data.targets)  #need np.arrays for stratify
            
        train_indices, remaining = train_test_split(full_idxs, train_size=fraction, stratify=labels)
        train_subset = Subset(self._data, indices=train_indices)

        remaining_labels = labels[remaining]
        val_indices, _ = train_test_split(remaining, train_size=0.5, stratify=remaining_labels)
        val_subset = Subset(self._data, indices=val_indices)


        train_loader = DataLoader(
            train_subset, batch_size=30, shuffle=True, pin_memory=True
        )
        val_loader = DataLoader(
            val_subset, batch_size=30, shuffle=True, pin_memory=True
        )
        return train_loader, val_loader
    

    def _avg_fitness(self):
        fitnesses = [i for i in self._fitnesses if i is not None]
        if not fitnesses:
            return None
        return torch.mean(torch.tensor(fitnesses)).item()
        

    def evolve(self, generations=10, subset_fraction=0.07, m_r=0.01, m_s=0.2):
        """
        evolution method👍

        careful: do you want access to mutation parameters????

        args:
            generations: number of generations
            report_jump: integer n, with report given every n generations
        """
        #######################################
        print("starting experiment")
        train_loader, _ = self._trainval_loaders(subset_fraction)
            
        self._fit_fn = model_fitness(train_loader, self._problem)

        self._fitnesses = group_fitness( # clamped within emp_bounds
            self._population, self._fit_fn
        )
        
        for gen in range(generations):
            # mating events
            flat_children = [] # TOURNAMENT 🔥
            for _ in range(self._pop_size//2): 
                
                parents = random.sample(self._population, k=2)
                parent1, parent2 = parents[0], parents[1]
                parent1 = flatten(parent1)
                parent2 = flatten(parent2)
                        
                child1, child2 = crossover(parent1, parent2)
                child1 = mutate(child1, m_rate=m_r, m_strength=m_s, mode="small")
                child2 = mutate(child2, m_rate=m_r, m_strength=m_s, mode="small")
                    
                flat_children.extend([child1, child2])

            remodelled_children = [remodel(f, deepcopy(self._model)) for f in flat_children]
            
            children_fitnesses = group_fitness(
                remodelled_children, self._fit_fn
            )

            parents = list(zip(self._population, self._fitnesses))
            children = list(zip(remodelled_children, children_fitnesses))
            whole = parents + children 
            
            sorted_whole = sorted(whole, key=lambda x: x[1], reverse=True)
            self._population = [m for m, _ in sorted_whole[:self._pop_size]]
            self._fitnesses = [f for _, f in sorted_whole[:self._pop_size]]

            if self._problem == "classification":
                print(f"{gen} gen | avg. population acc: {self._avg_fitness()}")
            else:
                print(f"{gen} gen | avg. population finess: {self._avg_fitness()}")
            

    def extract_best(self, k=1):
        zipped = list(zip(self._population, self._fitnesses))
        sorted_population = sorted(zipped, key=lambda x: x[1], reverse=True)
        return sorted_population[:k]






# class GeneticAlgorithm():


#     def __init__(
#             self,
#             model,
#             pop_size,
#             data:DataLoader,
#             fit_fn=model_fitness, # returns a function when instantiated
#             problem="AE"
#     ):
#         self._model = model
#         self._pop_size = pop_size # ⁉️need it⁉️
#         self._fit_fn = fit_fn(data, problem=problem) #model_fitness is HIGHER ORDER
#         self._data = data
#         self._population = [deepcopy(model) for i in range(self._pop_size)]
#         self._fitnesses = [None for i in range(self._pop_size)]
        
#     def evolve(self, generations=10, report_jump=2, m_prob=0.3):
#         """
#         evolution method👍

#         careful: do you want access to mutation parameters????

#         args:
#             generations: number of generations
#             report_jump: integer n, with report given every n generations
#         """
#         for i in range(generations):
#             parent_fitnesses = group_fitness(self._population, self._fit_fn)
#             parents = list(zip(self._population, parent_fitnesses))
            
#             children = []  # too big tournament??? pop_size//3??? just 3-4???
#             for tournament in range(self._pop_size//2):
#                 pool = random.sample(parents, k=self._pop_size//2)
#                 sorted_pool = sorted(pool, key=lambda x: x[1], reverse=True)
#                 parent1, parent2 = sorted_pool[0][0], sorted_pool[1][0]
#                 flat1, flat2 = flatten(parent1), flatten(parent2)
#                 child1, child2 = crossover(flat1, flat2)

#                 # if random.random() < m_prob:
#                 #     child1 = mutate(child1)
#                 # if random.random() < m_prob:
#                 #     child2 = mutate(child2)
                
#                 # do I want children to always mutate⁉️
#                 child1 = mutate(child1)
#                 child2 = mutate(child2)

#                 child1 = remodel(child1, deepcopy(self._model))
#                 child2 = remodel(child2, deepcopy(self._model))
                
#                 children.extend([child1, child2])

#             children_fitnesses = group_fitness(children, self._fit_fn)
#             children = list(zip(children, children_fitnesses))
#             whole = parents + children
#             sorted_whole = sorted(whole, key=lambda x: x[1], reverse=True)

#             self._population = [m for m, _ in sorted_whole[:self._pop_size]]
#             self._fitnesses = [f for _, f in sorted_whole[:self._pop_size]]

#             if (i+1) % report_jump == 0:
#                 print(f"{i+1}th gen | avg. population finess: {self.avg_fitness()}")


#     def avg_fitness(self):
#         fitnesses = [i for i in self._fitnesses if i is not None]
#         if not fitnesses:
#             return None
#         return torch.mean(torch.tensor(fitnesses)).item()
    
#     def extract_best(self, k=1):
#         zipped = list(zip(self._population, self._fitnesses))
#         sorted_population = sorted(zipped, key=lambda x: x[1], reverse=True)
#         return sorted_population[:k]