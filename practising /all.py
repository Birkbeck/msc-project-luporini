import os

import random
import time
import copy
from copy import deepcopy
from collections import defaultdict
import math

import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

mydevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def flatten(model):
    """
     flatten each parameter into a 1D tensor and concatenate
    """
    device = next(model.parameters()).device
    return torch.cat([param.view(-1) for param in model.parameters()]).to(device)
    # return torch.cat([param.data.view(-1) for param in model.parameters()]) # NO!!!!
                            # BREAKS COMPUTATION GRAPH.. don't risk it
                            # might need autograd later

def embed(model, biggest):
    device = next(model.parameters()).device
    flat = flatten(model)
    mu = flat.mean().item()

    size = flat.numel()
    difference = biggest - size # 
    if difference % 2 == 0:
        lx_pad_size = difference // 2
        rx_pad_size = lx_pad_size
    else:
        lx_pad_size = difference // 2
        rx_pad_size = difference - lx_pad_size
    
    # ⛔️: every time embed() called, torch.random introduces randomness
    lx_padding = torch.full((lx_pad_size,), mu, dtype=flat.dtype, device=device)
    rx_padding = torch.full((rx_pad_size,), mu, dtype=flat.dtype, device=device)

    return (
        torch.cat([lx_padding, flat, rx_padding]), size, model
    ) # (flat, size, archi) (f, s, a)


def remodel(embedded, original_size, model, biggest):
    device = next(model.parameters()).device
    difference = biggest - original_size
    lx = difference // 2
    flat = embedded[lx:lx + original_size].to(device)

    index = 0
    for param in model.parameters():
        with torch.no_grad():
            n = param.numel()
            param.copy_(flat[index:index+n].view_as(param))
            index += n
    return model


# rename to distinguish from crossover() in genalgo?
def crossover(parent1: tuple, parent2: tuple)-> tuple[torch.Tensor, torch.Tensor]:
    """
    masked crossover between two flat models:
    Args:
       parent1: flat model
       parent2: flat model
    """
    flat1, s, a = parent1
    flat2, _, _ = parent2
    
    device = flat1.device 
    mask = torch.randint(0, 2, flat1.shape, dtype=torch.bool, device=device) # mask with zeroes and ones
    child1 = (torch.where(mask, flat1, flat2), s, deepcopy(a))
    child2 = (torch.where(mask, flat2, flat1), s, deepcopy(a))

    return child1, child2


def mutate(guy:torch.Tensor, m_chance=0.2, m_rate=0.3, mode="small") -> torch.Tensor:
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
        strength = torch.randn_like(guy, device=device) # raw mutation effect
        # mutation = mask * noise #?? effect too great on 1s, needs scaling
        mutation = m_rate * mask * strength
        #⛔️half of the genes mutated on avg!!! AGGRESSIVE?
    else:
        mask = torch.rand_like(guy, device=device) < m_chance
        strength = torch.randn_like(guy, device=device)
        mutation = m_rate * mask * strength
    
    return guy + mutation


def model_fitness(data: DataLoader, problem="AE"):
    """
    returns a fitness function that computes 1/avg_loss = avg_fitness
    across batches given a model

    ⛔️ using non_blocking + pin_memory (DataLoader at the start of evolve):
    https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html

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
        device=next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            tot_loss = 0
            for X, y in data:
                X = X.to(device, non_blocking=True)
                y = out(X, y).to(device, non_blocking=True)
                pred = model(X)
                loss = loss_fn(pred, y)
                tot_loss += loss.item()   # ⛔️SPIKING FITNESS if avg_loss very small
            avg_loss = tot_loss / len(data) # enumerate starts from 0
            # avg_fitness = 1 / (avg_loss + 1e-8) # if avg_loss–>0, avg_fit–>inf!!
            avg_fitness = -avg_loss
        return avg_fitness
    
    return fitness


def model_runtime(data: DataLoader):
    """
    careful if using on GPU.. operations are asynchronous
    - torch.cuda.synchronise() ⁉️
    - synch for every time.time() ⁉️
    (https://discuss.pytorch.org/t/bizzare-extra-time-consumption-in-pytorch-gpu-1-1-0-1-2-0/87843)

    ALSO

    - just evaluate on representative batch ⁉️
    - return len(data)/interval ⁉️
    """
    def speed(model):
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            ##########################
            # wait for all GPU kernels to be done !!!
            if device.type == "cuda":
                torch.cuda.synchronize() 
            ##########################

            start = time.time()
            for X, _ in data:
                X = X.to(device)
                _ = model(X)

            ##########################
            # wait for all GPU kernels to be done !!!
            if device.type == "cuda":
                torch.cuda.synchronize() 
            ##########################
            
            finish = time.time()
            # interval = (finish - start) / len(data)
            interval = finish - start # just raw time, not avg. (would be too quick?)
        return 1/interval
    
    return speed


def normalise_objective(fitnesses: list, bound):
    """
    normalises fitnesses between 0 and 1
    normalised_f = (f - min)/(max - min) 
    """
    mino, maxo = bound
    normalised_fitnesses = [
        (f - mino) / (maxo - mino + 1e-8)
        for f in fitnesses
    ]
    return normalised_fitnesses


def group_fitness(pop:list, fn, bound_estimation:bool, bound:tuple)->list:
    """
    given a model pop and a fitness function, return fitness for each model
    - fitnesses are clamped between the given bound (empirical bounds!!!)
    """
    if bound_estimation:
       return [fn(i) for i in pop] 
    else:
        mino, maxo = bound
        return [  # min(fit, maxo) + max(fit, mino) # use generator inside!!
            max(min(fit, maxo), mino) for fit in (fn(i) for i in pop)
        ]


def non_dominated_sorting(whole, fits1, fits2):
    """
    careful:
    - dom_counts: list of integers (idx = )
    - dominateds: list of lists (idx = by whom)
    - dominated: list 
    """
    # HELPER FUNCTION
    def _dominates(i, j):
        """
        MAXIMISATION problem: i dominates j if neither fitness worse and at least one better
        "neither fitness worse" F1i >= F1j AND F2i >= F2j
        "at least one better" F1i > F1j OR F2i > F2j
        """
        return (
            (fits1[i] >= fits1[j] and fits2[i] >= fits2[j]) and
            (fits1[i] > fits1[j] or fits2[i] > fits2[j])
        )

    fronts = []
    dom_counts = []
    dominateds = []

    first_front = []
    for i in range(len(whole)): # for each solution idx in whole
        dom_count = 0
        dominated = []
        for j in range(len(whole)):
            if _dominates(i, j):
                dominated.append(j)
            elif _dominates(j, i):
                dom_count += 1
        if dom_count == 0:
            first_front.append(i)

        dom_counts.append(dom_count)
        dominateds.append(dominated)

    fronts.append(first_front)

    # a solution is in the next front if,
    # when current first is not considered (dom_count of underlying
    # solutions -=1), the solution dom_count == 0
    current_idx = 0
    while current_idx < len(fronts) and fronts[current_idx]:
        next_front = []
        for p in fronts[current_idx]: # for each p in current front
            for q in dominateds[p]: # go to solutions q that p dominates
                dom_counts[q] -= 1 # pretend no current front
                if dom_counts[q] == 0:
                    next_front.append(q)

        if next_front:
            fronts.append(next_front)

        current_idx +=1

    return fronts

# HELPER FUNCTION
def crowding_distance(front, *objectives):
    """
    Args:
        front: a list of idx from the populations
        objectives: list of o lists, where o == number of objectives 
        -> objectives[o] == list of fitnesses for ALL solutions in pop
    """
    distances = {s: float(0) for s in front}
    objectives = [o for o in objectives]

    for o in range(len(objectives)):
        sorted_front = sorted(front, key=lambda idx: objectives[o][idx]) # min –> max
        # each front solution (idx) -> fitnesses[idx]
        # each front solution is mapped to a fitness
        # and sorted by it, in order of increasing fitness
        distances[sorted_front[0]] = math.inf
        distances[sorted_front[-1]] = math.inf

        mins = objectives[o][sorted_front[0]] # fit_value corresponding a sorted_fron[0]
        maxs = objectives[o][sorted_front[-1]] # fit_value corresponding a sorted_fron[-1]

        if mins == maxs:
            continue
  
        for i in range(1, len(front)-1):
            higher = objectives[o][sorted_front[i+1]]
            lower = objectives[o][sorted_front[i-1]]
            gap = (higher - lower)/(maxs - mins) 
            distances[sorted_front[i]] += gap
            # divide by (max - min) so it makes sense
            # to add values from different objectives

    return distances

# HELPER FUNCTIONS for evaluation convergence and spread⛔️
def convergence(*fits):
    """for a model: Euclidean distance from ideal s in nD"""
    return math.sqrt(sum((f - 1)**2 for f in fits))

        
# def spread()



class NSGA2():
    """
    ⛔️ bug in evolve.TOURNAMENT ~ CONVERGENCE ⛔️
        - if evolution converges towards one archi only,
          only one key available
    
    multi-objective evolution based on Islands + original nsga-ii
    obj_1 = classic net optimisation (MSELoss/CrossEntropy)
    obj_2 = inference speed

    remember:
        - initialise islands????????
        - need the right 'problem' when initialising the Island class!!!
    """
    def __init__(
            self,
            pop_size,
            model,
            data,
            input_shape=(1, 28, 28),
            interval=[1, 4], # small interval compared to pop_size? ⛔️ representativeness
            problem = "AE"
    ):
        self._islands = None
        self._data = data
        self._input_shape = input_shape
        self._model = model # needs to be a class, not an istance!
        self._problem = problem
        self._pop_size = pop_size
        self._population = [
            deepcopy(
                model(
                    input_shape=input_shape,
                    stride=random.randint(interval[0], interval[1])
                ).to(mydevice)
            ) for i in range(pop_size)
        ]

        self._fit_fn_1 = model_fitness#(data, problem=problem) #model_fitness is HIGHER ORDER
        self._fit_fn_2 = model_runtime#(data)
        self._fitnesses_1 = None
        self._fitnesses_2 = None # why was it missing⁉️⁉️⁉️⁉️⁉️⁉️⁉️
        self._fitnesses_1_pool = []
        self._fitnesses_2_pool = []
        self._convergence = [] # list of lists: normalised distances per generation
        self._best_model = None
        self._best_convergence = None
        self._emp_bounds_1 = None # empirical bounds per objective
        self._emp_bounds_2 = None
        self._best_front = None

        self._biggest = max(
            sum(param.numel() for param in m.parameters()) # ⛔️ will change mid run????
            for m in self._population
        )

        # self._gen = 0
        # self._max_gen = None
    
    def _initialise_islands(self):
        self._islands = defaultdict(list)
        for m in self._population:
            key = m.get_stride()
            self._islands[key].append(m)
    
    def _check_biggest(self):
        current_biggest = max(
            sum(param.numel() for param in m.parameters()) 
                for m in self._population
        )
        if current_biggest != self._biggest:
            self._biggest = current_biggest
    
    def _initialise_fitness(self):
        self._fitness = group_fitness(self._population, self._fit_fn)
    
    def _sample_loader(self, fraction):
        full_idxs = list(range(len(self._data)))
        if isinstance(self._data.targets, torch.Tensor):
            labels = self._data.targets.numpy()
        elif isinstance(self._data.targets, list):
            labels = np.array(self._data.targets)
            
        random_indices, _ = train_test_split(full_idxs, train_size=fraction, stratify=labels)
        subset = Subset(self._data, indices=random_indices)

        loader = DataLoader(
            subset, batch_size=30, shuffle=True, pin_memory=True
        )
        return loader
    
    def _bounds_estimation(self, fitnesses):
        """update (or not) bounds"""
        mino = np.percentile(fitnesses, 5)
        maxo = np.percentile(fitnesses, 95)
        return mino, maxo
    
    def _estimate_convergence(self):
        """
        given the current population,
        update the model closer to the ideal solution
        and its distance from this ideal solution.
        """
        normalised_x = normalise_objective(self._fitnesses_2, self._emp_bounds_2) # x speed
        normalised_y = normalise_objective(self._fitnesses_1, self._emp_bounds_1) # y fitness

        # getting distance from ideal for each model 
        distances = [] # and record in self._convergence as pop_avg per gen
        for i in range(self._pop_size):
            distances.append(convergence(normalised_x[i], normalised_y[i]))
        self._convergence.append(distances)

        # finding most balanced model (closest to ideal)
        zipped = list(zip(self._population, distances))
        ordered = sorted(zipped, key= lambda x: x[1])
        best_model, best_convergence = ordered[0]
        if self._best_model is None or self._best_convergence is None:
            self._best_model = deepcopy(best_model)
            self._best_convergence = best_convergence
        else:
            if best_convergence < self._best_convergence:
                self._best_model = deepcopy(best_model)
                self._best_convergence = best_convergence
    
    def _clear_attributes(self, bound1, bound2):
        self._fitnesses_1 = None
        self._fitnesses_2 = None
        self._convergence = [] # list of lists: normalised distances per generation
        self._best_model = None
        self._best_convergence = None
        self._emp_bounds_1 = bound1 # empirical bounds per objective
        self._emp_bounds_2 = bound2

    def get_best(self):
        best = self._best_model, self._best_convergence
        return best
    
    def save_best(self, filepath):
        best = {
            "weights": self._best_model.state_dict(),
            "convergence": self._best_convergence
        }
        torch.save(best, filepath)
    
    def get_bounds(self):
        return self._emp_bounds_1, self._emp_bounds_2
    
    def set_bounds(self, b1, b2):
        self._emp_bounds_1 = b1
        self._emp_bounds_2 = b2
    
    def get_best_front(self):
        return self._best_front
    
    def conv_in_time(self):
        """returns list of avg.pop distance from ideal point per generation"""
        return [sum(i)/len(i) for i in self._convergence]
    
    def avg_convergence(self):
        """get FINAL population convergence (mean convergence at last gen)"""
        avg_convergence = sum(self._convergence[-1])/self._pop_size
        return avg_convergence
    
    def plot_convergence(self):
        """"""
        distances = self.conv_in_time()
        
        _, ax = plt.subplots()
        ax.plot(range(len(distances)), distances)
        ax.set_xlabel("generation")
        ax.set_ylabel("avg. distance from ideal solution")
        plt.show()
    
    # def reset(self, model, pop_size, interval, bound1, bound2):
    #     self._population = [
    #         deepcopy(
    #             model(
    #                 input_shape=self._input_shape,
    #                 stride=random.randint(interval[0], interval[1])
    #             ).to(mydevice)
    #         ) for i in range(pop_size)
    #     ]
    #     self._clear_attributes(bound1, bound2)
    #     self._gen = 0
    #     self._max_gen = None
    

    # def transfer(self, model, bound1, bound2, freeze=False):
    #     """should I just swap the new pop in??"""
    #     pop = []
    #     for m in self._population:
    #         weights = m.encoder.state_dict()
    #         new = model(m, stride=m.get_stride()).to(mydevice)
    #         new.encoder.load_state_dict(weights)

    #         if freeze:
    #             for param in new.parameters():
    #                 param.requires_grad = False

    #         pop.append(new)
        
    #     self._population = pop
    #     self._clear_attributes(bound1, bound2)

    def get_transfer_pop(self, model, freeze=False):
        """should I just swap the new pop in??"""
        pop = []
        for m in self._population:
            weights = m.encoder.state_dict()
            new = model(m, stride=m.get_stride()).to(mydevice)
            new.encoder.load_state_dict(weights)

            if freeze:
                for param in new.parameters():
                    param.requires_grad = False

            pop.append(new)
        
        return pop

    def transfer_pop(self, pop):
        self._population = pop

    def evolve(
            self,
            prestep=False,
            bound_estimation=True,
            generations=0,
            subset_fraction=0.07,
            m_r=0.3,
            m_c=0.2
    ):
        
        for gen in range(generations):

            loader_sample = self._sample_loader(subset_fraction)
            fit_fn_1 = self._fit_fn_1(loader_sample, self._problem)
            fit_fn_2 = self._fit_fn_2(loader_sample)

            
            self._fitnesses_1 = group_fitness( # clamped within emp_bounds
                self._population, fit_fn_1, bound_estimation, self._emp_bounds_1
            )
            self._fitnesses_2 = group_fitness( # clamped within emp_bounds
                self._population, fit_fn_2, bound_estimation, self._emp_bounds_2
            )

            ################################################
            if gen == 0:
                self._initialise_islands()
            ################################################

            # mating events, either within(more likely) or between(less likely)
            children = [] # TOURNAMENT 🔥
            self._check_biggest()
            for _ in range(self._pop_size//2):
                if random.random() < 0.1 and len(self._islands)>= 2: # unlikely cross-species crossover 🔥
                    random_keys = random.sample(list(self._islands.keys()), k=2)
                    key1, key2 = random_keys[0], random_keys[1]
                    pool1, pool2 = self._islands[key1], self._islands[key2]
                    parent1, parent2 = random.choice(pool1), random.choice(pool2)
                    parent1, parent2 = embed(parent1, self._biggest), embed(parent2, self._biggest)
                else: # regular intraspecies crossover 🔥
                    key = random.choice(list(self._islands.keys()))
                    pool = self._islands[key]
                    if len(pool) == 1:
                        parent1, parent2 = pool[0], deepcopy(pool[0])
                        parent1, parent2 = embed(parent1, self._biggest), embed(parent2, self._biggest)
                    elif len(pool) == 2:
                        parent1, parent2 = pool[0], pool[1]
                        parent1, parent2 = embed(parent1, self._biggest), embed(parent2, self._biggest)
                    else:
                        parents = random.sample(pool, k=2)
                        parent1, parent2 = parents[0], parents[1]
                        parent1, parent2 = embed(parent1, self._biggest), embed(parent2, self._biggest)
                
                self._check_biggest()

                child1, child2 = crossover(parent1, parent2)
                child1 = (mutate(child1[0], m_rate=m_r, m_chance=m_c), child1[1], child1[2]) # mutate 50% of genes
                child2 = (mutate(child2[0], m_rate=m_r, m_chance=m_c), child2[1], child2[2]) # mutate 50% of genes
                
                children.extend([child1, child2])

                                    # remodel(f,s,a,biggest)–>model!
            remodelled_children = [remodel(f, s, a, self._biggest) for f, s, a in children]
            children_fitnesses_1 = group_fitness(
                remodelled_children, fit_fn_1, bound_estimation, self._emp_bounds_1
            )
            children_fitnesses_2 = group_fitness(
                remodelled_children, fit_fn_2, bound_estimation, self._emp_bounds_2
            )
            all_solutions = self._population + remodelled_children
            all_fitnesses_1 = self._fitnesses_1 + children_fitnesses_1
            all_fitnesses_2 = self._fitnesses_2 + children_fitnesses_2
            
            assert len(self._population) == self._pop_size
            assert len(self._fitnesses_1) == self._pop_size
            assert len(self._fitnesses_2) == self._pop_size

            fronts = non_dominated_sorting(
                all_solutions, all_fitnesses_1, all_fitnesses_2
            )
            
            
            survivors_idx = []
            for front in fronts:
                if len(survivors_idx) + len(front) < self._pop_size:
                    survivors_idx.extend(front)
                elif len(survivors_idx) + len(front) == self._pop_size:
                    survivors_idx.extend(front)
                    break
                else:
                    distance = crowding_distance(front, all_fitnesses_1, all_fitnesses_2)
                    descending_distance = sorted(front, key=lambda idx: distance[idx], reverse=True)
                    free = self._pop_size - len(survivors_idx)
                    survivors_idx.extend(descending_distance[:free])
                    break

            self._population = [all_solutions[s] for s in survivors_idx]
            self._fitnesses_1 = [all_fitnesses_1[s] for s in survivors_idx]
            self._fitnesses_2 = [all_fitnesses_2[s] for s in survivors_idx]


            ##################################
            ###### here ends selection !!!!!!
            ##################################
            self._initialise_islands()
            self._check_biggest()

            #############################################
            ####### IF NOT PRESTEP: bounds/ convergence 
            #############################################
            if not prestep:
                if bound_estimation:
                    self._fitnesses_1_pool.extend(self._fitnesses_1)
                    self._fitnesses_2_pool.extend(self._fitnesses_2)
                    if gen == generations-1:
                        self._emp_bounds_1 = self._bounds_estimation(self._fitnesses_1_pool)
                        self._emp_bounds_2 = self._bounds_estimation(self._fitnesses_2_pool)
                else:
                    self._estimate_convergence()

                    print(f"gen:{gen} | #topo:{len(self._islands)} | {round(self.avg_convergence(), 5)}")

        #########################################
        ####### IF NOT PRESTEP: ################
        #########################################
        # get the first front of the last generation
        # ⛔️ in a normalised space !!!
        # ⛔️ ASSUMPTION: updating self._population, self._fitnesses_1/2
        # during selection is done adding fronts in order!!!
        if not prestep:
            f1_length = len(fronts[0])
            f1_fitnesses_1 = self._fitnesses_1[:f1_length]
            f1_fitnesses_2 = self._fitnesses_2[:f1_length]
            best_y = normalise_objective(f1_fitnesses_1, self._emp_bounds_1) 
            best_x = normalise_objective(f1_fitnesses_2, self._emp_bounds_2)

            self._best_front = list(zip(best_x, best_y))
            



# -----––--- version with checkpoints -----––----

# class NSGA2():
#     """
#     ⛔️ bug in evolve.TOURNAMENT ~ CONVERGENCE ⛔️
#         - if evolution converges towards one archi only,
#           only one key available
    
#     multi-objective evolution based on Islands + original nsga-ii
#     obj_1 = classic net optimisation (MSELoss/CrossEntropy)
#     obj_2 = inference speed

#     remember:
#         - initialise islands????????
#         - need the right 'problem' when initialising the Island class!!!
#     """
#     def __init__(
#             self,
#             pop_size,
#             model,
#             data,
#             input_shape=(1, 28, 28),
#             interval=[1, 4], # small interval compared to pop_size? ⛔️ representativeness
#             problem = "AE"
#     ):
#         self._islands = None
#         self._data = data
#         self._input_shape = input_shape
#         self._model = model # needs to be a class, not an istance!
#         self._problem = problem
#         self._pop_size = pop_size
#         self._population = [
#             deepcopy(
#                 model(
#                     input_shape=input_shape,
#                     stride=random.randint(interval[0], interval[1])
#                 ).to(mydevice)
#             ) for i in range(pop_size)
#         ]

#         self._fit_fn_1 = model_fitness#(data, problem=problem) #model_fitness is HIGHER ORDER
#         self._fit_fn_2 = model_runtime#(data)
#         self._fitnesses_1 = None
#         self._fitnesses_2 = None # why was it missing⁉️⁉️⁉️⁉️⁉️⁉️⁉️
#         self._fitnesses_1_pool = []
#         self._fitnesses_2_pool = []
#         self._convergence = [] # list of lists: normalised distances per generation
#         self._best_model = None
#         self._best_convergence = None
#         self._emp_bounds_1 = None # empirical bounds per objective
#         self._emp_bounds_2 = None
#         self._best_front = None

#         self._biggest = max(
#             sum(param.numel() for param in m.parameters()) # ⛔️ will change mid run????
#             for m in self._population
#         )

#         # self._gen = 0
#         # self._max_gen = None
    
#     def _initialise_islands(self):
#         self._islands = defaultdict(list)
#         for m in self._population:
#             key = m.get_stride()
#             self._islands[key].append(m)
    
#     def _check_biggest(self):
#         current_biggest = max(
#             sum(param.numel() for param in m.parameters()) 
#                 for m in self._population
#         )
#         if current_biggest != self._biggest:
#             self._biggest = current_biggest
    
#     def _initialise_fitness(self):
#         self._fitness = group_fitness(self._population, self._fit_fn)
    
#     def _sample_loader(self, fraction):
#         full_idxs = list(range(len(self._data)))
#         if isinstance(self._data.targets, torch.Tensor):
#             labels = self._data.targets.numpy()
#         elif isinstance(self._data.targets, list):
#             labels = np.array(self._data.targets)
            
#         random_indices, _ = train_test_split(full_idxs, train_size=fraction, stratify=labels)
#         subset = Subset(self._data, indices=random_indices)

#         loader = DataLoader(
#             subset, batch_size=30, shuffle=True, pin_memory=True
#         )
#         return loader
    
#     def _bounds_estimation(self, fitnesses):
#         """update (or not) bounds"""
#         mino = np.percentile(fitnesses, 5)
#         maxo = np.percentile(fitnesses, 95)
#         return mino, maxo
    
#     def _estimate_convergence(self):
#         """
#         given the current population,
#         update the model closer to the ideal solution
#         and its distance from this ideal solution.
#         """
#         normalised_x = normalise_objective(self._fitnesses_2, self._emp_bounds_2) # x speed
#         normalised_y = normalise_objective(self._fitnesses_1, self._emp_bounds_1) # y fitness

#         # getting distance from ideal for each model 
#         distances = [] # and record in self._convergence as pop_avg per gen
#         for i in range(self._pop_size):
#             distances.append(convergence(normalised_x[i], normalised_y[i]))
#         self._convergence.append(distances)

#         # finding most balanced model (closest to ideal)
#         zipped = list(zip(self._population, distances))
#         ordered = sorted(zipped, key= lambda x: x[1])
#         best_model, best_convergence = ordered[0]
#         if self._best_model is None or self._best_convergence is None:
#             self._best_model = deepcopy(best_model)
#             self._best_convergence = best_convergence
#         else:
#             if best_convergence < self._best_convergence:
#                 self._best_model = deepcopy(best_model)
#                 self._best_convergence = best_convergence
    
#     def _clear_attributes(self, bound1, bound2):
#         self._fitnesses_1 = None
#         self._fitnesses_2 = None
#         self._convergence = [] # list of lists: normalised distances per generation
#         self._best_model = None
#         self._best_convergence = None
#         self._emp_bounds_1 = bound1 # empirical bounds per objective
#         self._emp_bounds_2 = bound2
    
#     # def _checkpoint(self, path):
#     #     """path: "name.pth" """
#     #     obj = {
#     #         "population": [(m.state_dict(), m.get_stride()) for m in self._population],
#     #         "problem": self._problem,
#     #         "pop_size": self._pop_size,
#     #         "convergence": self._convergence, 
#     #         "best_model": (self._best_model.state_dict(), self._best_model.get_stride()),
#     #         "best_convergence": self._best_convergence,
#     #         "emp_bounds_1": self._emp_bounds_1,
#     #         "emp_bounds_2": self._emp_bounds_2,
#     #         "biggest": self._biggest,
#     #         "gen": self._gen,
#     #         "max_gen": self._max_gen
#     #     }
#     #     dirpath = "./checkpoints"
#     #     os.makedirs(dirpath, exist_ok=True)
#     #     filepath = os.path.join(dirpath, path)
#     #     torch.save(obj, filepath)
    
#     # def _load_checkpoint(self, filepath, model):
#     #     checkpoint = torch.load(filepath)
        
#     #     population = checkpoint["population"]
#     #     self._population = []
#     #     for state, s in population:
#     #         m = model(stride=s).to(mydevice)
#     #         m.load_state_dict(state)
#     #         self._population.append(m)
#     #     self._initialise_islands()
#     #     self._pop_size = checkpoint["pop_size"]

#     #     self._problem = checkpoint["problem"]
#     #     self._convergence = checkpoint["convergence"]
        
#     #     weights, s = checkpoint["best_model"]
#     #     best = model(stride=s).to(mydevice)
#     #     best.load_state_dict(weights)
#     #     self._best_model = best
#     #     self._best_convergence = checkpoint["best_convergence"]

#     #     self._emp_bounds_1 = checkpoint["emp_bounds_1"]
#     #     self._emp_bounds_2 = checkpoint["emp_bounds_2"]
#     #     self._biggest = checkpoint["biggest"]
#     #     self._gen = checkpoint["gen"]
#     #     self._max_gen = checkpoint["max_gen"]

#     def get_best(self):
#         best = self._best_model, self._best_convergence
#         return best
    
#     def save_best(self, filepath):
#         best = {
#             "weights": self._best_model.state_dict(),
#             "convergence": self._best_convergence
#         }
#         torch.save(best, filepath)
    
#     def get_bounds(self):
#         return self._emp_bounds_1, self._emp_bounds_2
    
#     def set_bounds(self, b1, b2):
#         self._emp_bounds_1 = b1
#         self._emp_bounds_2 = b2
    
#     def get_best_front(self):
#         return self._best_front
    
#     def conv_in_time(self):
#         """returns list of avg.pop distance from ideal point per generation"""
#         return [sum(i)/len(i) for i in self._convergence]
    
#     def avg_convergence(self):
#         """get FINAL population convergence (mean convergence at last gen)"""
#         avg_convergence = sum(self._convergence[-1])/self._pop_size
#         return avg_convergence
    
#     def plot_convergence(self):
#         """"""
#         distances = self.conv_in_time()
        
#         _, ax = plt.subplots()
#         ax.plot(range(len(distances)), distances)
#         ax.set_xlabel("generation")
#         ax.set_ylabel("avg. distance from ideal solution")
#         plt.show()
    
#     def reset(self, model, pop_size, interval, bound1, bound2):
#         self._population = [
#             deepcopy(
#                 model(
#                     input_shape=self._input_shape,
#                     stride=random.randint(interval[0], interval[1])
#                 ).to(mydevice)
#             ) for i in range(pop_size)
#         ]
#         self._clear_attributes(bound1, bound2)
#         self._gen = 0
#         self._max_gen = None
    


#     def transfer(self, model, bound1, bound2, freeze=False):
#         """should I just swap the new pop in??"""
#         pop = []
#         for m in self._population:
#             weights = m.encoder.state_dict()
#             new = model(m, stride=m.get_stride()).to(mydevice)
#             new.encoder.load_state_dict(weights)

#             if freeze:
#                 for param in new.parameters():
#                     param.requires_grad = False

#             pop.append(new)
        
#         self._population = pop
#         self._clear_attributes(bound1, bound2)

    
#     def evolve(
#             self,
#             # save=False,
#             # load=False,
#             # checkpoint_path=None,
#             prestep=False,
#             bound_estimation=True,
#             generations=0,
#             subset_fraction=0.07,
#             m_prob=0.3
#     ):

#         # if load and checkpoint_path:
#         #     self._load_checkpoint(checkpoint_path, self._model)
        
#         # if self._max_gen is None:
#         #     self._max_gen = generations

#         # to_go = self._max_gen - self._gen
        
#         # for gen in range(to_go):
        
#         for gen in range(generations):

#             loader_sample = self._sample_loader(subset_fraction)
#             fit_fn_1 = self._fit_fn_1(loader_sample, self._problem)
#             fit_fn_2 = self._fit_fn_2(loader_sample)

            
#             self._fitnesses_1 = group_fitness( # clamped within emp_bounds
#                 self._population, fit_fn_1, bound_estimation, self._emp_bounds_1
#             )
#             self._fitnesses_2 = group_fitness( # clamped within emp_bounds
#                 self._population, fit_fn_2, bound_estimation, self._emp_bounds_2
#             )

#             ################################################
#             if gen == 0:
#                 self._initialise_islands()
#             # if not bound_estimation:
#             #     print(f"gen {gen} | topologies: {len(self._islands)} | {self.avg_convergence()}")
#             ################################################

#             # mating events, either within(more likely) or between(less likely)
#             children = [] # TOURNAMENT 🔥
#             self._check_biggest()
#             for _ in range(self._pop_size//2):
#                 if random.random() < 0.1 and len(self._islands)>= 2: # unlikely cross-species crossover 🔥
#                     random_keys = random.sample(list(self._islands.keys()), k=2)
#                     key1, key2 = random_keys[0], random_keys[1]
#                     pool1, pool2 = self._islands[key1], self._islands[key2]
#                     parent1, parent2 = random.choice(pool1), random.choice(pool2)
#                     parent1, parent2 = embed(parent1, self._biggest), embed(parent2, self._biggest)
#                 else: # regular intraspecies crossover 🔥
#                     key = random.choice(list(self._islands.keys()))
#                     pool = self._islands[key]
#                     if len(pool) == 1:
#                         parent1, parent2 = pool[0], deepcopy(pool[0])
#                         parent1, parent2 = embed(parent1, self._biggest), embed(parent2, self._biggest)
#                     elif len(pool) == 2:
#                         parent1, parent2 = pool[0], pool[1]
#                         parent1, parent2 = embed(parent1, self._biggest), embed(parent2, self._biggest)
#                     else:
#                         parents = random.sample(pool, k=2)
#                         parent1, parent2 = parents[0], parents[1]
#                         parent1, parent2 = embed(parent1, self._biggest), embed(parent2, self._biggest)
                
#                 self._check_biggest()

#                 child1, child2 = crossover(parent1, parent2)
#                 child1 = (mutate(child1[0]), child1[1], child1[2]) # mutate 50% of genes
#                 child2 = (mutate(child2[0]), child2[1], child2[2]) # mutate 50% of genes
                
#                 children.extend([child1, child2])

#                                     # remodel(f,s,a,biggest)–>model!
#             remodelled_children = [remodel(f, s, a, self._biggest) for f, s, a in children]
#             children_fitnesses_1 = group_fitness(
#                 remodelled_children, fit_fn_1, bound_estimation, self._emp_bounds_1
#             )
#             children_fitnesses_2 = group_fitness(
#                 remodelled_children, fit_fn_2, bound_estimation, self._emp_bounds_2
#             )
#             all_solutions = self._population + remodelled_children
#             all_fitnesses_1 = self._fitnesses_1 + children_fitnesses_1
#             all_fitnesses_2 = self._fitnesses_2 + children_fitnesses_2
            
#             assert len(self._population) == self._pop_size
#             assert len(self._fitnesses_1) == self._pop_size
#             assert len(self._fitnesses_2) == self._pop_size

#             fronts = non_dominated_sorting(
#                 all_solutions, all_fitnesses_1, all_fitnesses_2
#             )
            
            
#             survivors_idx = []
#             for front in fronts:
#                 if len(survivors_idx) + len(front) < self._pop_size:
#                     survivors_idx.extend(front)
#                 elif len(survivors_idx) + len(front) == self._pop_size:
#                     survivors_idx.extend(front)
#                     break
#                 else:
#                     distance = crowding_distance(front, all_fitnesses_1, all_fitnesses_2)
#                     descending_distance = sorted(front, key=lambda idx: distance[idx], reverse=True)
#                     free = self._pop_size - len(survivors_idx)
#                     survivors_idx.extend(descending_distance[:free])
#                     break

#             self._population = [all_solutions[s] for s in survivors_idx]
#             self._fitnesses_1 = [all_fitnesses_1[s] for s in survivors_idx]
#             self._fitnesses_2 = [all_fitnesses_2[s] for s in survivors_idx]

#             self._initialise_islands()

#             self._check_biggest()

#             #########################################
#             ####### IF NOT PRESTEP: ################
#             #########################################
#             if not prestep:
#                 if bound_estimation:
#                     self._fitnesses_1_pool.extend(self._fitnesses_1)
#                     self._fitnesses_2_pool.extend(self._fitnesses_2)
#                     if gen == generations-1:
#                         self._emp_bounds_1 = self._bounds_estimation(self._fitnesses_1_pool)
#                         self._emp_bounds_2 = self._bounds_estimation(self._fitnesses_2_pool)
#                 else:
#                     self._estimate_convergence()

#                     print(f"gen:{gen} | #topo:{len(self._islands)} | {round(self.avg_convergence(), 5)}")


#             # self._gen +=1 # ⛔️don't need it.. not checkpointing within run!!!!
#             #checkpoint only if NOT BOUND ESTIMATION
            

#             # if save and checkpoint_path:
#             #     assert os.path.isdir(checkpoint_path)
#             #     self._checkpoint(checkpoint_path)
        

#         #########################################
#         ####### IF NOT PRESTEP: ################
#         #########################################
#         # get the first front of the last generation
#         # ⛔️ in a normalised space !!!
#         # ⛔️ ASSUMPTION: updating self._population, self._fitnesses_1/2
#         # during selection is done adding fronts in order!!!
#         if not prestep:
#             f1_length = len(fronts[0])
#             f1_fitnesses_1 = self._fitnesses_1[:f1_length]
#             f1_fitnesses_2 = self._fitnesses_2[:f1_length]
#             best_y = normalise_objective(f1_fitnesses_1, self._emp_bounds_1) 
#             best_x = normalise_objective(f1_fitnesses_2, self._emp_bounds_2)

#             self._best_front = list(zip(best_x, best_y))
            