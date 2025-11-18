import random
import time
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
    sigma = flat.std().item()

    size = flat.numel()
    difference = biggest - size
    if difference % 2 == 0:
        lx_pad_size = difference // 2
        rx_pad_size = lx_pad_size
    else:
        lx_pad_size = difference // 2
        rx_pad_size = difference - lx_pad_size
    
    # ⛔️: every time embed() called, torch.random introduces randomness
    lx_padding = torch.normal(mu, sigma, (lx_pad_size,), dtype=flat.dtype, device=device)
    rx_padding = torch.normal(mu, sigma, (rx_pad_size,), dtype=flat.dtype, device=device)

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
def crossover(parent1: torch.Tensor, parent2: torch.Tensor)-> tuple[torch.Tensor, torch.Tensor]:
    """
    masked crossover between two flat models:
    Args:
       parent1: flat model
       parent2: flat model
    """
    flat1, s, a = parent1
    flat2, _, _ = parent2
    
    device = parent1.device 
    mask = torch.randint(0, 2, flat1.shape, dtype=torch.bool, device=device) # mask with zeroes and ones
    child1 = (torch.where(mask, flat1, flat2), s, a)
    child2 = (torch.where(mask, flat2, flat1), s, a)

    return child1, child2


def mutate(guy:torch.Tensor, m_chance=0.2, mode="small", m_rate=0.3) -> torch.Tensor:
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
                X, y = X.to(device), out(X, y).to(device)
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


def normalise_fitness(fitnesses: list, bound):
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

def model_runtime(data: DataLoader):
    """
    careful if using on GPU.. operations are asynchronous
    - torch.cuda.synchronise() ⁉️
    - synch for every time.time() ⁉️ (https://discuss.pytorch.org/t/bizzare-extra-time-consumption-in-pytorch-gpu-1-1-0-1-2-0/87843)

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
                )
            ) for i in range(pop_size)
        ]

        self._fit_fn_1 = model_fitness#(data, problem=problem) #model_fitness is HIGHER ORDER
        self._fit_fn_2 = model_runtime#(data)
        self._fitnesses_1 = None
        self._fitnesses_2 = None # why was it missing⁉️⁉️⁉️⁉️⁉️⁉️⁉️
        self._convergence = [] # list of lists: normalised distances per generation
        self._best_model = None
        self._best_convergence = None
        self._emp_bounds_1 = None # empirical bounds per objective
        self._emp_bounds_2 = None

        self._biggest = max(
            sum(param.numel() for param in m.parameters()) # ⛔️ will change mid run????
            for m in self._population
        )

        self._gen = 0
        self._max_gen = None
    
    def _initialise_islands(self):
        self._islands = defaultdict(list)
        for m in self._population:
            key = m.get_stride()
            self._islands[key].append(m)
    
    def _initialise_fitness(self):
        self._fitness = group_fitness(self._population, self._fit_fn)
    
    def _bounds_estimation(sel, fitnesses, bound):
        """update (or not) bounds"""
        mino, maxo = min(fitnesses), max(fitnesses)
        bounds = (min(mino, bound[0]), max(maxo, bound[1]))
        return bounds
    
    def _estimate_convergence(self):
        normalised_x = normalise_fitness(self._fitnesses_1, self._emp_bounds_1) # x fitness
        normalised_y = normalise_fitness(self._fitnesses_2, self._emp_bounds_2) # y speed
            
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
    
    def _checkpoint(self, filepath):
        obj = {
            "population": [(m.state_dict(), m.get_stride()) for m in self._population],
            "problem": self._problem,
            "pop_size": self._pop_size,
            "convergence": self._convergence, 
            "best_model": (self._best_model.state_dict(), self._best_model.get_stride()),
            "best_convergence": self._best_convergence,
            "emp_bounds_1": self._emp_bounds_1,
            "emp_bounds_2": self._emp_bounds_2,
            "biggest": self._biggest,
            "gen": self._gen,
            "max_gen": self._max_gen
        }
        torch.save(obj, filepath)
    
    def load_checkpoint(self, filepath, model):
        checkpoint = torch.load(filepath)
        
        population = checkpoint["population"]
        self._population = []
        for state, s in population:
            m = model(stride=s)
            m.load_state_dict(state)
            self._population.append(m)
        self._initialise_islands()
        self._pop_size = checkpoint["pop_size"]

        self._problem = checkpoint["problem"]
        self._convergence = checkpoint["convergence"]
        
        weights, s = checkpoint["best_model"]
        best = model(stride=s)
        best.load_state_dict(weights)
        self._best_model = best
        self._best_convergence = checkpoint["best_convergence"]

        self._emp_bounds_1 = checkpoint["emp_bounds_1"]
        self._emp_bounds_2 = checkpoint["emp_bounds_2"]
        self._biggest = checkpoint["biggest"]
        self._gen = checkpoint["gen"]
        self._max_gen = checkpoint["max_gen"]

    def get_best(self):
        best = self._best_model, self._best_convergence
        return best
    
    def save_best(self, filepath):
        best = {
            "weights": self._best_model.state_dict(),
            "convergence": self._best_convergence
        }
        torch.save(best, filepath)
    
    def conv_in_time(self):
        """returns list of avg.pop distance from ideal point per generation"""
        return [sum(i)/len(i) for i in self._convergence]
    
    def avg_convergence(self):
        """get FINAL population convergence (mean convergence at last gen)"""
        avg_convergence = sum(self._convergence[-1])/self._pop_size
        return avg_convergence
    
    def get_bounds(self):
        return self._emp_bounds_1, self._emp_bounds_2
    
    def plot_convergence(self):
        """ what the fuck do I want to plot? WHO KNOWS"""
        distances = self.conv_in_time()
        
        _, ax = plt.subplots()
        ax.plot(range(len(distances)), distances)
        ax.set_xlabel("generation")
        ax.set_ylabel("avg. distance from ideal solution")
        plt.show()
    
    def reset(self, model, pop_size, interval, bound1, bound2):
        self._population = self._population = [
            deepcopy(
                model(
                    input_shape=self._input_shape,
                    stride=random.randint(interval[0], interval[1])
                )
            ) for i in range(pop_size)
        ]
        self._fitnesses_1 = None
        self._fitnesses_2 = None
        self._convergence = [] # list of lists: normalised distances per generation
        self._best_model = None
        self._best_convergence = None
        self._emp_bounds_1 = bound1 # empirical bounds per objective
        self._emp_bounds_2 = bound2
        self._gen = 0
        self._max_gen = None

    def transfer(self, model, bound1, bound2, freeze=False):
        """should I just swap the new pop in??"""
        pop = []
        for m in self._population:
            weights = m.encoder.state_dict()
            new = model(m, stride=m.get_stride())
            new.encoder.load_state_dict(weights)

            if freeze:
                for param in new.parameters():
                    param.requires_grad = False

            pop.append(new)
        
        self._population = pop
        self._fitnesses_1 = None
        self._fitnesses_2 = None
        self._convergence = [] # list of lists: normalised distances per generation
        self._best_model = None
        self._best_convergence = None
        self._emp_bounds_1 = bound1 # empirical bounds per objective
        self._emp_bounds_2 = bound2

    
    def evolve(
            self,
            bound_estimation=True,
            generations=10,
            subset_fraction=0.07,
            report_jump=2,        # UNUSED ⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️
            m_prob=0.3
    ):
        if self._max_gen is None:
            self._max_gen = generations
        
        for gen in range(self._gen, self._max_gen):

            full_idxs = list(range(len(self._data)))
            if isinstance(self._data.targets, torch.Tensor):
                labels = self._data.targets.numpy()
            elif isinstance(self._data.targets, list):
                labels = np.array(self._data.targets)
            random_indices, _ = train_test_split(full_idxs, train_size=subset_fraction, stratify=labels)
            subset = Subset(self._data, indices=random_indices)

            train_loader = DataLoader(subset, batch_size=30)
            fit_fn_1 = self._fit_fn_1(train_loader, self._problem)
            fit_fn_2 = self._fit_fn_2(train_loader)

            ################################################
            if gen == 0:
                self._initialise_islands()
            print(f"gen {gen} | topologies: {len(self._islands)}")
            ################################################
            
            self._fitnesses_1 = group_fitness(self._population, fit_fn_1)
            self._fitnesses_2 = group_fitness(self._population, fit_fn_2)
            
            # mating events, either within(more likely) or between(less likely)
            children = [] # TOURNAMENT 🔥
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
                        # parent1, parent2 = pool[0], deepcopy(pool[0])
                        continue
                    else:
                        parents = random.sample(pool, k=2)
                        parent1, parent2 = parents[0], parents[1]
                        parent1, parent2 = embed(parent1, self._biggest), embed(parent2, self._biggest)
                
                child1, child2 = crossover(parent1, parent2)
                child1 = (mutate(child1[0]), child1[1], child1[2]) # mutate 50% of genes
                child2 = (mutate(child2[0]), child2[1], child2[2]) # mutate 50% of genes
                
                children.extend([child1, child2])

                                    # remodel(f,s,a,biggest)–>model!
            remodelled_children = [remodel(f, s, a, self._biggest) for f, s, a in children]
            children_fitnesses_1 = group_fitness(remodelled_children, fit_fn_1)
            children_fitnesses_2 = group_fitness(remodelled_children, fit_fn_2)
            all_solutions = self._population + remodelled_children
            all_fitnesses_1 = self._fitnesses_1 + children_fitnesses_1
            all_fitnesses_2 = self._fitnesses_2 + children_fitnesses_2
            
            
            fronts = non_dominated_sorting(
                all_solutions, all_fitnesses_1, all_fitnesses_2
            )
            
            
            solutions = []
            for front in fronts:
                if len(solutions) + len(front) < self._pop_size:
                    solutions.extend(front)
                elif len(solutions) + len(front) == self._pop_size:
                    solutions.extend(front)
                    break
                else:
                    distance = crowding_distance(front, all_fitnesses_1, all_fitnesses_2)
                    descending_distance = sorted(front, key=lambda idx: distance[idx], reverse=True)
                    free = self._pop_size - len(solutions)
                    solutions.extend(descending_distance[:free])
                    break

            self._population = [all_solutions[s] for s in solutions]
            self._fitnesses_1 = [all_fitnesses_1[s] for s in solutions]
            self._fitnesses_2 = [all_fitnesses_2[s] for s in solutions]


            self._initialise_islands()


            current_biggest = max(
                sum(param.numel() for param in m.parameters()) 
                for m in self._population
            )
            if current_biggest != self._biggest:
                self._biggest = current_biggest


            if bound_estimation:
                if gen == 0:
                    b1 = (min(self._fitnesses_1), max(self._fitnesses_1))
                    b2 = (min(self._fitnesses_2), max(self._fitnesses_2))
                    bounds1 = self._bounds_estimation(self._fitnesses_1, b1)
                    bounds2 = self._bounds_estimation(self._fitnesses_2, b2)
                    self._emp_bounds_1 = bounds1
                    self._emp_bounds_2 = bounds2
                else:
                    bounds1 = self._bounds_estimation(self._fitnesses_1, self._emp_bounds_1)
                    bounds2 = self._bounds_estimation(self._fitnesses_2, self._emp_bounds_2)
                    self._emp_bounds_1 = bounds1
                    self._emp_bounds_2 = bounds2
            else:
                self._estimate_convergence()

                #checkpoint only if NOT BOUND ESTIMATION
                self._gen +=1
                # self._checkpoint(f"./checkpoints/nsga_{gen}_")