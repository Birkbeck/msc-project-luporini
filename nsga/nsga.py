from copy import deepcopy
import random
import math
from collections import defaultdict

from operators import crossover, mutate
from utils import flatten, embed, remodel, convergence, euclidean
from fitness import model_fitness, model_runtime, normalise_objective, group_fitness

import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, Subset



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
            problem="AE",
            device=None
    ):
        self._islands = None
        self._data = data
        self._input_shape = input_shape
        self._model = model # needs to be a class, not an istance!
        self._problem = problem
        self._device = device
        self._pop_size = pop_size
        self._population = [
            deepcopy(
                model(
                    input_shape=input_shape,
                    stride=random.randint(interval[0], interval[1])
                ).to(device)
            ) for i in range(pop_size)
        ]

        self._biggest = max(
            sum(param.numel() for param in m.parameters()) # ⛔️ will change mid run????
            for m in self._population
        )

        self._emp_bounds_1 = None # empirical bounds per objective
        self._emp_bounds_2 = None

        self._fit_fn_1 = model_fitness#(data, problem=problem) #model_fitness is HIGHER ORDER
        self._fit_fn_2 = model_runtime#(data)
        self._fitnesses_1 = None
        self._fitnesses_2 = None 
        self._fitnesses_1_pool = []
        self._fitnesses_2_pool = []
        
        self._convergence = [] # list of lists: normalised distances per generation
        self._deltas = [] # list of ints: Deb's ∆ per generation
        self._best_model = None
        self._best_convergence = None
        self._best_front = None

    
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
    
    def _bounds_estimation(self, fitnesses)->tuple:
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
        normalised_1 = normalise_objective(self._fitnesses_1, self._emp_bounds_1) # y fitness
        normalised_2 = normalise_objective(self._fitnesses_2, self._emp_bounds_2) # x speed
        
        # getting distance from ideal for each model 
        distances = [] # and record in self._convergence 
        for i in range(self._pop_size):
            distances.append(convergence(normalised_1[i], normalised_2[i]))
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
    
    def _estimate_spread(self, fits1:list, fits2:list)->float:
        """
        ⛔️ fitnesses already normalised ⛔️

        Deb's delta from the original paper (2002)
        normalised crowding distance of the last non-dominated front
        - trivial for one point
        - poorly informative for two points(∆ = 2/3 d, d=distance1-2)
        """
        if len(fits1) >= 3:
            points = sorted(zip(fits1, fits2), key=lambda x: x[0]) # sort by fits1
            N = len(points)

            distances = [euclidean(points[i], points[i+1]) for i in range(N - 1)]

            d_f, d_l = distances[0], distances[-1]
            avg_d = sum(distances) / len(distances)

            numerator = d_f + d_l + sum(abs(d - avg_d) for d in distances)
            denominator = d_f + d_l + (N - 1) * avg_d

            delta = numerator / denominator if denominator != 0 else float("nan")
        
            self._deltas.append(delta)
        else:
            self._deltas.append(float("nan")) # ⁉️

        
    def _clear_attributes(self, bound1, bound2):
        self._fitnesses_1 = None
        self._fitnesses_2 = None
        self._convergence = [] # list of lists: normalised distances per generation
        self._best_model = None
        self._best_convergence = None
        self._emp_bounds_1 = bound1 # empirical bounds per objective
        self._emp_bounds_2 = bound2
    
    def get_bounds(self):
        return self._emp_bounds_1, self._emp_bounds_2
    
    def set_bounds(self, b1, b2):
        self._emp_bounds_1 = b1
        self._emp_bounds_2 = b2
    
    def avg_convergence(self):
        """returns list of avg.pop distance from ideal point per generation"""
        return [sum(i)/len(i) for i in self._convergence]
    
    def final_convergence(self):
        return self.avg_convergence()[-1]
    
    def get_deltas(self):
        return self._deltas
    
    def final_delta(self):
        return self._deltas[-1]

    def get_best(self):
        best = self._best_model, self._best_convergence
        return best
    
    def save_best(self, filepath):
        best = {
            "weights": self._best_model.state_dict(),
            "convergence": self._best_convergence
        }
        torch.save(best, filepath)
    
    def get_best_front(self):
        return self._best_front

    def get_transfer_pop(self, to_model, in_shape, classes, freeze=False):
        """should I just swap the new pop in??"""
        pop = []
        for m in self._population:
            weights = m.encoder.state_dict()
            new = to_model(input_shape=in_shape, stride=m.get_stride(), classes=classes).to(self._device)
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
            if prestep:
                print(f" - gen {gen}")

            loader_sample = self._sample_loader(subset_fraction)
            fit_fn_1 = self._fit_fn_1(loader_sample, self._problem)
            fit_fn_2 = self._fit_fn_2(loader_sample)

            
            self._fitnesses_1 = group_fitness( # clamped within emp_bounds
                self._population, fit_fn_1, self._emp_bounds_1
            )
            self._fitnesses_2 = group_fitness( # clamped within emp_bounds
                self._population, fit_fn_2, self._emp_bounds_2
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
                    parent1, parent2 = embed(parent1, self._biggest, self._device), embed(parent2, self._biggest, self._device)
                else: # regular intraspecies crossover 🔥
                    key = random.choice(list(self._islands.keys()))
                    pool = self._islands[key]
                    if len(pool) == 1:
                        parent1, parent2 = pool[0], deepcopy(pool[0])
                        parent1 = embed(parent1, self._biggest, self._device)
                        parent2 = embed(parent2, self._biggest, self._device)
                    elif len(pool) == 2:
                        parent1, parent2 = pool[0], pool[1]
                        parent1 = embed(parent1, self._biggest, self._device)
                        parent2 = embed(parent2, self._biggest, self._device)
                    else:
                        parents = random.sample(pool, k=2)
                        parent1, parent2 = parents[0], parents[1]
                        parent1 = embed(parent1, self._biggest, self._device)
                        parent2 = embed(parent2, self._biggest, self._device)
                
                self._check_biggest()

                child1, child2 = crossover(parent1, parent2)
                child1 = (mutate(child1[0], m_rate=m_r, m_chance=m_c), child1[1], child1[2]) # mutate 50% of genes
                child2 = (mutate(child2[0], m_rate=m_r, m_chance=m_c), child2[1], child2[2]) # mutate 50% of genes
                
                children.extend([child1, child2])

                                    # remodel(f,s,a,biggest)–>model!
            remodelled_children = [remodel(f, s, a, self._biggest) for f, s, a in children]
            children_fitnesses_1 = group_fitness(
                remodelled_children, fit_fn_1, self._emp_bounds_1
            )
            children_fitnesses_2 = group_fitness(
                remodelled_children, fit_fn_2, self._emp_bounds_2
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
                    avg_conv = sum(self._convergence[-1]) / len(self._convergence[-1])
                    
                    f1 = fronts[0] # last non-dominated front
                    f1_length = len(f1)
                    f1_fitnesses_1 = self._fitnesses_1[:f1_length]
                    f1_fitnesses_2 = self._fitnesses_2[:f1_length]
                    best_1 = normalise_objective(f1_fitnesses_1, self._emp_bounds_1) 
                    best_2 = normalise_objective(f1_fitnesses_2, self._emp_bounds_2)
                    
                    self._estimate_spread(best_1, best_2) # Deb's ∆


                    print(f"gen:{gen} | #topo:{len(self._islands)} | avg_conv: {round(avg_conv, 3)} | ∆: {round(self._deltas[-1], 3)}")

        #########################################
        ####### IF NOT PRESTEP: ################
        #########################################
        # get the optimal front of the last generation
        # ⛔️ in a normalised space !!! FOR PLOTTING 🔥
        # ⛔️ ASSUMPTION: updating self._population, self._fitnesses_1/2
        # during selection is done adding fronts in order!!!
        if not prestep and not bound_estimation:
            self._best_front = (best_1, best_2)