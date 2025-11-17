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
from genalgo import mutate, group_fitness, model_fitness, normalise_fitness
from islands import embed, remodel, crossover

def model_runtime(data: DataLoader):
    """
    careful if using on GPU.. operations are asynchronous
    - torch.cuda.synchronise() ⁉️

    ALSO

    - just evaluate on representative batch ⁉️
    - return len(data)/interval ⁉️
    """
    def speed(model):
        model.eval()
        with torch.no_grad():
            
            start = time.time()
            for X, _ in data:
                _ = model(X)
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

def initialise_population(model, size, intervallo:tuple):
    return [
        deepcopy(
            model(stride=random.randint(intervallo[0], intervallo[1]))
        ) for i in range(size)
    ]

# HELPER FUNCTIONS for evaluation convergence and spread⛔️
def convergence(*fits):
    """for a model: Euclidean distance from ideal s in nD"""
    return math.sqrt(sum((f - 1)**2 for f in fits))

        
# def spread()




class NSGA2():
    """
    multi-objective evolution based on Islands + original nsga-ii
    obj_1 = classic net optimisation (MSELoss/CrossEntropy)
    obj_2 = inference speed

    remember:
        - initialise islands????????
        - need the right 'problem' when initialising the Island class!!!
    """
    def _initialise_islands(self):
        self._islands = defaultdict(list)
        for m in self._population:
            key = m.get_stride()
            self._islands[key].append(m)
    
    def _initialise_fitness(self):
        self._fitness = group_fitness(self._population, self._fit_fn)
    
    def _bounds_estimation(sel, fitnesses, bound):
        mino, maxo = min(fitnesses), max(fitnesses)
        bounds = (min(mino, bound[0]), max(maxo, bound[1]))
        return bounds
    
    # def _normalise_fitnesses(self):
    #     # normalise fitness space between 0, 1 using empirical bounds
    #     mino1, maxo1 = self._emp_bounds_1[0], self._emp_bounds_1[1]
    #     mino2, maxo2 = self._emp_bounds_2[0], self._emp_bounds_2[1]
    #     normalised_x = normalise_fitness(self._fitnesses_1, mino1, maxo1) # x fitness
    #     normalised_y = normalise_fitness(self._fitnesses_2, mino2, maxo2) # y speed
    #     return normalised_x, normalised_y
    
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
    
    def __init__(
            self,
            pop_size,
            model,
            data,
            bound1=(0.0, 0.0),
            bound2=(0.0, 0.0),
            interval=[1, 4], # small interval compared to pop_size? ⛔️ representativeness
            problem = "AE"
    ):
        self._islands = None
        self._data = data
        self._model = model # needs to be a class, not an istance!
        self._problem = problem
        self._population = initialise_population(model, pop_size, interval)
        self._pop_size = pop_size

        self._fit_fn_1 = model_fitness#(data, problem=problem) #model_fitness is HIGHER ORDER
        self._fit_fn_2 = model_runtime#(data)
        self._fitnesses_1 = None
        self._fitnesses_2 = None
        self._convergence = [] # list of lists: normalised distances per generation
        self._best_model = None
        self._best_convergence = None
        self._emp_bounds_1 = bound1 # empirical bounds per objective
        self._emp_bounds_2 = bound2

        self._biggest = max(
            sum(param.numel() for param in m.parameters()) # ⛔️ will change mid run????
            for m in self._population
        )

    def get_best(self):
        best = self._best_model, self._best_convergence
        return best
    
    def get_avg_convergence(self):
        """get FINAL population convergence"""
        avg_convergence = sum(self._convergence[-1])/self._pop_size
        return avg_convergence
    
    def get_bounds(self):
        return self._emp_bounds_1, self._emp_bounds_2
    
    def save_best(self, filepath):
        best = {
            "weights": self._best_model.state_dict(),
            "convergence": self._best_convergence
        }
        torch.save(best, filepath)

    
    def evolve(
            self,
            bound_estimation=True,
            generations=10,
            subset_fraction=0.07,
            report_jump=2,        # UNUSED ⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️
            m_prob=0.3
    ):
        
        for gen in range(generations):

            full_idxs = list(range(len(self._data)))
            labels = self._data.targets.numpy()
            random_indices, _ = train_test_split(full_idxs, train_size=subset_fraction, stratify=labels)
            subset = Subset(self._data, indices=random_indices)

            train_loader = DataLoader(subset, batch_size=30)
            fit_fn_1 = self._fit_fn_1(train_loader, self._problem)
            fit_fn_2 = self._fit_fn_2(train_loader)

            # if first gen:
            # initialise self._islands
            # initialise self._fitnesses coz can't add list + None later
            if gen == 0:
                self._initialise_islands()
            
            self._fitnesses_1 = group_fitness(self._population, fit_fn_1)
            self._fitnesses_2 = group_fitness(self._population, fit_fn_2)

            # checking topologies.. changing through generations ⁉️
            for key in sorted(self._islands):
                value = self._islands[key]
                print(f"{key}: {len(value)} models")
            print(f"\nwith {self._pop_size} individuals total")
            

            # mating events, either within(more likely) or between(less likely)
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
                bounds1 = self._bounds_estimation(self._fitnesses_1, self._emp_bounds_1)
                bounds2 = self._bounds_estimation(self._fitnesses_2, self._emp_bounds_2)
                self._emp_bounds_1 = bounds1
                self._emp_bounds_2 = bounds2
            else:
                self._estimate_convergence()


    # def plot_evolution(self, nrows, ncols, figsize):
        # """ what the fuck do I want to plot? WHO KNOWS"""
        # # plotting the current population in 2D fitness landscape
        # _, axes = plt.subplots(nrows, ncols, figsize)
        
        # for i in range(len(self._convergence)):
        #     # norm_x = normalise_fitness(self._fitnesses_1, self._emp_bounds_1) # x fitness
        #     # norm_y = normalise_fitness(self._fitnesses_2, self._emp_bounds_2) # y speed
            
        #     avg_conv = sum(self._convergence[i])/len(self._convergence[i])
        # colour = [m._stride for m in self._population] # ⁉️
        # axes[i, 0].scatter(x=norm_x, y=norm_y, c=colour) # ⁉️
        # axes[i, 0].set_aspect("equal", adjustable="box")


        # plt.show()   
            
             
