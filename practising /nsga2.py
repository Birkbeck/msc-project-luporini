import random
import time
from copy import deepcopy
from collections import defaultdict
import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from genalgo import mutate, group_fitness, model_fitness
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

class NSGA2():
    """
    multi-objective evolution based on Islands + original nsga-ii
    obj_1 = classic net optimisation (MSELoss/CrossEntropy)
    obj_2 = inference speed

    remember: need the right 'problem' when initialising the Island class!!!
    """
    def __init__(
            self,
            model,
            pop_size,
            data: DataLoader,
            interval=[1, 4], # small interval compared to pop_size? ⛔️ representativeness
            problem = "AE"
    ):
        self._data = data
        self._pop_size = pop_size
        self._model = model # needs to be a class, not an istance!
        self._population = [deepcopy(model(stride=random.randint(interval[0], interval[1]))) for i in range(pop_size)]
        
        self._fit_fn_1 = model_fitness(data, problem=problem) #model_fitness is HIGHER ORDER
        self._fit_fn_2 = model_runtime(data)
        self._fitnesses_1 = group_fitness(self._population, self._fit_fn_1)
        self._fitnesses_2 = group_fitness(self._population, self._fit_fn_2)
        
        self._biggest = max(
            sum(param.numel() for param in m.parameters()) # ⛔️ will change mid run????
            for m in self._population
        )

    
    def nsga_selection(self, generations=10, report_jump=2, m_prob=0.3):
        
        # HELPER FUNCTION
        def _non_dominated_sorting(whole, fits1, fits2):
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
        def _crowding_distance(front, *objectives):
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
        
        # EVOLUTION LOOP
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

                                    # remodel(f,s,a,biggest)–>model!
            remodelled_children = [remodel(f, s, a, self._biggest) for f, s, a in children]
            children_fitnesses_1 = group_fitness(remodelled_children, self._fit_fn_1)
            children_fitnesses_2 = group_fitness(remodelled_children, self._fit_fn_2)
            all_solutions = self._population + remodelled_children
            all_fitnesses_1 = self._fitnesses_1 + children_fitnesses_1
            all_fitnesses_2 = self._fitnesses_2 + children_fitnesses_2
            
            
            fronts = _non_dominated_sorting(
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
                    distance = _crowding_distance(front, all_fitnesses_1, all_fitnesses_2)
                    descending_distance = sorted(front, key=lambda idx: distance[idx], reverse=True)
                    free = self._pop_size - len(solutions)
                    solutions.extend(descending_distance[:free])
                    break

            self._population = [all_solutions[s] for s in solutions]
            self._fitnesses_1 = [all_fitnesses_1[s] for s in solutions]
            self._fitnesses_2 = [all_fitnesses_2[s] for s in solutions]

            current_biggest = max(
                sum(param.numel() for param in m.parameters()) 
                for m in self._population
            )
            if current_biggest != self._biggest:
                self._biggest = current_biggest
            
            # if (i+1) % report_jump == 0: #⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️
            print(f"{gen+1}th gen | avg. population finess: {self.avg_fitness()}")
            

    def avg_fitnesses(self):
        fitnesses = [i for i in self._fitnesses_1 if i is not None]
        if not fitnesses:
            return None
        return np.mean(fitnesses)              
