from copy import deepcopy
import random
import math
from collections import defaultdict

from .operators import crossover, mutate
from .utils import flatten, embed, remodel, convergence, euclidean
from .fitness import model_fitness, model_runtime, normalise_objective, group_fitness

import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, Subset



def non_dominated_sorting(whole:list, fits1, fits2):
    """
    Performs nondominated sorting of population indices based on two fitness lists
    whole = population

    careful:
    - dom_counts: list of integers (idx = )
    - dominateds: list of lists (idx = by whom)
    - dominated: list 
    """
    # HELPER FUNCTION
    def _dominates(i, j):
        """
        Computes dominance between solutions.

        MAXIMISATION formulation: i dominates j if neither fitness worse and at least one better
        "neither fitness worse" F1i >= F1j AND F2i >= F2j
        "at least one better" F1i > F1j OR F2i > F2j

        Args:
            i [int]: index of solution i in population
            j [int]: index of solution j in population
        
        Returns:
            bool: i dominates j ?
        
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
    computes the crowding distance for each solution in a given front
    (COULD CHANGE TO TWO OBJECTIVES FOR CONSISTENCY)

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
    Multi-objective NSGA-II-inspired evolution supporting heterogeneous populations.
    
    Here, it is referred to as an 'island' genetic algorithm (islands = species by complexity).

    Objectives:
        1. task fitness (MSELoss/CrossEntropy)
        2. runtime speed

    Args:
        pop_size [int]: population size
        model [torch.nn.Module]: classifier class (not instance!)
        train_data [DataLoader]: train data
        test_data [DataLoader]: test data
        input_shape [tuple]: (C, H, W) (e.g., MNIST (1, 28, 28))
        interval [tuple]: discrete stride interval
        problem [str]: "regression", "AE" or "classification"
        device [torch.device]: suitable device, default = cuda
    """
    def __init__(
            self,
            pop_size,
            model,
            train_data,
            test_data,
            input_shape=(1, 28, 28),
            interval=[1, 4], # small interval compared to pop_size? ⛔️ representativeness
            problem="AE",
            device=None
    ):
        self._islands = None
        self._train_data = train_data
        self._test_data = test_data
        self._input_shape = input_shape
        self._model = model
        self._problem = problem
        self._device = device
        self._pop_size = pop_size

        # initialise population 
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
        self._fitnesses_1_pool = [] # pool of fitnesses for bound estimation
        self._fitnesses_2_pool = [] # pool of fitnesses for bound estimation
        
        self._val_fitnesses = [] # list of avg. val fit per gen
        self._convergence = [] # list of avg. pop_conv per gen
        self._deltas = [] # list of Deb's ∆ per gen (ints)
        self._best_model = None
        self._best_front = None

    
    def _initialise_islands(self):
        """ Organise population into islands (species). """
        self._islands = defaultdict(list)
        for m in self._population:
            key = m.get_stride()
            self._islands[key].append(m)
    
    def _check_biggest(self):
        """ Check if biggest is still biggest and, in case, update."""
        current_biggest = max(
            sum(param.numel() for param in m.parameters()) 
                for m in self._population
        )
        if current_biggest != self._biggest:
            self._biggest = current_biggest
    

    def _initialise_fitness(self):
        self._fitness = group_fitness(self._population, self._fit_fn)
    

    def _trainval_loaders(self, fraction):
        """ Split data into train and validation DataLoaders."""
        full_idxs = list(range(len(self._train_data)))
        if isinstance(self._train_data.targets, torch.Tensor):
            labels = self._train_data.targets.numpy()
        elif isinstance(self._train_data.targets, list):
            labels = np.array(self._train_data.targets)  #need np.arrays for stratify
            
        train_indices, remaining = train_test_split(full_idxs, train_size=fraction, stratify=labels)
        train_subset = Subset(self._train_data, indices=train_indices)

        remaining_labels = labels[remaining]
        val_indices, _ = train_test_split(remaining, train_size=0.5, stratify=remaining_labels)
        val_subset = Subset(self._train_data, indices=val_indices)


        train_loader = DataLoader(
            train_subset, batch_size=30, shuffle=True, pin_memory=True
        )
        val_loader = DataLoader(
            val_subset, batch_size=30, shuffle=True, pin_memory=True
        )
        return train_loader, val_loader
    

    def _bounds_estimation(self, fitnesses)->tuple:
        """
        Given a pool of fitnesses, estimate bounds using 5th and 95th percentiles.
        """
        mino = np.percentile(fitnesses, 5)
        maxo = np.percentile(fitnesses, 95)
        return mino, maxo
    
    def _update_m_rate(self, m_c): #close:int, far:int): # close=3, far=10
        """ 
        NOT USED!!!

        Update m_rate based on how recent generations
        are doing compared to more distant generations
        - if recent generations improve, lower mutation
        - if not, increase mutation

        Args:
            m_c [float]: current mutation rate
        
        Returns:
            new_rate [float]
        """
        def avg_close_far(fit, what):
            """
            HELPER: Compute average performance over a number of generations.
            """
            if what == "convergence":
                recent = fit[-2:] if len(fit) >= 5 else fit[:] # pop_avgs last 2 gens
                distant = fit[-5:-2] if len(fit) >= 10 else fit[:-len(recent)] # pop_avgs previous 3 gens
                
            else: # avg_validations are collected every 3 generations!!!!
                recent = fit[-2:] if len(fit) >= 2 else fit[:] # last two avg_val
                distant = fit[-6:-2] if len(fit) >= 6 else fit[:-len(recent)] # previous 4 avg_vals
            
            recent_avg = sum(recent) / len(recent) if recent else 1.0 # avg conv over recent gens
            distant_avg = sum(distant) / len(distant) if distant else 1.0
            return  recent_avg, distant_avg

        
        c_close, c_far = avg_close_far(self._convergence, "convergence")
        c_factor = c_close / c_far if c_far != 0 else 1.0 # scale down if conv getting smaller

        v_close, v_far = avg_close_far(self._val_fitnesses, "validation")
        v_factor = v_far / v_close if v_close != 0 else 1.0 # scale down if val getting bigger (better generalisation)

        new_rate = m_c * c_factor * v_factor
        
        new_rate = max(min(new_rate, 0.4), 0.01)
        
        return new_rate
    
    # ------------------------------------------
    # convergence and diversity estimation 
    # ----------------------------------------
    def _estimate_convergence(self):
        """
        Given the current population, get average distance from utopia point.
        """
        normalised_1 = normalise_objective(self._fitnesses_1, self._emp_bounds_1) # y fitness
        normalised_2 = normalise_objective(self._fitnesses_2, self._emp_bounds_2) # x speed
        
        # getting distance from utopia for each model 
        distances = [] # and record in self._convergence 
        for i in range(self._pop_size):
            distances.append(convergence(normalised_1[i], normalised_2[i]))
        
        gen_avg = sum(distances)/len(distances)
        self._convergence.append(gen_avg)

        return gen_avg


    def _estimate_spread(self, fits1:list, fits2:list)->float:
        """
        (On  already normalised fitnesses)

        Compute Deb's ∆ (Deb et al., 2002) to measure diversity on nondominated front.

        ∆ is a normalised crowding distance measure:
        - trivial for one point
        - poorly informative for two points(∆ = 2/3 d, d=distance1-2)
        - needs more points!
        Here, computed if fronts includes at least 3 solutions.
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

    # -------------------------------
    # getter and setter methods
    # -----------------------------
    
    # get and set empirical bounds 
    def get_bounds(self):
        return self._emp_bounds_1, self._emp_bounds_2
    
    def set_bounds(self, b1, b2):
        self._emp_bounds_1 = b1
        self._emp_bounds_2 = b2
    

    # get the list of avg convergence per gen
    def get_convergence(self):
        return self._convergence
    
    # get the list of delta per gen
    def get_deltas(self):
        return self._deltas
    
    # get the list of avg val performance per gen
    def get_val_fitness(self):
        return self._val_fitnesses
    

    def get_best(self):
        best = self._best_model #(model, val, fit)
        return best
    
    # save best model
    def save_best(self, filepath):
        best = {
            "weights": self._best_model.state_dict()
        }
        torch.save(best, filepath)
    
    def get_best_front(self):
        return self._best_front

    # ----------------------------------------------
    # transfer weights from autoencode population
    # ---------------------------------------------
    def transfer_popV2(self, pop, to_model, in_shape, classes, freeze=False):
        """
        Transfer the weights.

        Use each model in population as template and copy AE parameter one by one.
        """
        finalpop = []
        for m in pop:
            weights = m.encoder.state_dict()
            new = to_model(input_shape=in_shape, stride=m.get_stride(), classes=classes).to(self._device)
            new.encoder.load_state_dict(weights)

            if freeze:
                for param in new.parameters():
                    param.requires_grad = False

            finalpop.append(new)
        
        self._population = finalpop
    



    def evolve(
            self,
            prestep=False,
            bound_estimation=True,
            generations=0,
            subset_fraction=0.07,
            inter_r=0.01,
            m_r_min=0.01,
            m_r_max=0.2,
            m_r_decay = True,
            m_s=0.3,
            power=2,
            m_mode="small"
    ):
        """
        NSGA evolutionary loop.

        Args:
            prestep [bool]: condition, AE initialisation if True
            bound_estimation [bool]: is this a bound estimation run?
            generations [int]: number of generations
            subset_fraction [float]: subset fraction
            inter_r [float]: cross-species crossover rate
            m_r_min [float]: lower bound for mutation rate
            m_r_max [float]: upper bound for mutation rate
            m_r_decay [bool]: use decay?
            m_s [float]: Gaussian mutation scaler
            power [int]: decay curve parameter
            m_mode [str]: what kind of mutation?
        """
        for gen in range(generations):
            
            # ---------------------------------------------------------------
            # initialise train and validatio loaders, and fitness functions
            # ---------------------------------------------------------------
            train_loader, val_loader = self._trainval_loaders(subset_fraction)
            
            fit_fn_1 = self._fit_fn_1(train_loader, self._problem)
            fit_fn_2 = self._fit_fn_2(train_loader)

            # compute both objective values for parents (beginning of generation)
            self._fitnesses_1 = group_fitness( # clamped within emp_bounds if available
                self._population, fit_fn_1, self._emp_bounds_1
            )
            self._fitnesses_2 = group_fitness( # clamped within emp_bounds if available
                self._population, fit_fn_2, self._emp_bounds_2
            )

            ################################################
            if gen == 0:
                self._initialise_islands()
            ################################################

            if m_r_decay:
                m_r = m_r_min + (m_r_max - m_r_min)*(1 - (gen/(generations - 1))**power)

            # ------------------------------------------------------------------------
            # mating events, within species (more likely) or between species (less likely)
            # ------------------------------------------------------------------------
            children = [] # TOURNAMENT 🔥
            self._check_biggest()
            interspecies = 0
            for _ in range(self._pop_size//2):
                # unlikely cross-species crossover 
                if random.random() < inter_r and len(self._islands)>= 2:
                    random_keys = random.sample(list(self._islands.keys()), k=2)
                    key1, key2 = random_keys[0], random_keys[1]
                    pool1, pool2 = self._islands[key1], self._islands[key2]
                    parent1, parent2 = random.choice(pool1), random.choice(pool2)
                    parent1, parent2 = embed(parent1, self._biggest, self._device), embed(parent2, self._biggest, self._device)
                    interspecies += 1
                else:
                    # regular intraspecies crossover 
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
                    
                
                self._check_biggest() # just make sure biggest is still biggest

                # --------------------
                # genetic operators
                # -------------------
                child1, child2 = crossover(parent1, parent2)
                child1 = (mutate(child1[0], m_mode, m_rate=m_r, m_strength=m_s), child1[1], child1[2]) 
                child2 = (mutate(child2[0], m_mode, m_rate=m_r, m_strength=m_s), child2[1], child2[2]) 
                
                children.extend([child1, child2])

            # ----------------------------------------------
            # remodel children and compute fitness values
            # ----------------------------------------------
                                    # remodel(f,s,a,biggest)–>model!
            remodelled_children = [remodel(f, s, a, self._biggest) for f, s, a in children]
            children_fitnesses_1 = group_fitness(
                remodelled_children, fit_fn_1, self._emp_bounds_1
            )
            children_fitnesses_2 = group_fitness(
                remodelled_children, fit_fn_2, self._emp_bounds_2
            )

            # define intermediate population
            all_solutions = self._population + remodelled_children
            all_fitnesses_1 = self._fitnesses_1 + children_fitnesses_1
            all_fitnesses_2 = self._fitnesses_2 + children_fitnesses_2
            
            assert len(self._population) == self._pop_size
            assert len(self._fitnesses_1) == self._pop_size
            assert len(self._fitnesses_2) == self._pop_size

            # ------------------
            # NSGA OPERATIONS
            # ------------------

            # sorting
            fronts = non_dominated_sorting(
                all_solutions, all_fitnesses_1, all_fitnesses_2
            )
            
            # next generation building based on ranks and crowding distance
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
                    available = self._pop_size - len(survivors_idx)
                    survivors_idx.extend(descending_distance[:available])
                    break
            
            # update population and fitnesses
            self._population = [all_solutions[s] for s in survivors_idx]
            self._fitnesses_1 = [all_fitnesses_1[s] for s in survivors_idx]
            self._fitnesses_2 = [all_fitnesses_2[s] for s in survivors_idx]

            # reinitialise islands and biggest
            self._initialise_islands()
            self._check_biggest()

            # ------------------------------------------------
            # bound estimation, convergence and validation
            # ------------------------------------------------
            if prestep:
                print(f"gen {gen}")
            
            else:
                # if bound estimation, collect fitnesses for percentile computation
                if bound_estimation:
                    print(f"gen {gen}")

                    self._fitnesses_1_pool.extend(self._fitnesses_1)
                    self._fitnesses_2_pool.extend(self._fitnesses_2)
                    if gen == generations-1:
                        self._emp_bounds_1 = self._bounds_estimation(self._fitnesses_1_pool)
                        self._emp_bounds_2 = self._bounds_estimation(self._fitnesses_2_pool)
                
                else:
                    # if not bound estimation, compute convergence
                    avg_conv = self._estimate_convergence()
                    
                    # and spread
                    f1 = fronts[0] # last non-dominated front
                    f1_length = len(f1)
                    f1_models = self._population[:f1_length]
                    f1_fitnesses_1 = self._fitnesses_1[:f1_length]
                    f1_fitnesses_2 = self._fitnesses_2[:f1_length]
                    normalised_1 = normalise_objective(f1_fitnesses_1, self._emp_bounds_1) 
                    normalised_2 = normalise_objective(f1_fitnesses_2, self._emp_bounds_2)
                    
                    self._estimate_spread(normalised_1, normalised_2) # Deb's ∆

                    # ----------------
                    # validation
                    # ----------------
                    fit_fn_1 = self._fit_fn_1(val_loader, self._problem)
                    fit_fn_2 = self._fit_fn_2(val_loader)

                    val_fitnesses = group_fitness( # within emp_bounds
                        self._population, fit_fn_1, self._emp_bounds_1
                    )
                        
                    avg_val = sum(val_fitnesses) / len(val_fitnesses)

                    self._val_fitnesses.append(avg_val)

                    # ---------------------------------------------
                    # extract best model (best val in first front)
                    # ---------------------------------------------
                    f1_val_fitnesses = val_fitnesses[:f1_length]
                    f1_modval = list(zip(f1_models, f1_val_fitnesses, f1_fitnesses_1))
                    sorted_f1_modval = sorted(f1_modval, key=lambda x: x[1], reverse=True)
                    best_model = sorted_f1_modval[0] # tuple (model, val)
                    if self._best_model is None:
                        self._best_model = best_model #(model, val, fit)
                    else:
                        if self._best_model[1] < best_model[1]:
                            self._best_model = best_model

                    # if avg_val is not None:
                    print(f"gen:{gen}|cross-species: {interspecies}|#topo:{len(self._islands)}|avg_val: {round(avg_val, 3)}|avg_conv: {round(avg_conv, 3)}| ∆: {round(self._deltas[-1], 3)}")
                    # else:
                    #     print(f"gen:{gen}|cross-species: {interspecies}|#topo:{len(self._islands)}|avg_val: {"non_comp"}|avg_conv: {round(avg_conv, 3)}|∆: {round(self._deltas[-1], 3)}")
                

                if gen >= 6 and gen % 3 == 0:
                    m_r = self._update_m_rate(m_r)
        #########################################
        ####### IF NOT PRESTEP: ################
        #########################################
        # get the optimal front of the last generation
        # ⛔️ in a normalised space !!! FOR PLOTTING 🔥
        # ⛔️ ASSUMPTION: updating self._population, self._fitnesses_1/2
        # during selection is done adding fronts in order!!!
        if not prestep and not bound_estimation:
            self._best_front = (normalised_1, normalised_2)
    
    
    def test(self, fraction, ensemble=False):
        """
        Evaluate the population on test data.

        Args:
            fraction [float]: fraction of test data to use
            ensemble [bool]: if true, return majority vote of population as ensemble
        
        Returns:
            avg_test_fitness_1 [float]: avg. objective 1 in pop
            avg_test_fitness_2 [float]: avg. objective 2 in pop
            voting_acc [float or None]: ensemble accuracy if ensemble=True
        """
        voting_acc = None

        full_idxs = list(range(len(self._test_data)))
        if isinstance(self._test_data.targets, torch.Tensor):
            labels = self._test_data.targets.numpy()
        elif isinstance(self._test_data.targets, list):
            labels = np.array(self._test_data.targets)  #need np.arrays for stratify
            
        test_indices, _ = train_test_split(full_idxs, train_size=fraction, stratify=labels)
        test_subset = Subset(self._test_data, indices=test_indices)
        test_loader = DataLoader(test_subset, batch_size=30)
        
        if ensemble:
            
            preds = []
            truth = []
            
            device = next(self._population[0].parameters()).device
            for X, y in test_loader:
                X = X.to(device)
                y = y.to(device)

                batch = []

                for m in self._population:
                    m.eval()
                    with torch.no_grad():
                        logits = m(X)
                        pred = torch.argmax(logits, dim=1) # shape (batch, classes) ~ (30, 10)
                        batch.append(pred.unsqueeze(0))
                votes = torch.cat(batch, dim=0)
                final = torch.mode(votes, dim=0).values
                preds.append(final)
                truth.append(y)
            
            preds, truth = torch.cat(preds), torch.cat(truth)
            voting_acc = (preds == truth).float().mean().item()


        fit_fn_1 = self._fit_fn_1(test_loader, self._problem)
        fit_fn_2 = self._fit_fn_2(test_loader)

        test_fitnesses_1 = group_fitness( # clamped within emp_bounds
            self._population, fit_fn_1, self._emp_bounds_1
        )
        test_fitnesses_2 = group_fitness( # clamped within emp_bounds
            self._population, fit_fn_2, self._emp_bounds_2
        )

        avg_test_fitness_1 = sum(test_fitnesses_1) / len(test_fitnesses_1)
        avg_test_fitness_2 = sum(test_fitnesses_2) / len(test_fitnesses_2)

        return avg_test_fitness_1, avg_test_fitness_2, voting_acc