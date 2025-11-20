import random
import json
import numpy as np
import torch
import all

from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10

class Experiment():
    def __init__(
            self,
            model,
            pop,
            dataset,
            problem,
            bound_gens,
            evo_gens,
            interval,
            seed,
            save_to,
            resume
    ):
        self._model = model
        self._pop = pop
        self._dataset = dataset.lower()
        self._problem = problem
        self._bound_gens = bound_gens
        self._evo_gens = evo_gens
        self._interval = interval
        self._seed = seed
        self._resume = resume
        self._run = 0
        self._max_runs = None

        self._fronts = []
        self._avg_convs = []
        self._convs_in_time = []

    def _setup(self):
        if self._dataset == "mnist":
            self._train = MNIST("./datasets", download=True, train=True, transform=transforms.ToTensor())
            self._input_shape = (1, 28, 28)
        elif self._dataset == "fashion":
            self._train = FashionMNIST("./datasets", download=True, train=True, transform=transforms.ToTensor())
            self._input_shape = (1, 28, 28)
        else:
            self._train = CIFAR10("./datasets", download=True, train=True, transform=transforms.ToTensor())
            self._input_shape = (3, 32, 32)
        self._train_loader = DataLoader(self._train, batch_size=30)

    def _set_seed(self):
        random.seed(self._seed)
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)
    
    def _checkpoint(self, filepath):
        with open(f"{dataset}_{probl}_{pop}_{exps}.json", "w") as f:
            json.dump({
                "fronts": self._fronts,
                "avg_convs": self._avg_convs,
                "convs_in_time": self._convs_in_time
            }, f)
    
    def _load_checkpoint(self, filepath)
        

    def go(self, runs, resume=False):

        if resume:
            self._load_checkpoint()

        if self._max_runs is None:
            self._max_runs = runs

        for e in range(self._run, self._max_runs):
            print(f"\nBeginning experiment {e}")
            self._set_seed()
            self._setup()

            evolver = all.NSGA2(
                pop_size=self._pop,
                model=self._model,
                input_shape=self._input_shape,
                interval=self._interval,
                data=self._train,
                problem=self._problem
            )

            print("\n- estimating the bounds..")
            evolver.evolve(
                generations=self._bound_gens,
                bound_estimation=True,
                checkpoint=False
            )

            b1, b2 = evolver.get_bounds()
            evolver.reset(
                self._model,
                self._pop,
                interval=self._interval,
                bound1=b1, bound2=b2
            )

            # actual evolution
            print("- actual evolution..")
            evolver.evolve(
                generations=self._evo_gens,
                bound_estimation=False,
                checkpoint=False
            )

            # extract results 
            front = evolver.get_best_front() #⛔️
            conv = evolver.conv_in_time()
            conv_final = evolver.avg_convergence()

            fronts.append(front)
            convs_in_time.append(conv)
            avg_convs.append(conv_final)
            print(f"Avg population convergence: {round(conv_final, 2)}")

            self._checkpoint()

            self._run +=1
            
            # update seed for next run
            seed += 2


        
    


################################################
######### set the experiments #################
################################################
m = TinyConvClassifier
pop = 15
dataset = "mnist"
probl = "classification"
exps = 1 # number of experiments
bound_g = 2 # bound exploration gens
evo_g = 6 # actual evolution gens
inter = [2, 6]
seed = 42

fronts = []
avg_convs = []
convs_in_time = []
for e in range(exps):
    print(f"\nBeginning experiment {e}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    evolver = all.NSGA2(
        pop_size=pop,
        model=m,
        input_shape=mnist,
        interval=inter,
        data=train_data,
        problem=probl
    )

        # exploratory runs for empirical min/max
    print("\n- estimating the bounds..")
    evolver.evolve(
        generations=bound_g,
        bound_estimation=True,
        checkpoint=False
    )

    b1, b2 = evolver.get_bounds()
    evolver.reset(
        m, pop, interval=inter, bound1=b1, bound2=b2
    )

    # actual evolution
    print("- actual evolution..")
    evolver.evolve(
        generations=evo_g,
        bound_estimation=False,
        checkpoint=False
    )

    # extract exp. results 🔥
    front = evolver.get_best_front() #⛔️
    conv = evolver.conv_in_time()
    conv_final = evolver.avg_convergence()

    fronts.append(front)
    convs_in_time.append(conv)
    avg_convs.append(conv_final)
    print(f"Avg population convergence: {round(conv_final, 2)}")
    
    seed += 2


################################################
######## save convergences ####################
################################################
with open(f"{dataset}_{probl}_{pop}_{exps}.json", "w") as f:
    json.dump({
        "fronts": fronts,
        "avg_convs": avg_convs,
        "convs_in_time": convs_in_time
    }, f)