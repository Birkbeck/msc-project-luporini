import random
import json
import os
from pathlib import Path
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
            evo_gens,
            bound_gens,
            interval,
            seed,
            best_path,
            check_path,
            resume=False,
            prestep=False,
            prestep_gens=0
    ):
        self._model = model
        self._pop = pop
        self._dataset = dataset.lower()
        self._interval = interval
        self._problem = problem
        self._prestep = prestep

        self._prestep_gens = prestep_gens
        self._evo_gens = evo_gens
        self._bound_gens = bound_gens

        self._seed = seed
        self._resume = resume
        self._best_filepath = best_path
        self._check_filepath = check_path

        

        self._bound_estimation = False if self._resume else True
        self._bounds1 = None
        self._bounds2 = None

        self._run = 0
        self._max_runs = None

        self._best = None
        self._fronts = []
        self._avg_convs = []
        self._convs_in_time = []
    
    def get_fronts(self):
        return self._fronts
    def get_avg_convs(self):
        return self._avg_convs
    def get_convs_in_time(self):
        return self._convs_in_time
    def get_empirical_bounds(self):
        return self._bounds1, self._bounds2

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
        with open(filepath, "w") as f:
            json.dump({
                "fronts": self._fronts,
                "avg_convs": self._avg_convs,
                "convs_in_time": self._convs_in_time,
                "bounds1": self._bounds1,
                "bounds2": self._bounds2,
                "run": self._run,
                "max_runs": self._max_runs,
            }, f)
    
    def _load_checkpoint(self, checkpoint):
        """checkpoint = file.json"""
        with open(checkpoint, "r") as f:
            data = json.load(f)

        self._fronts = data["fronts"]
        self._avg_convs = data["avg_convs"]
        self._convs_in_time = data["convs_in_time"]
        self._bounds1 = data["bounds1"]
        self._bounds2 = data["bounds2"]
        self._run = data["run"]
        self._max_runs = data["max_runs"]
    
    def _save_best(self, path):
        model, conv = self._best
        stride = model.get_stride()
        torch.save({
            "weights": model.state_dict(),
            "convergence": conv,
            "stride": stride
        }, path)

    def _load_best(self, path):
        if os.path.exists(path):
            data = torch.load(path)
            s = data["stride"]
            c = data["convergence"]
            w = data["weights"]
            new = self._model(stride=s)
            new.load_state_dict(w)
            self._best = (new, c)
        else:
            self._best = None


    ###########################################
    ### actual experiment workflow ##########
    ###########################################
    def run(self, bound_estimation_runs, runs):
        ##############################################
        ###### if resume, load checkpoint ########
        ###############################################
        if self._resume and self._check_filepath is not None:
            self._load_checkpoint(self._check_filepath)
            if self._best_filepath is not None:
                self._load_best(self._best_filepath)
        else:
            self._max_runs = runs
        ##############################################
        #############################################
        ###### if prestep, autoencoder !!! ########
        #############################################
        ##############################################
        if self._prestep:
            evolver = all.NSGA2(
                pop_size=self._pop,
                model=self._model,
                input_shape=self._input_shape,
                interval=self._interval,
                data=self._train,
                problem="AE"
            )

            evolver.evolve(
                generations=self._bound_gens,
                bound_estimation=False,
                prestep=True
            )

            autopop = evolver.get_transfer_pop()
        ##############################################
        #############################################
        ###### the actual classifier/YOLO exp ########
        #############################################
        ##############################################
        if self._bound_estimation:
            print("\n- estimating fitness bounds..")
            for e in range(bound_estimation_runs):
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

                if self._prestep:
                    evolver.transfer_pop(autopop) # ⛔️

                evolver.evolve(
                    generations=self._bound_gens,
                    bound_estimation=True,
                    prestep=False
                )

            bounds1 = evolver.get_bounds()[0]
            bounds2 = evolver.get_bounds()[1]
            self._bounds1 = bounds1
            self._bounds2 = bounds2
            print("- bounds have been estimated..")
        ################################
        #### starting experimental runs
        ################################
        for e in range(self._run, self._max_runs):
            suf = "st" if e==1 else "nd" if e==2 else "th" 

            # adjusting checkpoint filename according to run
            checkpath = self._check_filepath/f"checkpoint_{e}.json"

            # setting seed and preparing data
            print(f"\n- beginning {e}{suf} run")
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
            
            if self._prestep:
                evolver.transfer_pop(autopop)

            evolver.set_bounds(b1=self._bounds1, b2=self._bounds2)

            # actual evolution !!!
            print("- starting actual evolution..")
            evolver.evolve(
                generations=self._evo_gens,
                bound_estimation=False,
                prestep=False
            )

            print(f"- {e}{suf} run finished")

            ################################################
            # update best if current is 'better' than stored
            ################################################
            besto = evolver.get_best()
            if self._best is None or besto[1] < self._best[1]:
                self._best = besto

            ####################
            # extract results
            ###################
            front = evolver.get_best_front()
            conv = evolver.conv_in_time()
            conv_final = evolver.avg_convergence()

            self._fronts.append(front)
            self._convs_in_time.append(conv)
            self._avg_convs.append(conv_final)
            print(f"- last population convergence: {round(conv_final, 2)}")

            if self._check_filepath is not None:
                self._checkpoint(checkpath) #⛔️
                print(f"- hit checkpoint!")


            self._run +=1
            # update seed for next run
            self._seed += 2

        bestpath = self._best_filepath/f"best_{self._dataset}_{}"
        if self._best_filepath is not None:
            self._save_best(self._best_filepath)