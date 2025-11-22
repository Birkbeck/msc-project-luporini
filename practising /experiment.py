import random
import json
import os
import numpy as np
import torch
import all

from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10

class Experiment():
    def __init__(
            self,
            model1, # classifier/detector
            model2, # AE
            pop,
            dataset,
            problem,

            mutation_strength,
            mutation_prob,
            evo_gens,
            bound_gens,
            interval,
            seed,

            experiment_path,
            device=None,
            resume=False,
            prestep=False,
            prestep_gens=0
    ):
        self._model1 = model1
        self._model2 = model2
        self._pop = pop
        self._autopop = None
        self._dataset = dataset.lower()
        self._interval = interval
        self._problem = problem
        self._prestep = prestep
        self._device = device

        self._mutation_s = mutation_strength
        self._mutation_p = mutation_prob
        self._prestep_gens = prestep_gens
        self._evo_gens = evo_gens
        self._bound_gens = bound_gens

        self._resume = resume
        self._bound_estimation = False if self._resume else True
        self._bounds1 = None
        self._bounds2 = None

        self._seed = seed
        self._experiment_path = experiment_path

        self._run = 0
        self._max_runs = None

        self._best = None
        self._results = [] # list of dictionaries per run

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
                "results": self._results,
                "bounds1": self._bounds1,
                "bounds2": self._bounds2,
                "run": self._run,
                "seed": self._seed,
                "max_runs": self._max_runs,
            }, f)
    
    def _load_checkpoint(self, checkpath):
        """checkpoint = file.json"""
        with open(checkpath, "r") as f:
            data = json.load(f)

        self._results = data["results"]
        self._bounds1 = data["bounds1"]
        self._bounds2 = data["bounds2"]
        self._run = data["run"]
        self._seed = data["seed"]
        self._max_runs = data["max_runs"]

    def _save_autopop(self, path): # for autopop!!!! ⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️
        d = dict()
        for i in range(len(self._autopop)):
            d[i] = {}
            d[i]["weights"] = self._autopop[i].state_dict()
            d[i]["stride"] = self._autopop[i].get_stride()
        torch.save(d, path)
    
    def _load_autopop(self, path, model):
        data = torch.load(path)
        pop = []
        for i in sorted(data):
            w = data[i]["weights"]
            s = data[i]["stride"]
            new = model(stride=s)
            new.load_state_dict(w)
            pop.append(new)
        
        self._autopop = pop
    
    def _save_results(self, path):
        with open(path, "w") as f:
            json.dump(self._results, f)


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
        # if Path(path).exists():
            data = torch.load(path)
            s = data["stride"]
            c = data["convergence"]
            w = data["weights"]
            new = self._model1(stride=s)
            new.load_state_dict(w)
            self._best = (new, c)
        else:
            self._best = None

    def get_empirical_bounds(self):
        return self._bounds1, self._bounds2
    
    def get_results(self):
        return self._results
    

    ###########################################
    ###########################################
    # –––––––– EXPERIMENTAL WORKFLOW –––––––––
    ###########################################
    ###########################################
    def run(self, bound_estimation_runs=10, runs=30):
        # ⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️
        # create appropriate directory if directory does not exists
        # ⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️
        self._experiment_path.mkdir(parents=True, exist_ok=True)
        
        ###### if resume, load checkpoint ########
        if self._resume and self._experiment_path is not None:
            checkpoints = sorted(self._experiment_path.glob(f"checkpoint_*.json"))
            if checkpoints:
                last_checkpoint = checkpoints[-1]
                self._load_checkpoint(last_checkpoint)
            
            bestpath = self._experiment_path / "best.pth"
            if bestpath.exists() and bestpath.is_file():
                self._load_best(bestpath)
            
            autopath = self._experiment_path / "autopop.pth"
            if autopath.exists() and autopath.is_file():
                self._load_autopop(autopath)
                self._prestep = False
            
        else:
            self._max_runs = runs
        
        #############################################
        # –––––– if prestep, autoencoder !!! ––––––
        #############################################
        
        if self._prestep:
            self._setup()
            
            evolver = all.NSGA2(
                pop_size=self._pop,
                model=self._model2, # autoencoders!
                input_shape=self._input_shape,
                interval=self._interval,
                data=self._train,
                problem="AE",
                device=self._device
            )

            evolver.evolve(
                generations=self._prestep_gens,
                bound_estimation=False,
                prestep=True,
                m_r=self._mutation_s,
                m_c=self._mutation_p
            )

            print("evolved autoencoder population..")
            self._autopop = evolver.get_transfer_pop()
            self._save_autopop(self._experiment_path / "autopop.pth")

            # self._checkpoint() # ⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️
        #############################################
        #############################################
        # ––– YOLO/classifier experiment (–>DV) –––
        #############################################
        #############################################
        if self._bound_estimation:
            print("\n- estimating fitness bounds..")
            for e in range(bound_estimation_runs):
                self._set_seed()
                self._setup()

                evolver = all.NSGA2(
                    pop_size=self._pop,
                    model=self._model1,
                    input_shape=self._input_shape,
                    interval=self._interval,
                    data=self._train,
                    problem=self._problem,
                    device=self._device
                )

                if self._prestep:
                    evolver.transfer_pop(self._autopop) # ⛔️

                evolver.evolve(
                    generations=self._bound_gens,
                    bound_estimation=True,
                    prestep=False,
                    m_r=self._mutation_s,
                    m_c=self._mutation_p
                )

            bounds1 = evolver.get_bounds()[0]
            bounds2 = evolver.get_bounds()[1]
            self._bounds1 = bounds1
            self._bounds2 = bounds2
            print("- bounds have been estimated..")
        ########################################
        # –––– starting experimental runs –––––
        ########################################
        for e in range(self._run, self._max_runs):
            suf = "st" if e==1 else "nd" if e==2 else "rd" if e==3 else "th" 

            # setting seed and preparing data
            print(f"\n- beginning {e}{suf} run")
            self._set_seed()
            self._setup()

            evolver = all.NSGA2(
                pop_size=self._pop,
                model=self._model1,
                input_shape=self._input_shape,
                interval=self._interval,
                data=self._train,
                problem=self._problem,
                device=self._device
            )
            
            if self._prestep:
                evolver.transfer_pop(self._autopop)

            evolver.set_bounds(b1=self._bounds1, b2=self._bounds2)

            # actual evolution !!!
            print("- actual evolution..")
            evolver.evolve(
                generations=self._evo_gens,
                bound_estimation=False,
                prestep=False,
                m_r=self._mutation_s,
                m_c=self._mutation_p
            )

            print(f"- {e}{suf} run finished")

            # update best if current best better than stored
            besto = evolver.get_best()
            if self._best is None or besto[1] < self._best[1]:
                self._best = besto

            ##################################
            # extract results: conv + spread
            #################################
            front = evolver.get_best_front() # (fits1, fits2) (plot)
            convergence = evolver.avg_convergence() # avg per gen (plot)
            f_convergence = evolver.final_convergence() # final gen (dv)

            deltas = evolver.get_deltas()
            f_delta = evolver.final_delta()


            result = {
                "front": front,
                "conv": convergence,
                "conv_final": f_convergence,
                "deltas": deltas, 
                "delta_final": f_delta
            }

            self._results.append(result)

            ##################
            # checkpoint !!!!
            ###################
            checkpath = self._experiment_path/f"checkpoint_{e}.json"
            self._checkpoint(checkpath) #⛔️
            print(f"- hit checkpoint!")


            self._run +=1
            # update seed for next run
            self._seed += 2

        #############################################
        # –––-----– runs are over ––––––---
        #############################################
        # --------- save results ––––––––––
        resultpath = self._experiment_path/"results.json"
        if resultpath is not None:
            self._save_results(resultpath)

        # ---–---- save best model ––––––––
        bestpath = self._experiment_path/"best.pth"
        if bestpath is not None:
            self._save_best(bestpath)