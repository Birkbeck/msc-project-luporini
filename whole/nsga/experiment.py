import random
import json
import os
from copy import deepcopy
import subprocess
import numpy as np
import torch

from .nsga import NSGA2
from .models import create_AE_pop

from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, KMNIST

class ExperimentV3():
    """
    NOT USED.

    for run:
        seed
        autopop
        bound_estimation_runs
        evolution
    """
    def __init__(
            self,
            model1, # classifier/detector
            model2, # AE
            pop,
            dataset,
            problem,
            interval,
            seed,
            experiment_path,

            prestep=False,
            AEepochs=4,
            classes=10,
            bound_runs=2,
            bound_gens=2,
            runs=2,
            gens=2,
            intersp_cross_rate=0.01,
            mutation_strength=0.3,
            mutation_rate=0.1,
            mutation_mode="light",
            
            checkpoint=True,
            device=None,
            resume=False,
            git=False
    ):
        self._model1 = model1
        self._model2 = model2
        self._pop = pop
        # self._autopop = None # keep it for checkpoints!
        self._AEepochs = AEepochs
        self._dataset = dataset.lower()
        self._classes = classes
        self._interval = interval
        self._problem = problem
        self._prestep = prestep

        self._device = device
        self._git = git
        self._check = checkpoint

        self._inter_r = intersp_cross_rate
        self._mutation_s = mutation_strength
        self._mutation_r = mutation_rate
        self._m_mode = mutation_mode
        
        self._bound_runs = bound_runs
        self._bound_gens = bound_gens
        self._runs = runs
        self._gens = gens
        self._exp_condition = "AE" if prestep else "noAE"

        self._resume = resume
        self._bounds1 = []
        self._bounds2 = []

        self._seed = seed
        self._current_seed = None
        self._experiment_path = experiment_path

        self._run = 0

        self._best = None
        self._results = [{"dataset": self._dataset, # list of dictionaries 
                          "pop_size": self._pop, # first one, basic info
                          "bound_runs": self._bound_runs,
                          "bound_gens": self._bound_gens,
                          "runs": self._runs,
                          "gens": self._gens,
                          "exp_condition": self._exp_condition,
                          "interval": self._interval,
                          "seed": self._seed}] # then, one per gen

    def _setup(self):
        if self._dataset == "mnist":
            self._train = MNIST("./whole/datasets", download=True, train=True, transform=transforms.ToTensor())
            self._test = MNIST("./whole/datasets", download=True, train=False, transform=transforms.ToTensor())
            self._input_shape = (1, 28, 28)
        elif self._dataset == "fashion":
            self._train = FashionMNIST("./whole/datasets", download=True, train=True, transform=transforms.ToTensor())
            self._test = FashionMNIST("./whole/datasets", download=True, train=False, transform=transforms.ToTensor())
            self._input_shape = (1, 28, 28)
        else:
            self._train = CIFAR10("./whole/datasets", download=True, train=True, transform=transforms.ToTensor())
            self._test = CIFAR10("./whole/datasets", download=True, train=False, transform=transforms.ToTensor())
            self._input_shape = (3, 32, 32)
        
        self._train_loader = DataLoader(self._train, batch_size=30)
        self._test_loader = DataLoader(self._test, batch_size=30)
    

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def _checkpoint(self, filepath):
        with open(filepath, "w") as f:
            json.dump({
                "results": self._results,
                "bounds1": self._bounds1,
                "bounds2": self._bounds2,
                "run": self._run,
                "current_seed": self._current_seed
            }, f)
    
    def _load_checkpoint(self, checkpath):
        """checkpoint = file.json"""
        with open(checkpath, "r") as f:
            data = json.load(f)

        self._results = data["results"]
        self._bounds1 = data["bounds1"]
        self._bounds2 = data["bounds2"]
        self._run = data["run"]
        self._current_seed = data["current_seed"]
        self._runs = data["results"][0]["runs"]
        self._gens = data["results"][0]["gens"]
        self._seed = data["results"][0]["seed"]

    
    def _save_results(self, path):
        with open(path, "w") as f:
            json.dump(self._results, f)


    def _save_best(self, path):
        model = self._best[0] #(model, val, fit)
        stride = model.get_stride()
        val = self._best[1]
        fit = self._best[2]
        torch.save({
            "weights": model.state_dict(),
            "stride": stride,
            "val": val,
            "fit": fit
        }, path)

    def _load_best(self, path):
        if os.path.exists(path):
        # if Path(path).exists():
            data = torch.load(path)
            s = data["stride"]
            w = data["weights"]
            val = data["val"]
            fit = data["fit"]
            new = self._model1(stride=s)
            new.load_state_dict(w)
            self._best = (new, val, fit)
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
    def run(self):
        # ⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️
        # create appropriate directory if directory does not exists
        # ⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️
        self._experiment_path.mkdir(parents=True, exist_ok=True)
        
        self._setup()

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
            
            # print(self._seed)
            # sys.exit()

        else:
            #############################################
            # –––––– if prestep, autoencoder !!! ––––––
            #############################################
            print("\nstarting experimental runs..")
            seed = self._current_seed if self._resume else self._seed
            for run in range(self._run, self._evo_runs):
                print(f"\ninitiating round {run}")
                self._set_seed(seed)
                
                if self._prestep:
                    print("\n* creating autoencoders..")

                    autopop = create_AE_pop( # instead of storing in self._autopop
                        self._model2,
                        self._pop,
                        self._input_shape,
                        self._AEepochs,
                        self._interval,
                        self._train_loader,
                        noise=0.4,
                        device=self._device
                    )

                    print("  - autoencoder population was created..")

                #############################################
                #############################################
                # ––– YOLO/classifier experiment (–>DV) –––
                #############################################
                #############################################
                b1, b2 = [], []
                print("\n* estimating fitness bounds..")
                boundseed = self._current_seed + 100 if self._resume else self._seed + 100
                for boundrun in range(self._bound_runs):
                    print(f"  - bounds estimation round {boundrun}")
                    self._set_seed(boundseed)

                    evolver = NSGA2(
                        pop_size=self._pop,
                        model=self._model1,
                        input_shape=self._input_shape,
                        interval=self._interval,
                        data=self._train,
                        problem=self._problem,
                        device=self._device
                    )

                    if self._prestep:
                        evolver.transfer_popV2(autopop, self._model1, self._input_shape, self._classes) # ⛔️

                    evolver.evolve(
                        generations=self._bound_gens,
                        bound_estimation=True,
                        prestep=False,
                        inter_r=self._inter_r,
                        m_r=self._mutation_r,
                        m_s=self._mutation_s,
                        m_mode=self._m_mode
                    )

                    bound1 = evolver.get_bounds()[0] # (min, max) from one evo
                    bound2 = evolver.get_bounds()[1] # # (min, max) from one evo
                    b1.append(bound1) # [(min, max), (min, max), (min, max), ..]
                    b2.append(bound2)
                    
                    boundseed += 1
                    
                # ⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️
                # and now you need a way to reduce those lists of tuples
                # ⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️
                minmax1 = list(zip(*b1)) # [(min, min, ..), (max, max, ..)]
                minmax2 = list(zip(*b2)) # [(min, min, ..), (max, max, ..)]
                mino1, maxo1 = np.percentile(minmax1[0], 5), np.percentile(minmax1[1], 95)
                mino2, maxo2 = np.percentile(minmax2[0], 5), np.percentile(minmax2[1], 95)
                self._bounds1, self._bounds2 = (mino1, maxo1), (mino2, maxo2)
                print("  - bounds have been estimated..")
            ########################################
            # –––– starting experimental runs –––––
            ########################################

            evolver = NSGA2(
                pop_size=self._pop,
                model=self._model1,
                input_shape=self._input_shape,
                interval=self._interval,
                data=self._train,
                problem=self._problem,
                device=self._device
            )
            
            if self._prestep:
                evolver.transfer_popV2(autopop, self._model1, self._input_shape, self._classes)

            evolver.set_bounds(b1=self._bounds1, b2=self._bounds2)

            # actual evolution !!!
            print(f"\n* model evolution..")
            evolver.evolve(
                generations=self._evo_gens,
                bound_estimation=False,
                prestep=False,
                inter_r=self._inter_r,
                m_r=self._mutation_r,
                m_s=self._mutation_s,
                m_mode=self._m_mode
            )

            print(f"  - evolution finished.. {run} run finished..")

            # update best if current best better than stored
            besto = evolver.get_best() #(model, val, fit)
            if self._best is None or besto[1] < self._best[1]:
                self._best = besto

            ##################################
            # extract results: conv + spread
            #################################
            best_front = evolver.get_best_front() # (fits1, fits2) (plot)
            # convergence = evolver.avg_convergence() # avg per gen (plot)
            convergence = evolver.get_convergence() # avg per gen (plot)
            f_convergence = convergence[-1] # final gen (dv)
            val_fitnesses = evolver.get_val_fitness()
            deltas = evolver.get_deltas()
            f_delta = deltas[-1]


            result = { # save LAST POP IN NORMALISED? FITNESS SPACE ⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️
                # (fitnesses1, fitnesses2, strides~colour)
                "empirical_bounds": [self._bounds1, self._bounds2], # also plot????
                "conv": convergence,
                "conv_final": f_convergence,
                "deltas": deltas, 
                "delta_final": f_delta,
                "val_fitnesses": val_fitnesses, # list of avg. pop val_fitness per gen
                "front": best_front, # (fits1, fits2)
            }

            self._results.append(result)

            
            self._run +=1
            self._seed += 1
            seed += 1
            ##################
            # checkpoint !!!!
            ###################
            if self._check:
                self._current_seed = seed
                checkpath = self._experiment_path/f"checkpoint_{run+1}.json"
                self._checkpoint(checkpath) #⛔️
                print(f"\n  - hit checkpoint! next run coming..")

            ##################
            # git control ⛔️⛔️⛔️ careful, if NO CHECK–> NO CHANGES, nothing to commit!!!!
            ###################
            if self._git:
                print("\ngit!!!")
                actions = [
                    ["git", "add", "."],
                    ["git", "commit", "-m", f"finished run {run+1} for {self._dataset}_{self._exp_condition}"],
                    ["git", "push"] # yep, you need it for colab
                ]
                for a in actions:
                    try:
                        subprocess.run(a, check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"git action failed: {a}. Exception: {e}")
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
        

import sys


class ExperimentV4():
    """
    Explerimental workflow based on multi-objective NSGA,
    supporting heterogeneous populations of convolutional models.

    Experimental condition is defined based on prestep (if True, AE initialisation).
    
    High-level experimental structure:
        set_seed
        bound_estimation runs
        experimental runs
            for run:
                set_seed(seed)
                *AUTOPOP if prestep
                *EVOLUTION LOOP
                seed+=1
                save results
                save balanced model

    
    """
    def __init__(
            self,
            model1, # classifier/detector
            model2, # AE
            pop,
            dataset,
            subset_fraction,
            problem,
            interval,
            seed,
            experiment_path,
            prestep=False,
            AEepochs=4,
            
            bound_runs=2,
            bound_gens=2,
            evo_runs=2,
            evo_gens=2,

            intersp_cross_rate=0.01,
            mutation_rate_min=0.01,
            mutation_rate_max=0.2,
            mutation_rate_decay=True,
            mutation_strength=0.2,
            mutation_mode="light",
            ensemble=False,
            
            resume=False,
            device=None,
            git=False,
            checkpoint=True
    ):
        self._model1 = model1
        self._model2 = model2
        self._pop = pop
        # self._autopop = None # keep it for checkpoints!
        self._AEepochs = AEepochs
        self._dataset = dataset.lower()
        self._subset_fraction = subset_fraction
        self._interval = interval
        self._problem = problem
        self._prestep = prestep

        self._device = device
        self._git = git
        self._check = checkpoint

        self._inter_r = intersp_cross_rate
        self._m_r_min = mutation_rate_min
        self._m_r_max = mutation_rate_max
        self._decay = mutation_rate_decay
        self._mutation_s = mutation_strength
        self._m_mode = mutation_mode
        self._ensemble = ensemble
        
        self._bound_runs = bound_runs
        self._bound_gens = bound_gens
        self._evo_runs = evo_runs
        self._evo_gens = evo_gens
        self._exp_condition = "AE" if prestep else "noAE"

        self._resume = resume
        self._bounds1 = []
        self._bounds2 = []

        self._seed = seed
        self._current_seed = None
        self._experiment_path = experiment_path

        self._run = 0 # current run?
        self._runs = evo_runs

        self._best = None
        self._results = [{"dataset": self._dataset, # list of dictionaries 
                          "pop_size": self._pop, # first one, basic info
                          "bound_runs": self._bound_runs,
                          "bound_gens": self._bound_gens,
                          "evo_runs": self._evo_runs,
                          "evo_gens": self._evo_gens,
                          "prestep": self._prestep,
                          "interval": self._interval,
                          "seed": self._seed}] # then, one per gen

    def _setup(self):
        datasets = {"mnist": [MNIST, (1, 28, 28)],
                    "kmnist": [KMNIST, (1, 28, 28)],
                    "fashion": [FashionMNIST, (1, 28, 28)],
                    "cifar": [CIFAR10, (3, 32, 32)]}
        
        dataset = datasets[self._dataset][0]
        shape = datasets[self._dataset][1]
        self._test = dataset("./whole/datasets", download=True, train=False, transform=transforms.ToTensor())
        self._train = dataset("./whole/datasets", download=True, train=True, transform=transforms.ToTensor())
        self._input_shape = shape
        
        
        self._train_loader = DataLoader(self._train, batch_size=30)
        # self._test_loader = DataLoader(self._test, batch_size=30)
    

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    
    def _checkpoint(self, filepath):
        """ Save experiment state to .json file """
        with open(filepath, "w") as f:
            json.dump({
                "results": self._results,
                "bounds1": self._bounds1,
                "bounds2": self._bounds2,
                "prestep": self._prestep,
                "run": self._run,
                "current_seed": self._current_seed
            }, f)
    
    def _load_checkpoint(self, checkpath):
        """Resume experiment from checkpoint.json """
        with open(checkpath, "r") as f:
            data = json.load(f)

        self._results = data["results"]
        self._bounds1 = data["bounds1"]
        self._bounds2 = data["bounds2"]
        self._current_seed = data["current_seed"]
        self._prestep = data["prestep"]
        self._run = data["run"]
        self._runs = data["results"][0]["evo_runs"]
        self._gens = data["results"][0]["evo_gens"]
        # self._seed = data["results"][0]["seed"] # don't need it on resume

    
    def _save_results(self, path):
        with open(path, "w") as f:
            json.dump(self._results, f)


    def _save_best(self, path):
        """ Save 'best' model. (here, defined by convergence to utopia)"""
        model = self._best[0] #(model, val, fit)
        stride = model.get_stride()
        val = self._best[1]
        fit = self._best[2]
        torch.save({
            "weights": model.state_dict(),
            "stride": stride,
            "val": val,
            "fit": fit
        }, path)

    def _load_best(self, path):
        if os.path.exists(path):
        # if Path(path).exists():
            data = torch.load(path)
            s = data["stride"]
            w = data["weights"]
            val = data["val"]
            fit = data["fit"]
            new = self._model1(stride=s)
            new.load_state_dict(w)
            self._best = (new, val, fit)
        else:
            self._best = None

    def get_empirical_bounds(self):
        return self._bounds1, self._bounds2
    
    def get_results(self):
        return self._results
    

    ###########################################
    # -----------------------------------------
    # -------- EXPERIMENTAL WORKFLOW ---------
    # -----------------------------------------
    ###########################################
    def run(self):
        # ⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️
        # create appropriate directory if directory does not exists
        # ⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️
        self._experiment_path.mkdir(parents=True, exist_ok=True)
        
        self._setup()

        # ----------------------------------------------
        # if resume, load checkpoint and related data
        # ---------------------------------------------
        if self._resume and self._experiment_path is not None:
            checkpoints = sorted(self._experiment_path.glob(f"checkpoint_*.json"))
            if checkpoints:
                last_checkpoint = checkpoints[-1]
                self._load_checkpoint(last_checkpoint)
            
            bestpath = self._experiment_path / "best.pth"
            if bestpath.exists() and bestpath.is_file():
                self._load_best(bestpath)
            
            print(f"\nresuming experiment: {self._run} runs gone..")

        else:
            # ----------------------------------------------------------
            # otherwise, if not checkpoint, jump to bound estimation
            # ----------------------------------------------------------
            b1, b2 = [], [] # for each obj [(min, max), (min, max), ..] per boundrun
            print("\n* estimating fitness bounds..")
            boundseed = self._seed + 100

            # start estimation runs
            for boundrun in range(self._bound_runs):
                print(f"  - bounds estimation round {boundrun}")
                self._set_seed(boundseed)

                # if prestep, then bound estimation with autoencoder weights
                if self._prestep:
                    print("  - creating autoencoders..")

                    autopop = create_AE_pop( # instead of storing in self._autopop
                        self._model2,
                        self._pop,
                        self._input_shape,
                        self._AEepochs,
                        self._interval,
                        self._train_loader,
                        noise=0.4,
                        device=self._device
                    )

                    print("  - autoencoder population has been created..")

                # initialise NSGA
                evolver = NSGA2(
                    pop_size=self._pop,
                    model=self._model1,
                    input_shape=self._input_shape,
                    interval=self._interval,
                    train_data=self._train,
                    test_data=self._test,
                    problem=self._problem,
                    device=self._device
                )

                # if prestep, transfer weights from AE population
                if self._prestep:
                    evolver.transfer_popV2(autopop, self._model1, self._input_shape, self._classes) # ⛔️

                # evolve AE-initialised models
                evolver.evolve(
                    generations=self._bound_gens,
                    bound_estimation=True,
                    prestep=False,
                    subset_fraction=self._subset_fraction,
                    inter_r=self._inter_r,
                    m_r_min=self._m_r_min,
                    m_r_max=self._m_r_max,
                    m_r_decay = self._decay,
                    m_s=self._mutation_s,
                    m_mode=self._m_mode
                )

                # collect fitnesses
                bound1 = evolver.get_bounds()[0] # (min, max) from one evo
                bound2 = evolver.get_bounds()[1] # # (min, max) from one evo
                b1.append(bound1) # [(min, max), (min, max), (min, max), ..]
                b2.append(bound2)
                    
                boundseed += 1
            
                if self._prestep:
                    del autopop # make sure that GPU is freed
                    torch.cuda.empty_cache() # ⁉️
            

            # computing emptirical bounds from collected minima and maxima
            minmax1 = list(zip(*b1)) # [(min, min, ..), (max, max, ..)]
            minmax2 = list(zip(*b2)) # [(min, min, ..), (max, max, ..)]
            # mino1, maxo1 = np.percentile(minmax1[0], 5), np.percentile(minmax1[1], 95)
            # mino2, maxo2 = np.percentile(minmax2[0], 5), np.percentile(minmax2[1], 95)
            mino1, maxo1 = np.mean(minmax1[0]), np.mean(minmax1[1])
            mino2, maxo2 = np.mean(minmax2[0]), np.mean(minmax2[1])
            self._bounds1, self._bounds2 = (mino1, maxo1), (mino2, maxo2)
            print("  - bounds have been estimated..")
        
        # ---------------------------------------------
        # now, move on to ACTUAL EVOLUTION
        #
        # set_seed
        # for run in runs:
        #   if prestep:
        #       train AE population
        #       transfer weights
        #   evolution
        #   update_seed
        # -----------------------------------------------
        print("\n* starting experimental runs..")
        seed = self._current_seed if self._resume else self._seed
        
        for run in range(self._run, self._runs):
            print(f"\n* initiating round {run}")
            self._set_seed(seed)
            
            if self._prestep:
                print("  - creating autoencoders..")

                autopop = create_AE_pop( # instead of storing in self._autopop
                    self._model2,
                    self._pop,
                    self._input_shape,
                    self._AEepochs,
                    self._interval,
                    self._train_loader,
                    noise=0.4,
                    device=self._device
                )

                print("  - autoencoder population has been created..")
            
            ########################################
            # ––---–– starting evolution –––-––
            ########################################

            evolver = NSGA2(
                pop_size=self._pop,
                model=self._model1,
                input_shape=self._input_shape,
                interval=self._interval,
                train_data=self._train,
                test_data=self._test,
                problem=self._problem,
                device=self._device
            )
            
            if self._prestep:
                evolver.transfer_popV2(autopop, self._model1, self._input_shape, self._classes)

            evolver.set_bounds(b1=self._bounds1, b2=self._bounds2)

            # actual evolution !!!
            print(f"  - model evolution..")
            evolver.evolve(
                generations=self._evo_gens,
                bound_estimation=False,
                prestep=False,
                subset_fraction=self._subset_fraction,
                inter_r=self._inter_r,
                m_r_min=self._m_r_min,
                m_r_max=self._m_r_max,
                m_r_decay = self._decay,
                m_s=self._mutation_s,
                m_mode=self._m_mode
            )

            print(f"  - evolution finished.. {run} run finished..")

            # update best if current best better than stored
            besto = evolver.get_best() #(model, val, fit)
            m, v, f = deepcopy(besto[0]), besto[1], besto[2]
            if self._best is None or besto[1] < self._best[1]:
                self._best = (m, v, f)

            ##################################
            # extract results: conv + spread + test_acc
            #################################
            avg_test_fit_1, avg_test_fit_2, avg_ensemble_fit = evolver.test(self._subset_fraction, ensemble=self._ensemble)
            if self._ensemble:
                print(f"avg test accuracy: {avg_test_fit_1} | ensemble test accuracy: {avg_ensemble_fit}")
            else:
                print(f"avg test accuracy: {avg_test_fit_1}")
            
            val_fitnesses = evolver.get_val_fitness()
            best_front = evolver.get_best_front() # (fits1, fits2) (plot)
            # convergence = evolver.avg_convergence() # avg per gen (plot)
            convergence = evolver.get_convergence() # avg per gen (plot)
            f_convergence = convergence[-1] # final gen (dv)
            deltas = evolver.get_deltas()
            f_delta = deltas[-1]


            result = { # save LAST POP IN NORMALISED? FITNESS SPACE ⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️
                # (fitnesses1, fitnesses2, strides~colour)
                "test_fit_1": avg_test_fit_1,
                "test_fit_2": avg_test_fit_2,
                "val_fits": val_fitnesses, # list of avg. pop val_fitness per gen
                "ensemble_fit"
                "empirical_bounds": [self._bounds1, self._bounds2], # also plot????
                "conv": convergence,
                "conv_final": f_convergence,
                "deltas": deltas, 
                "delta_final": f_delta,
                "front": best_front, # (fits1, fits2)
            }

            self._results.append(result)

            
            self._run +=1
            self._seed += 1
            seed += 1

            if self._prestep:
                del autopop # make sure that GPU is freed
                torch.cuda.empty_cache()
            
            # ---------------------
            # checkpoint !!!!
            # ---------------------
            if self._check:
                self._current_seed = seed
                checkpath = self._experiment_path/f"checkpoint_{run+1}.json"
                self._checkpoint(checkpath) #⛔️
                print(f"\n  - hit checkpoint! next run coming..")

            # ----------------------------
            # git control (still to fix)
            # ---------------------------
            if self._git:
                print("\ngit!!!")
                actions = [
                    ["git", "add", "."],
                    ["git", "commit", "-m", f"finished run {run+1} for {self._dataset}_{self._exp_condition}"],
                    ["git", "push"] # yep, you need it for colab
                ]
                for a in actions:
                    try:
                        subprocess.run(a, check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"git action failed: {a}. Exception: {e}")
        
        
        # --------- save results ––––––––––
        resultpath = self._experiment_path/"results.json"
        if resultpath is not None:
            self._save_results(resultpath)

        # ---–---- save best model ––––––––
        bestpath = self._experiment_path/"best.pth"
        if bestpath is not None:
            self._save_best(bestpath)
        
