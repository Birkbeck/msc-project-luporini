from copy import deepcopy
import json
import random
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torchvision.transforms import ToTensor
from architectures import create_AE_pop

def flatten(model):
    """
     flatten each parameter into a 1D tensor and concatenate
    """
    device = next(model.parameters()).device
    return torch.cat([param.view(-1) for param in model.parameters()]).to(device)
    # return torch.cat([param.view(-1) for param in model.parameters()])
    # return torch.cat([param.data.view(-1) for param in model.parameters()]) # NO!!!!
                            # BREAKS COMPUTATION GRAPH.. don't risk it
                            # might need autograd later


def remodel(flat, model):
    """
    remap a flat model into a non flat torch.architecture
    """
    index = 0
    for param in model.parameters():
        with torch.no_grad():
            n = param.numel()
            param.copy_(flat[index:index+n].view_as(param))
            index += n
    return model


def mutate(guy:torch.Tensor, m_rate=0.2, m_strength=0.3, mode="small") -> torch.Tensor:
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
        # mutation = mask * noise #?? effect too great on 1s, needs scaling
        #⛔️half of the genes mutated on avg!!! AGGRESSIVE?
    else:
        mask = torch.rand_like(guy, device=device) < m_rate
    
    noise = torch.randn_like(guy, device=device)
    mutation = m_strength * mask * noise
    
    return guy + mutation


def crossover(parent1, parent2, mode="uniform"):
    """
    uniform crossover between two flat 1D tensors by default. If type != "uniform", then one-point.
    """
    device = parent1.device
    if mode == "uniform":
        mask = torch.randint_like(parent1, 2, device=device).bool()
        child1 = torch.where(mask, parent1, parent2)
        child2 = torch.where(mask, parent2, parent1)
    else:
        rip = torch.randint(0, parent1.numel()-1, size=(1,)).item() #for fun..
        child1 = torch.cat([parent1[:rip], parent2[rip:]])
        child2 = torch.cat([parent2[:rip], parent1[rip:]])
    return child1, child2


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
        out = lambda X, y: y      # if working with encodings, will break!
    elif problem == "AE":
        loss_fn = nn.MSELoss()
        out = lambda X, y: X
    
    def fitness(model):
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            tot_loss = 0
            if problem == "classification":
                correct = 0
                tot = 0
            for X, y in data:
                X = X.to(device, non_blocking=True)
                y = out(X, y).to(device, non_blocking=True)
                
                pred = model(X)
                loss = loss_fn(pred, y)
                tot_loss += loss.item()   # ⛔️SPIKING FITNESS if avg_loss very small
                
                if problem == "classification":
                    y_pred_class = torch.argmax(pred, dim=1)
                    correct += (y_pred_class == y).sum().item()
                    tot += y.size(0)
            
            if problem == "classification":
                return correct / tot
        
            avg_loss = tot_loss / len(data) # enumerate starts from 0
            # avg_fitness = 1 / (avg_loss + 1e-8) # if avg_loss–>0, avg_fit–>inf!!
            avg_fitness = -avg_loss
        
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
    mino, maxo = bound[0], bound[1]
    deno = (maxo - mino + 1e-8)
    normalised_fitnesses = [
        (f - mino) / deno
        for f in fitnesses
    ]
    return normalised_fitnesses




class GeneticAlgorithmV2():


    def __init__(
            self,
            model, # instance!!!
            population,
            pop_size,
            data,
            problem="AE",
            device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ):
        self._model = model
        self._pop_size = pop_size # ⁉️need it⁉️
        self._fit_fn = None
        self._data = data
        self._population = [deepcopy(model.to(device)) for i in range(self._pop_size)] if population else None
        self._fitnesses = [None for i in range(self._pop_size)]
        self._avgfitness = [] # per generation
        self._problem = problem
        self._device = device
    
    def _trainval_loaders(self, fraction):
        """
        produces train and val loaders
        val_loader_size < train_loader_size
        """
        full_idxs = list(range(len(self._data)))
        if isinstance(self._data.targets, torch.Tensor):
            labels = self._data.targets.numpy()
        elif isinstance(self._data.targets, list):
            labels = np.array(self._data.targets)  #need np.arrays for stratify
            
        train_indices, remaining = train_test_split(full_idxs, train_size=fraction, stratify=labels)
        train_subset = Subset(self._data, indices=train_indices)

        remaining_labels = labels[remaining]
        val_indices, _ = train_test_split(remaining, train_size=0.5, stratify=remaining_labels)
        val_subset = Subset(self._data, indices=val_indices)


        train_loader = DataLoader(
            train_subset, batch_size=30, shuffle=True, pin_memory=True
        )
        val_loader = DataLoader(
            val_subset, batch_size=30, shuffle=True, pin_memory=True
        )
        return train_loader, val_loader
    

    def _avg_fitness(self):
        fitnesses = [i for i in self._fitnesses if i is not None]
        if not fitnesses:
            return None
        return torch.mean(torch.tensor(fitnesses)).item()
        

    def evolve(self, generations=10, subset_fraction=0.07, m_r_min=0.01, m_r_max=0.2, m_r_decay = True, power=2, m_s=0.2, mode="uniform"):
        """
        evolution method👍

        careful: do you want access to mutation parameters????

        args:
            generations: number of generations
            report_jump: integer n, with report given every n generations
        """
        #######################################
        train_loader, _ = self._trainval_loaders(subset_fraction)
            
        self._fit_fn = model_fitness(train_loader, self._problem)

        self._fitnesses = group_fitness( # clamped within emp_bounds
            self._population, self._fit_fn
        )
        
        for gen in range(generations):
            if m_r_decay:
                m_r = m_r_min + (m_r_max - m_r_min)*(1 - (gen/(generations - 1))**power)

            # mating events
            flat_children = [] # TOURNAMENT 🔥
            for _ in range(self._pop_size//2): 
                
                parents = random.sample(self._population, k=2)
                parent1, parent2 = parents[0], parents[1]
                parent1 = flatten(parent1)
                parent2 = flatten(parent2)
                        
                child1, child2 = crossover(parent1, parent2, mode=mode)
                child1 = mutate(child1, m_rate=m_r, m_strength=m_s, mode="small")
                child2 = mutate(child2, m_rate=m_r, m_strength=m_s, mode="small")
                    
                flat_children.extend([child1, child2])

            remodelled_children = [remodel(f, deepcopy(self._model)) for f in flat_children]
            
            children_fitnesses = group_fitness(
                remodelled_children, self._fit_fn
            )

            parents = list(zip(self._population, self._fitnesses))
            children = list(zip(remodelled_children, children_fitnesses))
            whole = parents + children 
            
            sorted_whole = sorted(whole, key=lambda x: x[1], reverse=True)
            self._population = [m for m, _ in sorted_whole[:self._pop_size]]
            self._fitnesses = [f for _, f in sorted_whole[:self._pop_size]]

            self._avgfitness.append(self._avg_fitness())

            if self._problem == "classification":
                print(f"{gen} gen | avg. population acc: {self._avg_fitness()}")
            else:
                print(f"{gen} gen | avg. population fitness: {self._avg_fitness()}")
            

    def extract_best(self, k=1):
        zipped = list(zip(self._population, self._fitnesses))
        sorted_population = sorted(zipped, key=lambda x: x[1], reverse=True)
        return sorted_population[:k]
    

    def get_fitintime(self):
        """returns the list of fitnesses - avg.fit per generation"""
        return self._avgfitness

    
    def transfer_popV2(self, pop, model, in_shape, classes, freeze=False):
        """should I just swap the new pop in??"""
        finalpop = []
        for m in pop:
            weights = m.encoder.state_dict()
            new = model(input_shape=in_shape, stride=m.get_stride(), classes=classes).to(self._device)
            new.encoder.load_state_dict(weights)

            if freeze:
                for param in new.parameters():
                    param.requires_grad = False

            finalpop.append(new)
        
        self._population = finalpop
        


class GAExperiment():
    def __init__(
            self,
            model1, # task model
            model2, # autoencoder
            stride,
            pop,
            dataset,
            problem,
            seed,
            experiment_path,
            prestep=False,
            AEepochs=4,
            classes=10,
            runs=1,
            gens=30,
            mutation_rate_min=0.01,
            mutation_rate_max=0.2,
            mutation_rate_decay=True,
            mutation_strength=0.2,
            mutation_mode="light",
            my_device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
            resume=False,
            checkpoint=True
    ):
        self._model1 = model1
        self._model2 = model2
        self._stride = stride
        self._pop = pop
        self._dataset = dataset
        self._problem = problem
        self._seed = seed
        self._path = experiment_path
        self._prestep = prestep
        self._AEepochs = AEepochs
        self._classes = classes
        self._runs = runs
        self._gens = gens
        self._m_r_min = mutation_rate_min
        self._m_r_max = mutation_rate_max
        self._decay = mutation_rate_decay
        self._m_strength = mutation_strength
        self._m_mode = mutation_mode
        
        self._run = 0
        self._current_seed = None

        self._device = my_device
        self._resume = resume
        self._check = checkpoint
        
        self._results = [ # list of dictionaries
            {"dataset": self._dataset, # this is the first one, basic info
             "pop_size": self._pop, 
             "evo_runs": self._runs,
             "evo_gens": self._gens,
             "exp_condition": self._prestep,
             "seed": self._seed}
        ]
    
    def _setup(self):
        if self._dataset == "mnist":
            self._test = MNIST("./datasets", download=True, train=False, transform=ToTensor())
            self._train = MNIST("./datasets", download=True, train=True, transform=ToTensor())
            self._input_shape = (1, 28, 28)
        elif self._dataset == "fashion":
            self._train = FashionMNIST("./datasets", download=True, train=True, transform=ToTensor())
            self._test = FashionMNIST("./datasets", download=True, train=False, transform=ToTensor())
            self._input_shape = (1, 28, 28)
        else:
            self._train = CIFAR10("./datasets", download=True, train=True, transform=ToTensor())
            self._test = CIFAR10("./datasets", download=True, train=False, transform=ToTensor())
            self._input_shape = (3, 32, 32)
        
        self._train_loader = DataLoader(self._train, batch_size=30)
        self._test_loader = DataLoader(self._test, batch_size=30)
    

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    def _checkpoint(self, filepath):
        with open(filepath, "w") as f:
            json.dump({
                "results": self._results,
                "run": self._run,
                "seed": self._seed,
                "runs": self._runs,
                "current_seed": self._current_seed,
                "prestep": self._prestep
            }, f)
    
    def _load_checkpoint(self, checkpath):
        """checkpoint = file.json"""
        with open(checkpath, "r") as f:
            data = json.load(f)

        self._results = data["results"]
        self._run = data["run"]
        self._seed = data["seed"]
        self._runs = data["runs"]
        self._current_seed = data["current_seed"]
        self._prestep = data["prestep"]

    
    def _save_results(self, path):
        with open(path, "w") as f:
            json.dump(self._results, f)

    
    def get_results(self):
        return self._results
    

    def run(self):
        self._path.mkdir(parents=True, exist_ok=True)
        
        ###### if resume, load checkpoint ########
        if self._resume and self._path is not None:
            checkpoints = sorted(self._path.glob(f"checkpoint_*.json"))
            if checkpoints:
                last_checkpoint = checkpoints[-1]
                self._load_checkpoint(last_checkpoint)

        
        self._setup()
        seed = self._current_seed if self._resume else self._seed
        print(f"\n* starting experiment. AE condition: {self._prestep}")
        
        for run in range(self._run, self._runs):
            print(f"  - run {run}")
            self._set_seed(seed)

            if self._prestep:
                autopop = create_AE_pop(
                    self._model2,
                    self._pop,
                    self._input_shape,
                    self._AEepochs,
                    self._stride,
                    self._train_loader # requires dataloader...
                )
                print("  - autoencoder population has been created..")
            
            evolver = GeneticAlgorithmV2(
                self._model1(stride=self._stride),
                True,
                self._pop,
                self._train, # requires raw set to pass to _trainval_loaders...
                problem=self._problem
            )
            
            if self._prestep:
                evolver.transfer_popV2(
                    autopop, self._model1, self._input_shape, self._classes
                )
            
            evolver.evolve(
                generations=self._gens,
                subset_fraction=0.07,
                m_r_min=self._m_r_min,
                m_r_max=self._m_r_max,
                m_r_decay = self._decay,
                m_s=self._m_strength,
                mode=self._m_mode
            )

            # extract results
            fitintime = evolver.get_fitintime() # list of avg.fits
            finalfit = fitintime[-1] # lat gen's fit
            result = {"fit_in_time": fitintime, "finalfit":finalfit}
            self._results.append(result)

            # update run and seed for checkpointing
            self._run +=1
            self._seed +=1
            seed +=1

            if self._prestep:
                del autopop # make sure that GPU is freed
                torch.cuda.empty_cache() # ⁉️
            
            if self._check:
                self._current_seed = seed
                checkpath = self._path/f"checkpoint_{run}.json"
                self._checkpoint(checkpath) #⛔️
                print(f"\n  - hit checkpoint! next run coming..")
        

        resultpath = self._path/"results.json"
        if resultpath is not None:
            self._save_results(resultpath)