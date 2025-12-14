import os
from pathlib import Path
from whole.nsga import experiment as exp
from whole.nsga.models import TinyConvClassifier, TinyFlexyConvAE
import torch

# ––––––––––––––––––––––––––––––
# python -m whole.nsga.nsga #??????????
# ––––––––––––––––––––––––––––––

# –––––----- experiment details ––––––––––– #
dataset = "mnist" # mnist, cifar, fashion
problem = "classification"
prestep = False 
condition = "AE" if prestep else "noAE"
subset_fraction=0.07

model1 = TinyConvClassifier # main model
model2 = TinyFlexyConvAE # autoencoder for AE condition
interval = [1, 4]
pop = 10
AEepochs = 4
bound_runs = 2
bound_gens = 2
evo_runs = 3 # 'experiment' runs
evo_gens = 3

seed = 34
interspecies_r = 0.1 
mutation_rate_min = 0.01
mutation_rate_max = 0.3
mutation_rate_decay = True
mutation_strength = 0.2
mutation_mode = "light"

resume = False # from checkpoint.. set True ONLY IF CHECKPOINT_SAVE EXISTS
mydevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
git = False
checkpoint = False



print(os.getcwd())
# move to appropriate nsga directory
name = "msc_project_repo"
basedir = None
cwd = Path().cwd().resolve()
if cwd.name != name:
    try:
        basedir = next(e for e in cwd.rglob("*") if e.is_dir() and e.name == name)
    except StopIteration:
        raise FileNotFoundError("could not find nsga directory")
else:
    basedir = cwd

print(os.getcwd())
basepath = basedir / "whole" / "tests" / "NSGA" / f"{dataset}" / f"{condition}"

basepath.mkdir(parents=True, exist_ok=True)


# --–––---––– initialisation –––---–––-- #
print("\nstarting workflow!")
workflow = exp.ExperimentV4(
    model1=model1,
    model2=model2,
    pop=pop,
    dataset=dataset,
    subset_fraction=subset_fraction,
    problem=problem,
    interval=interval,
    seed=seed,
    experiment_path=basepath,
    prestep=prestep, # autoencoder condition?
    AEepochs=AEepochs,

    bound_runs=bound_runs,
    bound_gens=bound_gens,
    evo_runs=evo_runs,
    evo_gens=evo_gens,
    
    intersp_cross_rate=interspecies_r,
    mutation_rate_min=mutation_rate_min,
    mutation_rate_max=mutation_rate_max,
    mutation_rate_decay=mutation_rate_decay,
    mutation_strength=mutation_strength,
    mutation_mode=mutation_mode,
    
    
    resume=resume, # from checkpoint?
    device=mydevice,
    git=git,
    checkpoint=checkpoint
)


# --––––- experiment --–––- #
workflow.run()


print("\nexperiment's over!")

# results = workflow.get_results()
        

