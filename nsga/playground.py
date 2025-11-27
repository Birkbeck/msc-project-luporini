import os
from pathlib import Path
import experiment as exp
import torch
from models import TinyConvClassifier, TinyFlexyConvAE



# –––––----- experiment details ––––––––––– #
model1 = TinyConvClassifier # main model
model2 = TinyFlexyConvAE # autoencoder for AE condition

interval = [1, 4]

pop = 5
bound_runs = 2
bound_gens = 2
evo_runs = 3 # 'experiment' runs
evo_gens = 5
interspecies_r = 0.1 
mutation_rate = .78 # stay between 0.1% - 1%
mutation_strength = .2 
mutation_mode = "light"

seed = 37
resume = True # from checkpoint?
mydevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
git = False
checkpoint = True

dataset = "mnist" # mnist, cifar, fashion
problem = "classification"

# enable AE condition?
prestep = False 
if prestep:
    condition = "AE"
else:
    condition = "noAE"


print(os.getcwd())
# move to appropriate nsga directory
name = "nsga"
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
basepath = basedir / "tests" / f"{dataset}" / f"{condition}"

basepath.mkdir(parents=True, exist_ok=True)


# --–––---––– initialisation –––---–––-- #
print("\nstarting workflow!")
workflow = exp.ExperimentV3(
    model1=model1,
    model2=model2,
    pop=pop,
    dataset=dataset,
    problem=problem,
    bound_runs=bound_runs,
    bound_gens=bound_gens,
    evo_runs=evo_runs,
    evo_gens=evo_gens,
    prestep=prestep, # autoencoder condition?
    intersp_cross_rate=interspecies_r,
    mutation_rate=mutation_rate,
    mutation_strength=mutation_strength,
    mutation_mode=mutation_mode,
    interval=interval,
    seed=seed,
    experiment_path=basepath,
    resume=resume, # from checkpoint?
    device=mydevice,
    git=git,
    checkpoint=checkpoint
)


# --––––- experiment --–––- #
workflow.run()


print("\nexperiment's over.. g'byeeee")

# results = workflow.get_results()
        

