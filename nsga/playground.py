import os
from pathlib import Path
import experiment as exp
import torch
from models import TinyConvClassifier, TinyFlexyConvAE



# –––––----- experiment details ––––––––––– #
dataset = "mnist" # mnist, cifar, fashion
problem = "classification"

model1 = TinyConvClassifier # main model
model2 = TinyFlexyConvAE # autoencoder for AE condition

interval = [1, 4]

prestep, prestep_gens = False, 0 # enable AE condition
if prestep and prestep_gens is not None:
    AE_pop = 10
else:
    AE_pop = None

pop = 10
bound_runs = 3
bound_gens = 3
evo_runs = 2
evo_gens = 3
mutation_strength = .2 
mutation_probability = .01 # stay between 0.1% - 1%
mutation_mode = "light"

seed = 37
resume = False # from checkpoint?
mydevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
git = True
checkpoint = False

if prestep:
    condition = "AE"
else:
    condition = "noAE"

print(os.getcwd())
# move to appropriate nsga directory
name = "nsga"
basedir = None
cwd = Path().cwd().resolve()
for e in cwd.rglob("*"):
    if e.is_dir() and e.name == name:
        basedir = e
        break
if basedir is None:
    raise FileNotFoundError("could not find nsga directory")

os.chdir(basedir)
print(os.getcwd())
basepath = basedir / "tests" / f"{dataset}" / f"{condition}"



# --–––---––– initialisation –––---–––-- #
print("\nstarting workflow!")
workflow = exp.Experiment(
    model1=model1,
    model2=model2,
    pop=pop,
    AEpop=AE_pop,
    dataset=dataset,
    problem=problem,
    bound_runs=bound_runs,
    bound_gens=bound_gens,
    evo_runs=evo_runs,
    evo_gens=evo_gens,
    prestep=prestep, # autoencoder condition?
    prestep_gens=prestep_gens,
    mutation_strength=mutation_strength,
    mutation_prob=mutation_probability,
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
        

