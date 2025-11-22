# don't call mnist_control.py
# just call playgroud.py ⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️

from pathlib import Path
import experiment as exp
import torch
from architectures import TinyConvClassifier, TinyFlexyConvAE


dataset = "mnist" # mnist, cifar, fashion
problem = "classification"
prestep = False # enable AE condition
prestep_gens = None

model1 = TinyConvClassifier # main model
model2 = TinyFlexyConvAE # autoencoder for AE condition

interval = [1, 4]
pop = 10
evo_gens = 2
bound_gens = 2

mutation_strength = .3 
mutation_probability = .1
seed = 37

resume = False # from checkpoint?

mydevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if prestep:
    condition = "AE"
else:
    condition = "noAE"

cwd = Path().cwd().resolve()
basepath = cwd / "project" / "experiments" / f"{dataset}" / f"{condition}"


workflow = exp.Experiment(
    model1=model1,
    model2=model2,
    pop=pop,
    dataset=dataset,
    problem=problem,
    evo_gens=evo_gens,
    bound_gens=bound_gens,
    interval=interval,
    seed=seed,
    experiment_path=basepath,
    resume=resume, # from checkpoint?
    prestep=prestep, # autoencoder condition?
    device=mydevice
)

workflow.run(
    bound_estimation_runs=2, runs=2
)

results = workflow.get_results()
        

