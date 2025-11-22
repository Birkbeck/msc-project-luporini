# don't call mnist_control.py
# just call playgroud.py ⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️

import sys
import os
from pathlib import Path
import experiment as exp
import torch
from architectures import TinyConvClassifier, TinyFlexyConvAE


model1 = TinyConvClassifier
model2 = TinyFlexyConvAE
pop = 10
dataset = "mnist" # mnist, cifar, fashion
problem = "classification"
prestep = False # autoencoder condition?
prestep_gens = None

mutation_strength = .3 
mutation_probability = .1
evo_gens = 2
bound_gens = 2
interval = [1, 4]
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

fronts = workflow.get_fronts()     
avg_convs = workflow.get_avg_convs()        
convs_in_time = workflow.get_convs_in_time()
empirical_bounds = workflow.get_empirical_bounds()
        

