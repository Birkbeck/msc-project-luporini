import sys
import os
from pathlib import Path
import experiment as exp
from architectures import TinyConvClassifier


model = TinyConvClassifier
pop = 10
dataset = "mnist" # mnist, cifar, fashion
problem = "classification"
evo_gens = 2
bound_gens = 2
interval = [1, 4]
seed = 37
resume = False

cwd = Path().cwd()
best_path = cwd /"practising"/"best"
check_path = cwd /"practising"/"checkpoints"


workflow = exp.Experiment(
    model=model,
    pop=pop,
    dataset=dataset,
    problem=problem,
    evo_gens=evo_gens,
    bound_gens=bound_gens,
    interval=interval,
    seed=seed,
    best_path=best_path,
    check_path=check_path,
    resume=resume
)

workflow.run(
    bound_estimation_runs=2, runs=2
)

fronts = workflow.get_fronts()     
avg_convs = workflow.get_avg_convs()        
convs_in_time = workflow.get_convs_in_time()
empirical_bounds = workflow.get_empirical_bounds()
        

