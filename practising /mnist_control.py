# don't call mnist_control.py
# just call playgroud.py ⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️⛔️

from pathlib import Path
import experiment as exp
import torch
from architectures import TinyConvClassifier, TinyFlexyConvAE


dataset = "mnist" # mnist, cifar, fashion
problem = "classification"
prestep, prestep_gens = True, 2 # enable AE condition

model1 = TinyConvClassifier # main model
model2 = TinyFlexyConvAE # autoencoder for AE condition

interval = [1, 4]

if prestep and prestep_gens is not None:
    AE_pop = 10

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

print("\nstarting workflow..")
workflow = exp.Experiment(
    model1=model1,
    model2=model2,
    pop=pop,
    AEpop=AE_pop,
    dataset=dataset,
    problem=problem,
    evo_gens=evo_gens,
    prestep=prestep, # autoencoder condition?
    prestep_gens=prestep_gens,
    mutation_strength=mutation_strength,
    mutation_prob=mutation_probability,
    bound_gens=bound_gens,
    interval=interval,
    seed=seed,
    experiment_path=basepath,
    resume=resume, # from checkpoint?
    device=mydevice
)

workflow.run()

print("experiment's over.. g'byeeee")

# results = workflow.get_results()
        

