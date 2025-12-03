from pathlib import Path
import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from whole.ga.models import TinyFlexyConvAE, TinyConvClassifier, create_AE_pop
import whole.ga.genalgo as gen


# train = MNIST("./whole/datasets", download=True, train=True, transform=ToTensor())
# test = MNIST("./whole/datasets", download=True, train=False, transform=ToTensor())
# train_loader = DataLoader(train, batch_size=30, shuffle=True)

autoencoder = TinyFlexyConvAE
classifier = TinyConvClassifier
stride = 2
pop = 10
dataset = "fashion"
subset_fraction = 0.2  # have used 0.07 so far.. ⛔️ 
problem = "classification"
seed = 34
AEepochs = 4
classes = 10
runs = 30
gens = 30
mutation_rate_min = 0.01
mutation_rate_max = 0.2
mutations_rate_decay = True
mutation_strength = 0.2
mutation_mode = "light"
my_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
resume = False
checkpoint = False


prestep = False 
if prestep:
    condition = "AE"
else:
    condition = "noAE"


# --------- file management -------- #
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
basepath = basedir / "whole" / "tests" / "GA" / f"{dataset}" / f"{condition}"
basepath.mkdir(parents=True, exist_ok=True)


# ---------- experiment ----------- #
workflow = gen.GAExperiment(
    model1=classifier, # task model
    model2=autoencoder, # autoencoder
    stride=stride,
    pop=pop,
    dataset=dataset,
    subset_fraction=subset_fraction,
    problem=problem,
    seed=seed,
    experiment_path=basepath,
    prestep=prestep,
    AEepochs=AEepochs,
    classes=classes,
    runs=runs,
    gens=gens,
    mutation_rate_min=mutation_rate_min,
    mutation_rate_max=mutation_rate_max,
    mutation_rate_decay=mutations_rate_decay,
    mutation_strength=mutation_strength,
    mutation_mode=mutation_mode,
    my_device=my_device,
    resume=resume,
    checkpoint=checkpoint
)


workflow.run()

print("experiment's finished!")
