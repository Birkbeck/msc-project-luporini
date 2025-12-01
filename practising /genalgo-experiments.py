from pathlib import Path
import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from architectures import TinyFlexyConvAE, TinyConvClassifier, create_AE_pop
import genalgo as gen

train = MNIST("./datasets", download=True, train=True, transform=ToTensor())
test = MNIST("./datasets", download=True, train=False, transform=ToTensor())
train_loader = DataLoader(train, batch_size=30, shuffle=True)

autoencoder = TinyFlexyConvAE
classifier = TinyConvClassifier
stride = 2
pop = 10
dataset = "mnist"
problem = "classification"
seed = 34
AEepochs=4,
classes=10,
evo_runs=1,
evo_gens=30,
mutation_rate=0.2,
mutation_strength=0.2,
mutation_mode="light",
my_device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
resume=False,
checkpoint=True


prestep = False 
if prestep:
    condition = "AE"
else:
    condition = "noAE"


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
basepath = basedir / "practising" / f"{dataset}" / f"{condition}"
basepath.mkdir(parents=True, exist_ok=True)


workflow = gen.GAExperiment(
    model1=classifier, # task model
    model2=autoencoder, # autoencoder
    stride=stride,
    pop=pop,
    dataset=dataset,
    problem=problem,
    seed=seed,
    experiment_path=basepath,
    prestep=prestep,
    AEepochs=AEepochs,
    classes=classes,
    evo_runs=evo_runs,
    evo_gens=evo_gens,
    mutation_rate=mutation_rate,
    mutation_strength=mutation_strength,
    mutation_mode=mutation_mode,
    my_device=my_device,
    resume=resume,
    checkpoint=checkpoint
)


workflow.run()




