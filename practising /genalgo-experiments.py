from torch import nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from architectures import TinyConvClassifier
import genalgo as gen

train = MNIST("./datasets", download=True, train=True, transform=ToTensor())
test = MNIST("./datasets", download=True, train=False, transform=ToTensor())

model = TinyConvClassifier(stride=2)

evolver = gen.GeneticAlgorithmV2(
    model, 10, train, problem="classification"
)

evolver.evolve(50, m_r=0.03, m_s=0.2)