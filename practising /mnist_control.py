import experiment as exp
from architectures import TinyConvClassifier

workflow = exp.Experiment(
    model=TinyConvClassifier,
    pop=10,
    dataset="mnist",
    problem="classification",
    evo_gens=5,
    bound_gens=2
)