import experiment as exp
from architectures import TinyConvClassifier

workflow = exp.Experiment(
    model=TinyConvClassifier,
    pop=10,
    dataset="mnist",
    problem="classification",
    evo_gens=2,
    bound_gens=2,
    interval=[1, 4],
    seed=37,
    best_path="./best.pth",
    check_path="./checkpoint.json",
    resume=False
)

workflow.run(
    bound_estimation_runs=2, runs=2
)

