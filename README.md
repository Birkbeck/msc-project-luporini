# Evolving lightweight convolutional classifiers

Deploying deep neural networks in real-world settings requires balancing accuracy and computational costs. Increasing model parameters improves expressiveness, but it also introduces memory and storage constraints.

This project implements neuroevolution to optimise CNN classifiers for accuracy and size simultaneously. It is closely related to Neural Architecture Search and is inspired by recent research on visual assistive technologies for the visually impaired (Kuriakose et al., 2023)

## 1. Key contributions

- Implemented single- and multi-objective genetic algorithms from scratch in PyTorch
- Designed end-to-end evolutionary framework for evolving CNN architectures
- Demonstrated feasibility of the framework to evolve CNNs under multiple objectives

## 2. The algorithm

The final NSGA pipeline evolved populations of heterogeneous architectures. Evolutionary selection favoured high accuracy and small size.

Performance was evaluated using:

- avg. accuracy across generations (classification accuracy throughout evolution, averaged across a number of runs).
- avg. convergence across generations (the Euclidean distance to the ideal solution in the objective space).

## 3. Results

Population performance consistently improves across generations (Fig.1), with populations converging towards the Pareto-optimal front (Fig.2).

This demonstrates that NSGA-II can effectively explore the trade-offs between accuracy and model size, converging towards balanced architectures.

![Accuracy over generations](/whole/tests/visualisation/acc_in_time.png)
Figure 1. Avg. accuracy increases over generations.

![Convergence over generations](/whole/tests/visualisation/conv_in_time.png)
Figure 2. Avg. distance from the ideal solution decreases over generations.

- **NSGA_analysis.ipynb** contains the full multi-objective NSGA analysis

## 4. Limitations and applications

While NSGA-II can be applied to CNN architecture discovery:

- it is computationally expensive compared to gradient-based methods
- it depends heavily on design choices such as network encoding, selection pressure, and evolutionary schemes
- scalability to large architectures and/or datasets remains challenging

Ultimately, neuroevolution may be most suitable for hyper-specialised settings, where search space and computationl costs remain manageable.

# Running experiments

- **ga-playground.py** is the single-objective GA pipeline

- **nsga-playground.py** is the multi-objective NSGA pipeline

```bash
# create environment
conda env create -f requirements.yml

# run GA experiment
python ga-playground.py

# run NSGA experiment (GPU acceleration is recommended for full-scale, multi-objective runs)
python nsga-playground.py
```

⚠️ Playground scripts and analysis notebooks can be run as-is. Small-scale, single-objective experiments run reasonably fast on my 2020 MacBook (1.4 GHz Quad-Core Intel Core i5). However, the full-scale NSGA script was implemented on Google Colab with GPU acceleration.

⚠️ Analysis notebooks are initialised with the project's analyses, which may be replicated straightaway.

# Repo structure

- whole/ga/ -- single-objective GA implementation
- whole/nsga/ -- multi-objective NSGA implementation
- whole/tests/ -- experimental results and visualisation
- whole/practising/ -- training material and prototyping material
