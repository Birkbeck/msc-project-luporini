# msc-project-source-code-files-24-25-lupor01

Welcome!

This repository's root contains the full implementation and results for single- and multi-objective neuroevolution applied to convolutional classifiers. The material included supports quick pipeline inspection, result replication and further experimentation.

## Running experiments
- ga-playground.py contains a small-scale single-objective GA example
- genalgo.ipynb contains the full-scale single-objective GA pipeline that appears in the report 

- nsga-playground.py contains the multi-objective NSGA example
- nsga.ipynb contains the full-scale multi-objective NSGA pipeline that appears in the report

## Analysis
- analysis.ipynb contains the single-objective GA analysis
- NSGA_analysis.ipynb contains the multi-objective NSGA analysis


The playground scripts and analysis notebooks can run as-is. Playgrounds are set for small-scale experiments, and they run smoothly on my 2020 consumer-grade laptop. However, the full-scale experimental notebooks run on Google Colab with GPU acceleration.

Analysis notebooks are initialised with the project's analyses, which may be replicated straightaway.

More granular implementation details are found in individual modules stored in the repository, whose structure is as follows:

root/
    |- whole/
        |- ga/  # single-objective GA implementation
        |- nsga/  # multi-objective NSGA implementation
        |- practising/  #  training material and legacy scripts
        |- tests/  #  experimental results
            |- GA/ 
            |- NSGA/

The directory ./whole/tests/ stores all experimental results for both GA and NSGA experiments.
The directories ./whole/ga/ and ./whole/nsga/ contain the full implementation for GA and NSGA, respectively.


# -------------------------------------------------------------
# --------------- CODE: what and where ---------------------
# -------------------------------------------------------------

# single-objective GA implementation (./whole/ga/)
- fitness.py contains fitness-related functions
- models.py contains classifier and autoencoder classes and related functions
- operators.py contains functions for crossover and mutation
- utils.py contains functions for flattening and reconstructing PyTorch models
- genalgo.py contains the core GeneticAlgorithm and the Experiment classes


# multi-objective NSGA implementation (./whole/nsga/)
- fitness.py same as above
- models.py same as above
- operators.py same as above
- utils.py same as above
- nsga.py contains the NSGA class, nondominated sorting and crowding distance functions
- experiment.py contains the Experiment class

Both in GA and NSGA implementations, the Experiment class is the highest-level abstraction that playground scripts use.


