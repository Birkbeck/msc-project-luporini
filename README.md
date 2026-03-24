# msc-project-luporini

##- problem
##- solution/strategy
##- results


### Welcome!

This repository's root contains the full implementation and results for single- and multi-objective neuroevolution applied to convolutional classifiers. The material included supports quick pipeline inspection, result replication and further experimentation.

    Important: a PyTorch environment is required to run the code.

## Running experiments
- ga-playground.py contains the single-objective GA pipeline
- genalgo.ipynb enables the single-objective experiment as run for the report 

- nsga-playground.py contains the multi-objective NSGA pipeline
- nsga.ipynb enables the multi-objective experiment as run for the report


## Analysis
- analysis.ipynb contains the full single-objective GA analysis
- NSGA_analysis.ipynb contains the full multi-objective NSGA analysis


Playground scripts and analysis notebooks can run as-is.

Playgrounds are ready to go with small-scale experiments, and they perform smoothly on my 2020 consumer-grade laptop. However, the full-scale experimental notebooks were implemented on Google Colab with GPU acceleration.

Analysis notebooks are initialised with the project's analyses, which may be replicated straightaway.

More granular implementation details are found in individual modules stored in the repository, whose structure is as follows:

root/
    |- whole/
        |- ga/  # single-objective GA implementation
        |- nsga/  # multi-objective NSGA implementation
        |- practising/  #  training material and legacy scripts
        |- tests/  #  * experimental results *
            |- GA/ 
            |- NSGA/

The directory ./whole/tests/ stores all experimental results for both GA and NSGA experiments.
The directories ./whole/ga/ and ./whole/nsga/ contain the full implementation for GA and NSGA, respectively.


# -------------------------------------------------------------
# --------------- CODE: what and where ----------------------
# -------------------------------------------------------------

# GA implementation (./whole/ga/)
- genalgo.py contains the GeneticAlgorithm and Experiment classes
  .genalgo.GAExperiment enables the final GA abstraction
  .genalgo.GeneticAlgorithmV2 enables the single-objective evolutionary algorithm

- fitness.py contains fitness-related functions
- models.py contains classifier and autoencoder classes and related functions
- operators.py contains crossover and mutation functions
- utils.py contains functions for flattening and reconstructing PyTorch models


# multi-objective NSGA implementation (./whole/nsga/)
- experiment.py contains the Experiment class
  .experiment.ExperimentV4 is the final NSGA abstraction 

- nsga.py contains the NSGA class and related functions
  .nsga.NSGA2 enables the multi-objective evolutionary algorithm

- fitness.py same as above
- models.py same as above
- operators.py same as above
- utils.py same as above but for adjusted for NSGA


In both the GA and NSGA implementations, the Experiment class is the highest-level abstraction that playground.py scripts use.


