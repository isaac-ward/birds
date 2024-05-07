# birds 

This is a project for evolving virtual creatures using genetic algorithms in tandem with neural networks.

# Project layout

- ```src/genetic``` is a module that contains the gene/chromosome class, fitness selection/evaluation functions, and virtual creature class.
- ```src/learning``` is a module that contains all the neural network & data code.
- ```src/logs``` is a folder for storing logging data.
- ```src/dynamics.py``` contains the code for the physical simulation of creatures.
- ```src/globals.py``` contains the chromosome description and editing it changes the design space.
- ```src/visuals.py``` contains the code for visualizing the runs.
- ```main_genetic.py``` is the main file for running the genetic algorithm.
- ```main_learning.py``` is the main file for running the neural network learning.
- ```main_test_dynamics.py``` is the main file for testing out how the dynamics affect a virtual creature over time.

# Running this

The project is implemented in stock python with standard packages, though a file ```env.yaml``` is provided which will automatically install the required packages into a new environment called 'birds' using conda, [if conda is installed on your machine](https://docs.anaconda.com/free/miniconda/miniconda-install/) and if the command is ran from the root folder.

```
conda env create --file env.yaml
conda activate birds
```

For regenerating the environment file after making development changes, run the following in the root folder of the project:
```
conda activate birds
<install new packages>
conda env export --no-builds | grep -v "^prefix: " > env.yaml
```