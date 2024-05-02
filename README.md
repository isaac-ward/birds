# birds 

This is a project for evolving virtual creatures using genetic algorithms in tandem with neural networks.

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