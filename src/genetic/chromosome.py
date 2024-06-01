import numpy as np 
import random
import pickle
import copy

class Gene:
    """
    Helper class to store a named gene and explicitly define
    its feasible range
    """
    def __init__(self, name, min_val, max_val):
        """
        Allows gene to be initialized with specific values
        """
        self.name = name
        # Inclusive
        self.min_val = min_val
        self.max_val = max_val
    
    def __getstate__(self):
        return self.__dict__
    def __setstate__(self, state):
        self.__dict__.update(state)
    
class GeneDiscrete(Gene):
    """
    A subclass of Gene that only allows discrete values inclusive
    of the min and max values
    """
    def __init__(self, name, min_val, max_val):
        # Inclusive
        super().__init__(name, min_val, max_val)
        assert isinstance(min_val, int) and isinstance(max_val, int), "Discrete genes must have integer min and max values"


from globals import CHROMOSOME_DEFINITION

class Chromosome:
    """
    Helper class to store named genes neatly and explicitly
    define their feasible ranges
    """

    def __init__(
        self,
        vector,
    ):
        """
        Allows chromosome to be initialized with specific values
        """

        # Check that the vector has the correct length
        assert len(vector) == len(CHROMOSOME_DEFINITION)

        # Check that each gene is within its feasible range
        for i, gene in enumerate(CHROMOSOME_DEFINITION):
            assert gene.min_val <= vector[i] <= gene.max_val, f"Gene {gene.name} out of bounds {gene.min_val} <= {vector[i]} <= {gene.max_val}"

        # And that the discrete genes are integers
        for i, gene in enumerate(CHROMOSOME_DEFINITION):
            if isinstance(gene, GeneDiscrete):
                assert isinstance(vector[i], int), f"Gene {gene.name} must be an integer"

        # Assign the genes to named attributes
        for i, gene in enumerate(CHROMOSOME_DEFINITION):
            setattr(self, gene.name, vector[i])

    @classmethod
    def random_init(cls):
        """
        Initializes a chromosome with random values within feasible ranges
        """
        vector = []
        for gene in CHROMOSOME_DEFINITION:
            if isinstance(gene, GeneDiscrete):
                vector.append(np.random.randint(gene.min_val, gene.max_val + 1))
            else:
                vector.append(np.random.uniform(gene.min_val, gene.max_val))
        return cls(vector)
    
    def __str__(self):
        """
        Returns a string representation of the chromosome
        """
        s = ""
        for gene in CHROMOSOME_DEFINITION:
            s += f"{gene.name} = {getattr(self, gene.name)}\n"
        return s
    
    def crossover(self, other):
        # Single point crossover (i.e., the first part of one chromosome is combined with the second part of the other)
        crossover_point = random.randint(0, len(CHROMOSOME_DEFINITION) - 1)
        new_vector = [getattr(self, gene.name) if i <= crossover_point else getattr(other, gene.name) for i, gene in enumerate(CHROMOSOME_DEFINITION)]
        return Chromosome(new_vector)
    
    def mutate(self):
        # If it's discrete we can just randomly select a new value
        new_vector = []
        for gene in CHROMOSOME_DEFINITION:
            if isinstance(gene, GeneDiscrete):
                new_vector.append(int(np.random.randint(gene.min_val, gene.max_val + 1)))
            else:
                # Otherwise we can just add some Gaussian noise
                sigma = 0.15 * (gene.max_val - gene.min_val)
                new_vector.append(np.clip(getattr(self, gene.name) + np.random.normal(0, sigma), gene.min_val, gene.max_val))

        # Return a new chromosome with the mutated values
        return Chromosome(np.array(new_vector))

    def __getstate__(self):
        return self.__dict__
    def __setstate__(self, state):
        self.__dict__.update(state)