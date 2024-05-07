import numpy as np 

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
        self.min_val = min_val
        self.max_val = max_val

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
            assert gene.min_val <= vector[i] <= gene.max_val

        # Assign the genes to named attributes
        for i, gene in enumerate(CHROMOSOME_DEFINITION):
            setattr(self, gene.name, vector[i])

    @classmethod
    def random_init(cls):
        """
        Initializes a chromosome with random values within feasible ranges
        """
        vector = np.array([
            np.random.uniform(gene.min_val, gene.max_val)
            for gene in CHROMOSOME_DEFINITION
        ])
        return cls(vector)
    
    def __str__(self):
        """
        Returns a string representation of the chromosome
        """
        s = ""
        for gene in CHROMOSOME_DEFINITION:
            s += f"{gene.name} = {getattr(self, gene.name)}\n"
        return s
