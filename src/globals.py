import numpy as np

from genetic.chromosome import Gene

# TODO: Kusal to edit the chromosome definition to parameterize
# the morphology of the creature (not behavior). Kusal, for
# the purpose of learning it's important that every gene in
# the chromosome be a scalar. If you need to encode a vector
# of values, you can use multiple genes to encode each value.
# For example, if you need to encode a 3D position of the com
# of the creature, you can use 3 genes to encode this position:
# "com_x", "com_y", "com_z". I think this structure is nice but
# you can use a different structure if you prefer; to add 
# /edit genes you just need to add them to the following chromosome
# definition, along with their feasible ranges.

# Define the order of the genes in the chromosome
# and their feasible ranges
CHROMOSOME_DEFINITION = [
    # Gene(gene_name, min_val, max_val)
    Gene("wingspan", 0.0, 10.0),
    Gene("norm_wrist_position", 0.0, 10.0),
    Gene("wing_root_chord", 0.0, 1.0),
    Gene("taper_armwing", 0.0, 1.0),
    Gene("taper_handwing", 0.0, 1.0),
    Gene("COG_position", 0.0, 1.0),
    Gene("airfoil_armwing", 0.0, 1.0), #NOTE: Change later
    Gene("airfoil_handwing", 0.0, 1.0) #NOTE: Change later
]

# If you go too low it's negative infinity, so you never
# get selected as a parent for the next generation
# Note that the creature starts at the origin, and down
# is negative z
TOO_LOW_Z = -100
FITNESS_PENALTY_TOO_LOW = -np.inf

# Global aerodynamic parameters
air_density = 1.225             #kg/m^3
gravity_acceleration = 9.81     #m/s^2
bird_density = 10               #kg/m^3