import numpy as np

from genetic.chromosome import Gene

# Define the order of the genes in the chromosome
# and their feasible ranges
CHROMOSOME_DEFINITION = [
    # Gene(gene_name, min_val, max_val)
    Gene("wingspan", 0.0, 10.0),
    Gene("norm_wrist_position", 0.0, 1.0),
    Gene("wing_root_chord", 0.0, 1.0),
    Gene("taper_armwing", 0.0, 1.0),
    Gene("taper_handwing", 0.0, 1.0),
    Gene("norm_COG_position", 0.0, 1.0),
    Gene("airfoil_armwing", 0.0, 1.0), #NOTE: Change later
    Gene("airfoil_handwing", 0.0, 1.0) #NOTE: Change later
]

# Global aerodynamic parameters
AIR_DENSITY = 1.225 #kg/m^3
GRAVITY = 9.81      #m/s^2
BIRD_DENSITY = 10   #kg/m^3

# Global functions
def wrapRads(angle:float) -> float:
    wrapped_angle = (angle + np.pi) % (2 * np.pi) - np.pi
    return wrapped_angle