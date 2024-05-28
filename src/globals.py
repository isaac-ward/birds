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

#Fitness Constants
# Define the Z-value below which the creature is considered "dead"
TOO_LOW_Z = -1
# Define the fitness penalty for going too low
FITNESS_PENALTY_TOO_LOW = -1000
# Define the distance threshold within which a waypoint is considered "reached"
WAYPOINT_THRESHOLD = 1.0
# Define the reward/penalty for reaching a waypoint
WAYPOINT_REWARD = 100
WAYPOINT_PENALTY = -50

# Global functions
def wrapRads(angle:float) -> float:
    wrapped_angle = (angle + np.pi) % (2 * np.pi) - np.pi
    return wrapped_angle