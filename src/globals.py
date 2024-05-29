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
    Gene("airfoil_handwing", 0.0, 1.0), #NOTE: Change later
    Gene("basis_left_const", -1.0, 1.0),
    Gene("basis_left_poly1", -1.0, 1.0),
    Gene("basis_left_poly2", -1.0, 1.0),
    Gene("basis_left_poly3", -1.0, 1.0),
    Gene("basis_left_poly4", -1.0, 1.0),
    Gene("basis_left_poly5", -1.0, 1.0),
    Gene("basis_left_sinamp1", -1.0, 1.0),
    Gene("basis_left_sinfreq1", -4.0, 4.0),
    Gene("basis_left_sinamp2", -1.0, 1.0),
    Gene("basis_left_sinfreq2", -4.0, 4.0),
    Gene("basis_left_sawtooth", 0.1, 5.0),
    Gene("basis_left_expamp1", -1.0, 1.0),
    Gene("basis_left_exppwer1", -4.0, 4.0),
    Gene("basis_left_expamp2", -1.0, 1.0),
    Gene("basis_left_exppwr2", -4.0, 4.0),
    Gene("basis_right_const", -1.0, 1.0),
    Gene("basis_right_poly1", -1.0, 1.0),
    Gene("basis_right_poly2", -1.0, 1.0),
    Gene("basis_right_poly3", -1.0, 1.0),
    Gene("basis_right_poly4", -1.0, 1.0),
    Gene("basis_right_poly5", -1.0, 1.0),
    Gene("basis_right_sinamp1", -1.0, 1.0),
    Gene("basis_right_sinfreq1", -4.0, 4.0),
    Gene("basis_right_sinamp2", -1.0, 1.0),
    Gene("basis_right_sinfreq2", -4.0, 4.0),
    Gene("basis_right_sawtooth", 0.1, 5.0),
    Gene("basis_right_expamp1", -1.0, 1.0),
    Gene("basis_right_exppwer1", -4.0, 4.0),
    Gene("basis_right_expamp2", -1.0, 1.0),
    Gene("basis_right_exppwr2", -4.0, 4.0)
]

# Global aerodynamic parameters
AIR_DENSITY = 1.225 #kg/m^3
GRAVITY = 9.81      #m/s^2
BIRD_DENSITY = 10   #kg/m^3

# Fitness Constants
# Define the distance threshold within which a waypoint is considered "reached"
WAYPOINT_THRESHOLD = 1.0
# Define the reward/penalty for reaching a waypoint
WAYPOINT_REWARD = 100
WAYPOINT_PENALTY = -50

# Global airfoil data
# airfoil_database parameters:
# 1) dcL/dAlpha (1/rad)
# 2) angle of min stall (rad)
# 3) angle of max stall (rad)
# 4) cD0
# 5) thickness
AIRFOIL_DATABASE = {
    "NACA 0012": [5.72958, -0.122173, 0.122173, 0.021, 0.12]
}

# Global functions
def wrapRads(angle:float) -> float:
    wrapped_angle = (angle + np.pi) % (2 * np.pi) - np.pi
    return wrapped_angle