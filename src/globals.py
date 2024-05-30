import numpy as np

from genetic.chromosome import Gene

# Define the order of the genes in the chromosome
# and their feasible ranges
CHROMOSOME_DEFINITION = [
    # Gene(gene_name, min_val, max_val)
    Gene("wingspan", 3, 6),
    Gene("norm_wrist_position", 0.1, 0.9),
    Gene("wing_root_chord", 0.1, 0.9),
    Gene("taper_armwing", 0.5, 0.9),
    Gene("taper_handwing", 0.1, 0.5),
    Gene("norm_COG_position", 0.1, 0.9),
    Gene("airfoil_armwing", 0.1, 0.9),  #NOTE: Change later
    Gene("airfoil_handwing", 0.1, 0.9), #NOTE: Change later
    # Control parameters
    # Constant parameters
    Gene("basis_left_const", -1.0, 1.0),
    Gene("basis_left_poly1", 0, 0),
    Gene("basis_left_poly2", 0, 0),
    Gene("basis_left_poly3", 0, 0),
    Gene("basis_left_poly4", 0, 0),
    Gene("basis_left_poly5", 0, 0),
    # 2 sinusoids
    Gene("basis_left_sinamp1", -1.0, 1.0),
    Gene("basis_left_sinfreq1", -4.0, 4.0),
    Gene("basis_left_sinamp2", -1.0, 1.0),
    Gene("basis_left_sinfreq2", -4.0, 4.0),
    # A sawtooth function
    Gene("basis_left_sawtooth", 1000000, 3000000), # TODO unused! Discountinuities mess with the dynamics - no sawtooth allowed!
    # 2 exponentials
    Gene("basis_left_expamp1", 0, 0),
    Gene("basis_left_exppwr1", 0, 0), 
    Gene("basis_left_expamp2", 0, 0),
    Gene("basis_left_exppwr2", 0, 0),
    # And same for the right
    Gene("basis_right_const", -1.0, 1.0),
    Gene("basis_right_poly1", 0, 0),
    Gene("basis_right_poly2", 0, 0),
    Gene("basis_right_poly3", 0, 0),
    Gene("basis_right_poly4", 0, 0),
    Gene("basis_right_poly5", 0, 0),
    Gene("basis_right_sinamp1", -1.0, 1.0),
    Gene("basis_right_sinfreq1", -4.0, 4.0),
    Gene("basis_right_sinamp2", -1.0, 1.0),
    Gene("basis_right_sinfreq2", -4.0, 4.0),
    Gene("basis_right_sawtooth", 1000000, 3000000), # TODO unused! Discountinuities mess with the dynamics - no sawtooth allowed!
    Gene("basis_right_expamp1", 0, 0),
    Gene("basis_right_exppwr1", 0, 0),
    Gene("basis_right_expamp2", 0, 0),
    Gene("basis_right_exppwr2", 0, 0),
]

# Simulation parameters
SIMULATION_T = 10
DT = 0.0005

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