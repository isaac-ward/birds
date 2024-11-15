import numpy as np

from genetic.chromosome import Gene
from genetic.chromosome import GeneDiscrete

# Define the order of the genes in the chromosome
# and their feasible ranges
CHROMOSOME_DEFINITION = [
    # Gene(gene_name, min_val, max_val)
    Gene("wingspan", 5, 10),
    Gene("norm_wrist_position", 0.2, 0.8),
    Gene("wing_root_chord", 1.0, 1.2),
    Gene("taper_armwing", 0.5, 1.0),
    Gene("taper_handwing", 0.1, 0.5),
    Gene("norm_COG_position", 0.248, 0.252), # 0.25 is ideal
    # These are indices into a list of airfoil names
    GeneDiscrete("airfoil_armwing", 0, 0), # 2
    GeneDiscrete("airfoil_handwing", 0, 0), # 2
    # Control parameters
    # Constant parameters
    Gene("basis_left_const", -0.05, 0.05),
    Gene("basis_left_poly1", -0.01, 0.01),
    Gene("basis_left_poly2", -0.01, 0.01),
    Gene("basis_left_poly3", -0.01, 0.01),
    Gene("basis_left_poly4", 0, 0),
    Gene("basis_left_poly5", 0, 0),
    # 2 sinusoids
    Gene("basis_left_sinamp1", -0.05, 0.05),
    Gene("basis_left_sinfreq1", -0.01, 0.01),
    Gene("basis_left_sinamp2", 0, 0),
    Gene("basis_left_sinfreq2", 0, 0),
    # And same for the right
    Gene("basis_right_const", -0.05, 0.05),
    Gene("basis_right_poly1", -0.01, 0.01),
    Gene("basis_right_poly2", -0.01, 0.01),
    Gene("basis_right_poly3", -0.01, 0.01),
    Gene("basis_right_poly4", 0, 0),
    Gene("basis_right_poly5", 0, 0),
    Gene("basis_right_sinamp1", -0.05, 0.05),
    Gene("basis_right_sinfreq1", -0.01, 0.01),
    Gene("basis_right_sinamp2", 0, 0),
    Gene("basis_right_sinfreq2", 0, 0),
]

# Simulation parameters
SIMULATION_T = 7 # may be cut short if unstable
DT = 0.001

# Global aerodynamic parameters
AIR_DENSITY = 1.225 #kg/m^3
GRAVITY = 9.81      #m/s^2
BIRD_DENSITY = 10   #kg/m^3
MAX_ANGULAR_VELOCITY = 7 #rad/s

# Fitness Constants
FITNESS_PENALTY_INVALID_STATE = -1e4
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
    "NACA 0012": [5.72958, -0.122173, 0.122173, 0.021, 0.12],
    "NACA 1408": [6.58901, -0.139626, 0.122173, 0.016, 0.08],
    "NACA 2412": [6.49352, -0.122173, 0.174533, 0.022, 0.12]
}

# Global functions
def wrapRads(angle:float) -> float:
    wrapped_angle = (angle + np.pi) % (2 * np.pi) - np.pi
    return wrapped_angle

def quaternion_to_euler(quat):
    """
    Convert a quaternion into Euler angles (roll, pitch, yaw).
    Quaternion format: [q1, q2, q3, q0] where q0 is the scalar part.
    """
    q1, q2, q3, q0 = quat

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (q0 * q1 + q2 * q3)
    cosr_cosp = 1 - 2 * (q1 * q1 + q2 * q2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (q0 * q2 - q3 * q1)
    if np.abs(sinp) >= 1:
        pitch = np.sign(sinp) * np.pi / 2  # Use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (q0 * q3 + q1 * q2)
    cosy_cosp = 1 - 2 * (q2 * q2 + q3 * q3)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])