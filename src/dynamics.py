import numpy as np
import scipy as sci
import globals

# TODO: Kusal to implement how the state evolves at every step 
# starting with a single virtual creature in a simple aerodynamic
# environment/model

def get_aero_data(airfoil: str, alpha: float, AR: float, S:float, V_inf:float, rho_inf:float) -> float float, float, float:
    # inputs:
    # - airfoil: airfoilf name
    # - alpha: angle of attack (deg)
    # - AR: aspect ratio
    # Outputs:
    # - L: lift (N)
    # - D: drag (N)

    # airfoil_database parameters:
    # 1) dcL/dAlpha (1/rad)
    # 2) angle of min stall (rad)
    # 3) angle of max stall (rad)
    # 4) cD0

    airfoil_database = {
        "NACA 0012": [0.00204, -0.122173, 0.122173, 5.72958]
    }
    [dcLdAlpha, alpha_min, alpha_max, cD0] = airfoil_database[airfoil]
    cL = dcLdAlpha * alpha

    if alpha < alpha_min or alpha > alpha_max:
        cL = 0

    e = 1.78 * (1 - 0.045 * AR^0.68) - 0.64
    cD = cD0 + cL**2 / (np.pi * e * AR)

    L = 0.5*rho_inf*cL*S*V_inf**2
    D = 0.5*rho_inf*cD*S*V_inf**2
    return L, D

def get_wing_parameters(virtual_creature):
    # Get chromosome data
    wingspan, norm_wrist_position, wing_root_chord = virtual_creature.chromosome.wingspan, virtual_creature.chromosome.norm_wrist_position, virtual_creature.chromosome.wing_root_chord
    taper_armwing, taper_handwing, COG_position = virtual_creature.chromosome.taper_armwing, virtual_creature.chromosome.taper_handwing, virtual_creature.chromosome.COG_position
    airfoil_armwing, airfoil_handwing = virtual_creature.chromosome.airfoil_armwing, virtual_creature.chromosome.airfoil_handwing

    # Wing characteristics
    span_aw = wingspan * norm_wrist_position
    span_hw = wingspan - span_aw
    area = lambda b, cr, gamma: (1+gamma)/2*cr*b
    area_aw = area(span_aw, wing_root_chord, taper_armwing)
    area_hw = area(span_hw, wing_root_chord*taper_armwing, taper_handwing)
    AR_aw = span_aw / area_aw
    AR_hw = span_hw / area_hw

    # Moment of inertia

    return AR_aw, AR_hw, area_aw, area_hw, airfoil_armwing, airfoil_handwing



def forward_step(virtual_creature):
    """
    Takes a virtual creature and updates its state by one step based on
    it's current state and the aerodynamic environment
    """
    # +x is forward, +y is right, +z is up
    p, v, a, r, wa = virtual_creature.position_xyz, virtual_creature.velocity_xyz, virtual_creature.acceleration_xyz, virtual_creature.rotation_xyz, virtual_creature.wing_angle

    # TODO: Do all the funky physics here and compute new state
    # Can access the genes of the chromosome like this:
    # Pull chromosome data
    rho_inf, g = globals.air_density, globals.gravity_acceleration

    # Get wing parameters
    AR_aw, AR_hw, area_aw, area_hw, airfoil_armwing, airfoil_handwing = get_wing_parameters(virtual_creature)

    # Find angle of attack & V_inf
    angle_of_attack = wa + r[1]
    V_inf = v[0] * np.cos(angle_of_attack) + v[1] * np.cos(angle_of_attack)


    lift_aw, drag_aw = get_aero_data(airfoil=airfoil_armwing,
                                 alpha=angle_of_attack,
                                 AR=AR_aw,
                                 S=area_aw,
                                 V_inf=V_inf,
                                 rho_inf=rho_inf
                                 )
    
    lift_hw, drag_hw = get_aero_data(airfoil=airfoil_handwing,
                                 alpha=angle_of_attack,
                                 AR=AR_hw,
                                 S=area_hw,
                                 V_inf=V_inf,
                                 rho_inf=rho_inf)
    
    lift = lift_aw + lift_hw
    drag = lift_aw + lift_hw


    p = p + 0.1 * np.array([1, 0, 0])  # Move forward 0.1m
    # etc.

    # Finish by updating the state of the virtual creature
    virtual_creature.update_state(
        position_xyz=p,
        velocity_xyz=v,
        acceleration_xyz=a,
        rotation_xyz=r,
    )