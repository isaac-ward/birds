import numpy as np
import scipy as sci
import globals

# TODO: Kusal to implement how the state evolves at every step 
# starting with a single virtual creature in a simple aerodynamic
# environment/model

def get_aero_data(airfoil: str, alpha: float, AR: float, S:float, V_inf:float, rho_inf:float):
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

def get_airfoil_thickness(airfoil:str) -> float:
    airfoil_database = {
        "NACA 0012": 0.12
    }
    return airfoil_database[airfoil]

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

    chord_avg_aw = (1+taper_armwing)/2 * wing_root_chord
    chord_avg_hw = (1+taper_handwing)/2 * taper_armwing * wing_root_chord
    chord_avg = 1/2 * (norm_wrist_position*chord_avg_aw + (1-norm_wrist_position)*chord_avg_hw)
    COL_position = wing_root_chord - 3/4 * chord_avg

    # Wing volume
    chord_thickness_aw = get_airfoil_thickness(airfoil_armwing)
    chord_thickness_hw = get_airfoil_thickness(airfoil_handwing)
    wing_volume_aw = chord_avg_aw**2 * span_aw * chord_avg_aw
    wing_volume_hw = chord_avg_hw**2 * span_hw * chord_avg_hw
    bird_volume = wing_volume_aw + wing_volume_hw

    # Moment of inertia
    Ix = 1
    Iy = 1
    Iz = 0

    return AR_aw, AR_hw, area_aw, area_hw, COG_position, COL_position, bird_volume, Ix, Iy, Iz


def forward_step(virtual_creature, dt):
    """
    Takes a virtual creature and updates its state by one step based on
    it's current state and the aerodynamic environment
    """
    
    # +x is forward, +y is right, +z is up
    p, v, a, r, wa, omega = virtual_creature.position_xyz, virtual_creature.velocity_xyz, virtual_creature.rotation_xyz, virtual_creature.wing_angle, virtual_creature.angular_velocity

    # TODO: Do all the funky physics here and compute new state
    # Can access the genes of the chromosome like this:
    # Pull chromosome data
    airfoil_armwing, airfoil_handwing = virtual_creature.chromosome.airfoil_armwing, virtual_creature.chromosome.airfoil_handwing
    rho_inf, rho_bird, g = globals.air_density, globals.bird_density, globals.gravity_acceleration

    # Get wing parameters
    AR_aw, AR_hw, area_aw, area_hw, COG_position, COL_position, bird_volume, Ix, Iy, Iz = get_wing_parameters(virtual_creature)

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
    
    # Calculate key forces
    bird_mass = rho_bird * bird_volume
    lift = lift_aw + lift_hw
    drag = drag_aw + drag_hw
    weight = bird_mass * g

    # Stepper
    p = p + dt*v

    v_dot = np.array([0, 0, 0])
    v_dot[1] = (lift - weight) / bird_mass
    v_dot[0] = - drag / bird_mass
    v = v + v_dot * dt

    r = r + dt*omega

    omega_dot = np.array([0, 0, 0])
    omega_dot[2] = lift/Iz * (COG_position - COL_position)
    omega = omega + omega_dot*dt

    # Finish by updating the state of the virtual creature
    virtual_creature.update_state(
        position_xyz=p,
        velocity_xyz=v,
        acceleration_xyz=a,
        rotation_xyz=r,
        angular_velocity=omega,
        wing_angle=wa,
        
    )