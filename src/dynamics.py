import numpy as np
import globals

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

    [dcLdAlpha, alpha_min, alpha_max, cD0] = globals.AIRFOIL_DATABASE[airfoil][:-1]
    cL = dcLdAlpha * alpha

    if alpha < alpha_min or alpha > alpha_max:
        cL = 0

    e = 1.78 * (1 - 0.045 * AR**0.68) - 0.64
    cD = cD0 + cL**2 / (np.pi * e * AR)

    L = 0.5*rho_inf*cL*S*V_inf**2
    D = 0.5*rho_inf*cD*S*V_inf**2
    return L, D

def get_wing_parameters(virtual_creature):
    # Get chromosome data
    wingspan, norm_wrist_position, wing_root_chord = virtual_creature.chromosome.wingspan, virtual_creature.chromosome.norm_wrist_position, virtual_creature.chromosome.wing_root_chord
    taper_armwing, taper_handwing, norm_COG_position = virtual_creature.chromosome.taper_armwing, virtual_creature.chromosome.taper_handwing, virtual_creature.chromosome.norm_COG_position
    airfoil_armwing, airfoil_handwing = virtual_creature.chromosome.airfoil_armwing, virtual_creature.chromosome.airfoil_handwing
    bird_density = globals.BIRD_DENSITY
    COG_position = norm_COG_position * wing_root_chord

    airfoil_armwing, airfoil_handwing = "NACA 0012", "NACA 0012" #NOTE: This will need to be changed

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
    COL_position = 1/4 * wing_root_chord

    # Wing volume
    chord_thickness_aw = globals.AIRFOIL_DATABASE[airfoil_armwing][-1]
    chord_thickness_hw = globals.AIRFOIL_DATABASE[airfoil_handwing][-1]
    wing_volume_aw = chord_avg_aw**2 * span_aw * chord_thickness_aw
    wing_volume_hw = chord_avg_hw**2 * span_hw * chord_thickness_hw
    bird_volume = wing_volume_aw + wing_volume_hw
    bird_mass = bird_volume * bird_density

    # Moment of Inertia (simplified)
    span_avg = (area_aw + area_hw) / wing_root_chord
    Ix = bird_mass*(span_avg**2)/12 + bird_mass*(COG_position-0.5*wing_root_chord)**2
    Iy = bird_mass*(span_avg**2 + wing_root_chord**2)/12
    Iz = bird_mass*(wing_root_chord**2)/12

    return AR_aw, AR_hw, area_aw, area_hw, COG_position, COL_position, bird_mass, Ix, Iy, Iz

def forward_step(virtual_creature, dt=1.0):
    """
    Takes a virtual creature and updates its state by one step based on
    it's current state and the aerodynamic environment

    dt units are in seconds
    """
    
    # +x is forward, +y is up, +z is right
    p = virtual_creature.position_xyz
    v = virtual_creature.velocity_xyz
    a = virtual_creature.acceleration_xyz
    r = virtual_creature.rotation_xyz
    omega = virtual_creature.angular_velocity
    wa = virtual_creature.wing_angle

    # Pull chromosome data
    airfoil_armwing, airfoil_handwing = virtual_creature.chromosome.airfoil_armwing, virtual_creature.chromosome.airfoil_handwing
    AR_aw, AR_hw, area_aw, area_hw = virtual_creature.AR_aw, virtual_creature.AR_hw, virtual_creature.area_aw, virtual_creature.area_hw
    COG_position, COL_position, bird_mass = virtual_creature.COG_position, virtual_creature.COL_position, virtual_creature.bird_mass
    Ix, Iy, Iz = virtual_creature.Ix, virtual_creature.Iy, virtual_creature.Iz
    rho_inf, rho_bird, g = globals.AIR_DENSITY, globals.BIRD_DENSITY, globals.GRAVITY

    airfoil_armwing, airfoil_handwing = "NACA 0012", "NACA 0012" #NOTE: This will need to be changed

    # Find angle of attack & V_inf
    # angle_of_attack = globals.wrapRads(wa + r[2])
    velocity_angle = globals.wrapRads(np.arctan2(-v[1],v[0]))
    angle_of_attack = wa + r[2] + velocity_angle
    V_inf = np.linalg.norm(v[0:1])

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
    # NOTE: convert to matrices later
    lift = lift_aw + lift_hw
    lift_x = - lift * np.sin(r[2] + wa)
    lift_y = lift * np.cos(r[2] + wa)
    drag = drag_aw + drag_hw
    drag_x = -drag * np.cos(r[2] + wa)
    drag_y = -drag * np.sin(r[2] + wa)
    weight = - bird_mass * g

    # Stepper
    v_dot = np.array([0.0, 0.0, 0.0])
    v_dot[1] = (lift_y + drag_y + weight) / bird_mass
    v_dot[0] = (drag_x + lift_x) / bird_mass
    v = v + v_dot * dt

    p = p + dt*v

    omega_dot = np.array([0, 0, 0])
    # Encounter an issue here where python int too large to convert to C long,
    # and it's because of the lift/Iz term. Iz is constant, but lift
    # is becoming too large
    omega_dot[2] = lift/Iz * (COG_position - COL_position)
    omega = omega + omega_dot*dt

    r = r + dt*omega

    # Wrap angles to [-pi, pi]
    # r = globals.wrapRads(r)
    # wa = globals.wrapRads(wa)


    # Finish by updating the state of the virtual creature
    virtual_creature.update_state(
        position_xyz=p,
        velocity_xyz=v,
        acceleration_xyz=v_dot,
        rotation_xyz=r,
        angular_velocity=omega,
        wing_angle=wa,
    )