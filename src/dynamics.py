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

    airfoil_database = {
        "NACA 0012": [5.72958, -0.122173, 0.122173, 0.021]
    }
    [dcLdAlpha, alpha_min, alpha_max, cD0] = airfoil_database[airfoil]
    cL = dcLdAlpha * alpha

    if alpha < alpha_min or alpha > alpha_max:
        cL = 0

    e = 1.78 * (1 - 0.045 * AR**0.68) - 0.64
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

    # Center of lift
    chord_avg_aw = (1+taper_armwing)/2 * wing_root_chord
    chord_avg_hw = (1+taper_handwing)/2 * taper_armwing * wing_root_chord
    chord_avg = (norm_wrist_position*chord_avg_aw + (1-norm_wrist_position)*chord_avg_hw)
    # x_COL_position = wing_root_chord - 3/4 * chord_avg
    x_COL_position = 1/4 * wing_root_chord

    z_COL_position = wingspan/4

    # Wing volume
    chord_thickness_aw = get_airfoil_thickness(airfoil_armwing)
    chord_thickness_hw = get_airfoil_thickness(airfoil_handwing)
    wing_volume_aw = chord_avg_aw**2 * span_aw * chord_thickness_aw
    wing_volume_hw = chord_avg_hw**2 * span_hw * chord_thickness_hw
    bird_volume = wing_volume_aw + wing_volume_hw
    bird_mass = bird_volume * bird_density

    # # Moment of inertia
    # c1 = taper_armwing * taper_handwing * wing_root_chord
    # c2 = (1-taper_handwing) * taper_armwing * wing_root_chord
    # c3 = (1-taper_armwing) * wing_root_chord
    # b1 = wingspan
    # b2 = (1+norm_wrist_position)/2 * wingspan
    # b3 = norm_wrist_position/2 * wingspan
    # m1 = c1*b1/(c1*b1+c2*b2+c3*b3)*bird_mass
    # m2 = c2*b2/(c1*b1+c2*b2+c3*b3)*bird_mass
    # m3 = c3*b3/(c1*b1+c2*b2+c3*b3)*bird_mass
    # x_COG1 = c3 + c2 + c1/2
    # x_COG2 = c3 + c2/2
    # x_COG3 = c3/2
    # Ix_COG1 = m1*b1**2/12
    # Iy1 = m1*(b1**2 + c1**2)/12
    # Iz1 = m1*c1**2/12
    # Ix_COG2 = m2*b2**2/12
    # Iy2 = m2*(b2**2 + c2**2)/12
    # Iz2 = m2*c2**2/12
    # Ix_COG3 = m3*b3**2/12
    # Iy3 = m3*(b3**2 + c3**2)/12
    # Iz3 = m3*c3**2/12
    # Ix = Ix_COG1 + m1*(COG_position-x_COG1)**2 + Ix_COG2 + m2*(COG_position-x_COG2)**2 + Ix_COG3 + m3*(COG_position-x_COG3)**2
    # Iy = Iy1 + Iy2 + Iy3
    # Iz = Iz1 + Iz2 + Iz3

    # Moment of Inertia (simplified)
    span_avg = (area_aw + area_hw) / wing_root_chord
    Ix = bird_mass*(span_avg**2)/12 + bird_mass*(COG_position-0.5*wing_root_chord)**2
    Iz = bird_mass*(span_avg**2 + wing_root_chord**2)/12
    Iy = bird_mass*(wing_root_chord**2)/12
    I_mat = np.diag([Ix, Iy, Iz])

    return AR_aw, AR_hw, area_aw, area_hw, COG_position, x_COL_position, z_COL_position, bird_mass, I_mat

def get_bird2world(phi, theta, psi):
    R1 = np.array([[1, 0, 0],
                   [0, np.cos(phi), np.sin(phi)],
                   [0, np.sin(phi), np.cos(phi)]])
    R2 = np.array([[np.cos(theta), 0, -np.sin(theta)],
                   [0, 1, 0],
                   [np.sin(theta), 0, np.cos(theta)]])
    R3 = np.array([[np.cos(psi), np.sin(psi), 0],
                   [-np.sin(psi), np.cos(psi), 0],
                   [0, 0, 1]])
    R4 = np.array([[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, -1]])
    R_bird2world = R1 @ R2 @ R3 @ R4

    return R_bird2world


def forward_step(virtual_creature, dt=1.0):
    """
    Takes a virtual creature and updates its state by one step based on
    it's current state and the aerodynamic environment

    dt units are in seconds
    """
    
    # +x is forward, +y is up, +z is right
    pos_world = virtual_creature.position_xyz
    vel_world = virtual_creature.velocity_xyz
    acc_world = virtual_creature.acceleration_xyz
    rot_world = virtual_creature.rotation_xyz
    omega_world = virtual_creature.angular_velocity
    wa_left = virtual_creature.wing_angle_left
    wa_right = virtual_creature.wing_angle_right

    phi, theta, psi = rot_world

    # Convert frome bird frame to world frame
    R_bird2world = get_bird2world(phi, theta, psi)
    
    uvw = R_bird2world.T @ vel_world

    R_pqr2euler = np.array([[1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
                            [0, np.cos(phi), -np.sin(phi)],
                            [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]])
    pqr = R_pqr2euler.T @ omega_world

    # Pull chromosome data
    airfoil_armwing, airfoil_handwing = virtual_creature.chromosome.airfoil_armwing, virtual_creature.chromosome.airfoil_handwing
    rho_inf, rho_bird, g = globals.AIR_DENSITY, globals.BIRD_DENSITY, globals.GRAVITY

    airfoil_armwing, airfoil_handwing = "NACA 0012", "NACA 0012" #NOTE: This will need to be changed

    # Get wing parameters
    AR_aw, AR_hw, area_aw, area_hw, COG_position, x_COL_position, z_COL_position, bird_mass, I_mat = get_wing_parameters(virtual_creature)

    # Find angle of attack & V_inf
    V_inf = np.linalg.norm(uvw)
    alpha_left = wa_left + np.arctan2(uvw[2],uvw[0])
    alpha_right = wa_right + np.arctan2(uvw[2],uvw[0])
    beta = np.arcsin(uvw[1]/V_inf)

    # Get aeroydenmic forces
    lift_aw_left, drag_aw_left = get_aero_data(airfoil=airfoil_armwing,
                                               alpha=alpha_left,
                                               AR=AR_aw,
                                               S=area_aw/2,
                                               V_inf=V_inf,
                                               rho_inf=rho_inf
                                               )
    lift_aw_right, drag_aw_right = get_aero_data(airfoil=airfoil_armwing,
                                                 alpha=alpha_right,
                                                 AR=AR_aw,
                                                 S=area_aw/2,
                                                 V_inf=V_inf,
                                                 rho_inf=rho_inf
                                                 )
    lift_hw_left, drag_hw_left = get_aero_data(airfoil=airfoil_handwing,
                                               alpha=alpha_left,
                                               AR=AR_hw,
                                               S=area_hw/2,
                                               V_inf=V_inf,
                                               rho_inf=rho_inf)
    lift_hw_right, drag_hw_right = get_aero_data(airfoil=airfoil_handwing,
                                                 alpha=alpha_right,
                                                 AR=AR_hw,
                                                 S=area_hw/2,
                                                 V_inf=V_inf,
                                                 rho_inf=rho_inf)

    # Find forces in bird frame
    lift_left = lift_aw_left + lift_hw_left
    lift_right = lift_aw_right + lift_hw_right
    drag_left = drag_aw_left + drag_hw_left
    drag_right = drag_aw_right + drag_hw_right

    F_x_left = lift_left*np.sin(alpha_left) - drag_left*np.cos(beta)*np.sin(alpha_left)
    F_x_right = lift_right*np.sin(alpha_right) - drag_right*np.cos(beta)*np.sin(alpha_right)
    F_x = F_x_left + F_x_right
    F_y = (drag_left+drag_right) * np.sin(beta)
    F_z_left = -lift_left*np.cos(alpha_left) - drag_left*np.cos(beta)*np.sin(alpha_left)
    F_z_right = -lift_right*np.cos(alpha_right) - drag_right*np.cos(beta)*np.sin(alpha_right) 
    F_z = F_z_left + F_z_right
    F_vector = np.array([F_x, F_y, F_z])

    # Find moments in bird frame
    M_x = (F_z_right - F_z_left) * z_COL_position
    M_y = (-F_z) * (COG_position - x_COL_position)
    M_z = (F_x_left - F_x_right) * z_COL_position
    M_vector = np.array([M_x, M_y, M_z])
    g_vector = globals.GRAVITY * np.array([-np.sin(theta),
                                           np.cos(theta)*np.sin(phi),
                                           np.cos(theta)*np.cos(phi)])
    
    # Find state in bird frame
    uvw_dot = -1/bird_mass * np.cross(pqr, uvw) + F_vector + g_vector
    pqr_dot = np.linalg.inv(I_mat) @ (-(np.cross(pqr, (I_mat@pqr))) + M_vector).T

    # Update state in bird frame
    uvw += uvw_dot*dt
    pqr += pqr_dot*dt

    # Convert to world frame
    vel_world = R_bird2world @ uvw
    acc_world = R_bird2world @ uvw_dot

    # NOTE: Watch out for singularity
    omega_world = R_pqr2euler @ pqr

    # Update world position/angles
    pos_world += dt*vel_world
    rot_world += dt*omega_world

    for i in range(rot_world.size):
        rot_world[i] = globals.wrapRads(rot_world[i])

    # Finish by updating the state of the virtual creature
    virtual_creature.update_state(
        position_xyz=pos_world,
        velocity_xyz=vel_world,
        acceleration_xyz=acc_world,
        rotation_xyz=rot_world,
        angular_velocity=omega_world,
        wing_angle_left=wa_left,
        wing_angle_right=wa_right
    )