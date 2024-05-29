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

    [dcLdAlpha, alpha_min, alpha_max, cD0] = globals.AIRFOIL_DATABASE[airfoil][:-1]
    cL = dcLdAlpha * alpha

    if alpha < alpha_min or alpha > alpha_max:
        cL = 0

    e = 1.78 * (1 - 0.045 * AR**0.68) - 0.64
    cD = cD0 + cL**2 / (np.pi * e * AR)

    L = 0.5*rho_inf*cL*S*V_inf**2
    D = 0.5*rho_inf*cD*S*V_inf**2
    return L, D

def get_bird2world(quats):
    q1, q2, q3, q0 = quats
    R_bird2world = np.array([[q0**2 + q1**2 - q2**2 - q3**2, 2*(q1*q2 - q0*q3), 2*(q0*q2 + q1*q3)],
                             [2*(q1*q2 + q0*q3), q0**2 - q1**2 + q2**2 - q3**2, 2*(q2*q3 - q0*q1)],
                             [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), q0**2 - q1**2 - q2**2 + q3**2]])
    return R_bird2world


def forward_step(virtual_creature, t, dt=1.0):
    """
    Takes a virtual creature and updates its state by one step based on
    it's current state and the aerodynamic environment

    dt units are in seconds
    """

    # Get wing angle
    wa_left = virtual_creature.calc_wing_angle(t, "left")
    wa_right = virtual_creature.calc_wing_angle(t, "right")
    
    # +x is forward, +y is up, +z is right
    pos_world = virtual_creature.position_xyz
    vel_world = virtual_creature.velocity_xyz
    acc_world = virtual_creature.acceleration_xyz
    quats = virtual_creature.quaternions
    pqr = virtual_creature.angular_velocity

    # Convert frome bird frame to world frame
    R_bird2world = get_bird2world(quats)
    
    uvw = R_bird2world.T @ vel_world

    # Pull chromosome data
    airfoil_armwing, airfoil_handwing = virtual_creature.chromosome.airfoil_armwing, virtual_creature.chromosome.airfoil_handwing
    rho_inf, rho_bird, g = globals.AIR_DENSITY, globals.BIRD_DENSITY, globals.GRAVITY

    airfoil_armwing, airfoil_handwing = "NACA 0012", "NACA 0012" #NOTE: This will need to be changed

    # Get wing parameters
    AR_aw, AR_hw, area_aw, area_hw = virtual_creature.AR_aw, virtual_creature.AR_hw, virtual_creature.area_aw, virtual_creature.area_hw
    COG_position, x_COL_position, z_COL_position = virtual_creature.COG_position, virtual_creature.x_COL_position, virtual_creature.z_COL_position
    bird_mass, I_mat = virtual_creature.bird_mass, virtual_creature.I_mat

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
                                            rho_inf=rho_inf
                                            )
    lift_hw_right, drag_hw_right = get_aero_data(airfoil=airfoil_handwing,
                                                alpha=alpha_right,
                                                AR=AR_hw,
                                                S=area_hw/2,
                                                V_inf=V_inf,
                                                rho_inf=rho_inf
                                                )

    # Find forces in bird frame
    lift_left = lift_aw_left + lift_hw_left
    lift_right = lift_aw_right + lift_hw_right
    drag_left = drag_aw_left + drag_hw_left
    drag_right = drag_aw_right + drag_hw_right

    F_x_left = lift_left*np.sin(alpha_left) - drag_left*np.cos(beta)*np.cos(alpha_left)
    F_x_right = lift_right*np.sin(alpha_right) - drag_right*np.cos(beta)*np.cos(alpha_right)
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
    g_vector = R_bird2world.T @ np.array([0, 0, g])
    
    # Find state in bird frame
    uvw_dot = 1/bird_mass * (-np.cross(pqr, uvw) + F_vector) + g_vector
    pqr_dot = np.linalg.inv(I_mat) @ (-(np.cross(pqr, (I_mat@pqr))) + M_vector).T

    # Update state in bird frame
    uvw += uvw_dot*dt
    pqr += pqr_dot*dt

    # Convert to world frame
    vel_world = R_bird2world @ uvw
    acc_world = R_bird2world @ uvw_dot

    # Update quaternions
    q1, q2, q3, q0 = quats
    R_pqr2quats = np.array([[q0, -q3, q2],
                           [q3, q0, -q1],
                           [-q2, q1, q0],
                           [-q1, -q2, -q3]])
    quats_dot = R_pqr2quats @ pqr

    # Update world position/angles
    pos_world += dt*vel_world
    quats += dt*quats_dot
    quats *= 1/np.linalg.norm(quats)

    # Finish by updating the state of the virtual creature
    virtual_creature.update_state(
        position_xyz=pos_world,
        velocity_xyz=vel_world,
        acceleration_xyz=acc_world,
        quaternions=quats,
        angular_velocity=pqr
    )

    return t + dt