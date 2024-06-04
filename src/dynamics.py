import numpy as np
import globals

def get_aero_data(airfoil: str, alpha: float, AR: float, S:float, V_inf:float, rho_inf:float, creature):
    # inputs:
    # - airfoil: airfoil name
    # - alpha: angle of attack (rad)
    # - AR: aspect ratio
    # Outputs:
    # - L: lift (N)
    # - D: drag (N)

    [dcLdAlpha, alpha_min, alpha_max, cD0] = globals.AIRFOIL_DATABASE[airfoil][:-1]
    cL = dcLdAlpha * alpha

    e = 1.78 * (1 - 0.045 * AR**0.68) - 0.64
    cD = cD0 + cL**2 / (np.pi * e * AR)

    # if alpha < alpha_min or alpha > alpha_max:
    #     creature.has_stalled = True

    if alpha < alpha_min or alpha > alpha_max:
        cL = 0
        cD = np.abs(0.5 * np.sin(alpha)) + cD0

    L = 0.5*rho_inf*cL*S*V_inf**2
    D = 0.5*rho_inf*cD*S*V_inf**2

    return L, D

def get_bird2world(quats):
    q1, q2, q3, q0 = quats
    R_bird2world = np.array([[q0**2 + q1**2 - q2**2 - q3**2, 2*(q1*q2 - q0*q3), 2*(q0*q2 + q1*q3)],
                             [2*(q1*q2 + q0*q3), q0**2 - q1**2 + q2**2 - q3**2, 2*(q2*q3 - q0*q1)],
                             [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), q0**2 - q1**2 - q2**2 + q3**2]])
    return R_bird2world

def norm_state_quats(state):
    # quats in state[6:10]
    state[6:10] *= 1/np.linalg.norm(state[6:10])
    return state

def euler_step(t, state, virtual_creature, dt, verbose=False):
    """
    Takes a virtual creature and updates its state by one step based on
    it's current state and the aerodynamic environment

    dt units are in seconds
    """

    if verbose: print("t", t)

    # Get wing angle
    wa_left = virtual_creature.calc_wing_angle(t, "left")
    wa_right = virtual_creature.calc_wing_angle(t, "right")
    
    # +x is forward, +y is up, +z is right
    pos_world = state[0:3]
    vel_world = state[3:6]
    quats = state[6:10] # q1, q2, q3, q0
    pqr = state[10:13]

    euler_angles = np.degrees(globals.quaternion_to_euler(quats))
    if verbose: print("euler_angles (deg)", euler_angles)

    # Convert frome bird frame to world frame
    R_bird2world = get_bird2world(quats)
    
    uvw = R_bird2world.T @ vel_world
    
    # Enforce non-negative velocities
    if uvw[0] < 0: uvw[0] = 0

    if verbose: print("uvw", uvw)

    # Pull simulation constants
    rho_inf, rho_bird, g = globals.AIR_DENSITY, globals.BIRD_DENSITY, globals.GRAVITY

    airfoil_armwing_index = virtual_creature.chromosome.airfoil_armwing
    airfoil_handwing_index = virtual_creature.chromosome.airfoil_handwing
    #print(airfoil_armwing_index, airfoil_handwing_index)
    # Recall that dict keys are ordered as of Python 3.7
    airfoil_keys = list(globals.AIRFOIL_DATABASE.keys())
    airfoil_armwing  = airfoil_keys[airfoil_armwing_index]
    airfoil_handwing = airfoil_keys[airfoil_handwing_index]

    # Get wing parameters
    AR_aw, AR_hw, area_aw, area_hw = virtual_creature.AR_aw, virtual_creature.AR_hw, virtual_creature.area_aw, virtual_creature.area_hw
    COG_position, x_COL_position, z_COL_position = virtual_creature.COG_position, virtual_creature.x_COL_position, virtual_creature.z_COL_position
    bird_mass, I_mat = virtual_creature.bird_mass, virtual_creature.I_mat

    # Find angle of attack & V_inf
    V_t = np.linalg.norm(uvw)
    V_inf = np.linalg.norm([uvw[0], uvw[2]])
    alpha_left = wa_left + np.arctan2(uvw[2],uvw[0])
    alpha_right = wa_right + np.arctan2(uvw[2],uvw[0])
    beta = np.arcsin(uvw[1]/V_t)
    # beta = 0.0

    if verbose: print("alpha_left", alpha_left)
    if verbose: print("alpha_right", alpha_right)
    if verbose: print("beta", beta)

    # Get aerodynamic forces
    lift_aw_left, drag_aw_left = get_aero_data(airfoil=airfoil_armwing,
                                            alpha=alpha_left,
                                            AR=AR_aw,
                                            S=area_aw/2,
                                            V_inf=V_inf,
                                            rho_inf=rho_inf,
                                            creature=virtual_creature
                                            )
    lift_aw_right, drag_aw_right = get_aero_data(airfoil=airfoil_armwing,
                                                alpha=alpha_right,
                                                AR=AR_aw,
                                                S=area_aw/2,
                                                V_inf=V_inf,
                                                rho_inf=rho_inf,
                                                creature=virtual_creature
                                                )
    lift_hw_left, drag_hw_left = get_aero_data(airfoil=airfoil_handwing,
                                            alpha=alpha_left,
                                            AR=AR_hw,
                                            S=area_hw/2,
                                            V_inf=V_inf,
                                            rho_inf=rho_inf,
                                            creature=virtual_creature
                                            )
    lift_hw_right, drag_hw_right = get_aero_data(airfoil=airfoil_handwing,
                                                alpha=alpha_right,
                                                AR=AR_hw,
                                                S=area_hw/2,
                                                V_inf=V_inf,
                                                rho_inf=rho_inf,
                                                creature=virtual_creature
                                                )

    # Find forces in bird frame
    lift_left = lift_aw_left + lift_hw_left
    lift_right = lift_aw_right + lift_hw_right
    drag_left = drag_aw_left + drag_hw_left
    drag_right = drag_aw_right + drag_hw_right

    if verbose: print("lift_left", lift_left)
    if verbose: print("lift_right", lift_right)
    if verbose: print("drag_left", drag_left)
    if verbose: print("drag_right", drag_right)

    F_x_left = lift_left*np.sin(alpha_left) - drag_left*np.cos(beta)*np.cos(alpha_left)
    F_x_right = lift_right*np.sin(alpha_right) - drag_right*np.cos(beta)*np.cos(alpha_right)
    F_x = F_x_left + F_x_right
    F_y = (drag_left+drag_right) * np.sin(beta)
    F_z_left = -lift_left*np.cos(alpha_left) - drag_left*np.cos(beta)*np.sin(alpha_left)
    F_z_right = -lift_right*np.cos(alpha_right) - drag_right*np.cos(beta)*np.sin(alpha_right) 
    F_z = F_z_left + F_z_right
    F_vector = np.array([F_x, F_y, F_z])

    if verbose: print("F_vector", F_vector)

    # Find moments in bird frame
    M_x, M_y, M_z, = 0.0, 0.0, 0.0
    M_x = (F_z_right - F_z_left) * z_COL_position
    M_y = (-F_z) * (COG_position - x_COL_position)
    M_z = (F_x_left - F_x_right) * z_COL_position
    M_vector = np.array([M_x, M_y, M_z])
    # M_vector = np.array([0.0, 0.0, 0.0])
    g_vector = R_bird2world.T @ np.array([0, 0, g])
    
    # Find state in bird frame
    #print(uvw_dit )
    uvw_dot = 1/bird_mass * (-np.cross(pqr, uvw) + F_vector) + g_vector
    pqr_dot = np.linalg.inv(I_mat) @ (-(np.cross(pqr, (I_mat@pqr))) + M_vector).T
    # uvw_dot = 1/bird_mass * (F_vector) + g_vector
    # pqr_dot = np.linalg.inv(I_mat) @ (M_vector)

    if verbose: print("uvw_dot", uvw_dot)
    if verbose: print("pqr_dot", pqr_dot)

    # Update state in bird frame
    uvw += uvw_dot*dt
    pqr += pqr_dot*dt

    # Enforce non-negative velocities
    if uvw[0] < 0:
        uvw[0] = 0
        uvw_dot[0] = 0

    # Convert to world frame
    vel_world = R_bird2world @ uvw
    acc_world = R_bird2world @ uvw_dot

    # Kill sim if angular velocity is too high
    if np.linalg.norm(pqr) > globals.MAX_ANGULAR_VELOCITY:
        vel_world = np.array([0, 0, 20])
        acc_world = np.array([0, 0, g])

    if verbose: print("vel_world", vel_world)
    if verbose: print("acc_world", acc_world)
    #if verbose: print("pos change", np.linalg.norm(vel_world) * dt)
    if verbose: print("----")

    # Update quaternions
    q1, q2, q3, q0 = quats
    R_pqr2quats = 0.5 * np.array([[q0, -q3, q2],
                           [q3, q0, -q1],
                           [-q2, q1, q0],
                           [-q1, -q2, -q3]])
    quats_dot = R_pqr2quats @ pqr

    # print(f"M_x = {(F_z_right - F_z_left) * z_COL_position}, t = {t}, acceleration = {acc_world}, beta = {np.degrees(beta)}")

    # # Update world position/angles
    # pos_world += dt*vel_world
    # quats += dt*quats_dot

    # Get stateDot
    state_dot = np.concatenate([vel_world,
                               acc_world,
                               quats_dot,
                               pqr_dot])
    
    return state_dot

def forward_step(virtual_creature, t, dt):
    # Get current state [pos, vel, acceleration, quats, pqr]

    state = virtual_creature.get_dynamics_state_vector()

    #verbose = (t % (dt * 100) < 1e-5) and (t > 2.86)
    verbose = False

    k1 = euler_step(t,      state,                             virtual_creature, dt, verbose=verbose)
    k2 = euler_step(t+dt/2, norm_state_quats(state+(k1*dt/2)), virtual_creature, dt/2)
    k3 = euler_step(t+dt/2, norm_state_quats(state+(k2*dt/2)), virtual_creature, dt/2)
    k4 = euler_step(t+dt,   norm_state_quats(state+(k3*dt)),   virtual_creature, dt)

    state_dot = k1/6 + k2/3 + k3/3 + k4/6

    new_state = state_dot * dt + state

    # verbose = (t > 2.86 and t < 3) and t % (dt * 100) < 1e-5
    # if verbose: print("t", t)
    # if verbose: print("state", state)
    # if verbose: print("k1", k1)
    # if verbose: print("k2", k2)
    # if verbose: print("k3", k3)
    # if verbose: print("k4", k4)
    # if verbose: print("state_dot", state_dot)
    # if verbose: print("vel_measure", np.linalg.norm(new_state[3:6]) * dt)
    # if verbose: print("pos_measure", np.linalg.norm(state[0:3] - new_state[0:3]))

    # Extract state
    pos_world = new_state[0:3]
    vel_world = new_state[3:6]
    acc_world = state_dot[3:6]
    quats = new_state[6:10]
    pqr = new_state[10:13]

    quats *= 1/np.linalg.norm(quats)
    
    # Finish by updating the state of the virtual creature
    wing_angle_left  = virtual_creature.calc_wing_angle(t+dt, "left")
    wing_angle_right = virtual_creature.calc_wing_angle(t+dt, "right")
    
    virtual_creature.update_state(
        position_xyz=pos_world,
        velocity_xyz=vel_world,
        acceleration_xyz=acc_world,
        quaternions=quats,
        angular_velocity=pqr,
        wing_angle_left=wing_angle_left,
        wing_angle_right=wing_angle_right,
        euler_angles=np.degrees(globals.quaternion_to_euler(quats))
    )