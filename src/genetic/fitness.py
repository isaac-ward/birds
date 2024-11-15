import numpy as np
from scipy.stats import rv_discrete
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

from dynamics import forward_step
import globals

def parallel_evaluate_fitness(virtual_creatures, test_mode=1, return_logging_data=True):
    """
    A parallelized way to evaluate the fitness of a list of virtual creatures
    """
    results = []

    # Log progress with a progress bar
    pbar = tqdm(total=len(virtual_creatures), desc="Evaluating fitness of virtual creature(s) (in parallel)")

    num_processes = min(len(virtual_creatures), os.cpu_count() - 4)
    num_processes = max(num_processes, 1)
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = {
            executor.submit(
                evaluate_fitness, 
                vc, 
                test_mode, 
                return_logging_data
            ): vc for vc in virtual_creatures
        }

        # Loop over the futures as they complete
        for future in as_completed(futures):
            vc = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Virtual creature {vc} generated an exception: {e}")
            pbar.update(1)

    return results

def evaluate_fitness(virtual_creature, test_mode=1, return_logging_data=True):

    # Ensure the creature's state is reset
    virtual_creature.reset_state()

    # Start with no fitness
    fitness = 0

    # We'll log all this
    state_trajectory = []
    fitness_components = {}

    def valid_state_check(state):
        # If any of the values are >> 1000, then
        # the simulation has failed 
        if np.any(np.abs(state) > 1000000):
            return False
            
        # If the creature goes too far down, then it's over
        if state[2] > 10:
            return False
        
        # # And too far up
        # if state[2] < 30:
        #     return False

        # And too far to the side
        if abs(state[1]) > 20:
            return False
        
        # If the creature spins too much, then it's over
        angular_velocity_limit = 5
        if np.any(np.abs(state[13:16]) > angular_velocity_limit):
            return False
        
        # If we get any nans, then it's over
        if np.any(np.isnan(state)):
            return False
        
        # If the creature pitches too far up, then it's over
        #euler_angles = state[18:21]
        if np.any(np.abs(state[18:21]) > 90):
            return False
        
        return True

    if test_mode == 1:

        # Rollout a simulation
        # Simulation parameters
        t = 0
        num_steps = int(globals.SIMULATION_T / globals.DT)
        for i in tqdm(range(num_steps), desc="Evaluating virtual creature's fitness", leave=False, disable=True):
            # Run the dynamics forward
            forward_step(virtual_creature, t, dt=globals.DT)
            t += globals.DT

            # Get the state vector and log
            state_vector = virtual_creature.get_state_vector()

            # If we reach some certain termination conditions, then assign fitnesses
            # and break out of the loop
            if not valid_state_check(state_vector):
                fitness = globals.FITNESS_PENALTY_INVALID_STATE
                #fitness_components["planar_distance_travelled"] = globals.FITNESS_PENALTY_INVALID_STATE
                fitness_components["last_x_position"] = globals.FITNESS_PENALTY_INVALID_STATE
                fitness_components["z_mean"] = globals.FITNESS_PENALTY_INVALID_STATE
                fitness_components["y_mean"] = globals.FITNESS_PENALTY_INVALID_STATE
                fitness_components["angular_divergence"] = globals.FITNESS_PENALTY_INVALID_STATE
                # #fitness_components["average_lateral_divergence"] = globals.FITNESS_PENALTY_INVALID_STATE
                #fitness_components["max_angular_divergence"] = globals.FITNESS_PENALTY_INVALID_STATE
                #fitness_components["max_acceleration"] = globals.FITNESS_PENALTY_INVALID_STATE
                fitness_components["penalty_invalid_state"] = globals.FITNESS_PENALTY_INVALID_STATE

                #print(f"Simulation went unstable at step={i}, time={t}")
                break

            # Otherwise log it
            state_trajectory.append(state_vector)

        # Now evaluate the fitness based on the trajectory
        # A fitter creature will have:
        # - gone forward more in the +x direction
        # - gone down less in the +z direction
        # - goes straight

        # This is basically just a glide ratio
        # This should be positive
        #forward_position  = state_trajectory[-1][0] - state_trajectory[0][0]
        # This should be positive
        z_mean = np.mean([state[2] for state in state_trajectory])

        # We want the creature to move in the xy plane maximally, so tally
        # up the distance travelled in the xy plane over all steps
        y_mean = np.mean([state[1] for state in state_trajectory])

        # Incentivize the creature to move as far in the x-direction as possible
        last_x_position = state_trajectory[-1][0]

        # Penalize for going off course (in the y direction)
        # This will be >= 0. Remeber to abs
        #average_lateral_divergence = np.mean(np.abs([state[1] for state in state_trajectory]))

        # Penalize for going beyond +- 90 degrees pitch
        #qx, qy, qz, qw = np.array([state[9:13] for state in state_trajectory])
        #euler_angles = 

        # Penalize for spinning too fast
        # 13, 14, 15 are angular 
        angular_divergence = \
            np.mean(np.abs([state[13] for state in state_trajectory])) + \
            np.mean(np.abs([state[14] for state in state_trajectory])) + \
            np.mean(np.abs([state[15] for state in state_trajectory]))
        
        # # Penalize for accelerating too fast
        # max_acceleration = \
        #     np.mean(np.abs([state[6] for state in state_trajectory])) + \
        #     np.mean(np.abs([state[7] for state in state_trajectory])) + \
        #     np.mean(np.abs([state[8] for state in state_trajectory]))

        # Fitness is a mix of these, accounting for positive/negative. Note that
        # we are trying to maximize this!
        # fitness = 10 * planar_distance_travelled - 1000 * downwards_position - 800 * max_angular_divergence - 1000 * max_acceleration
        # fitness = 10 * planar_distance_travelled - 500 * downwards_position - 400 * max_angular_divergence
        fitness = 60 * last_x_position - 40 * z_mean - 200 * y_mean - 200 * angular_divergence

        # Store the fitness components
        #fitness_components["planar_distance_travelled"] = planar_distance_travelled
        fitness_components["last_x_position"] = last_x_position
        fitness_components["z_mean"] = z_mean
        fitness_components["y_mean"] = y_mean
        fitness_components["angular_divergence"] = angular_divergence
        # #fitness_components["average_lateral_divergence"] = average_lateral_divergence
        #fitness_components["max_angular_divergence"] = max_angular_divergence
        #fitness_components["max_acceleration"] = max_acceleration
        fitness_components["penalty_invalid_state"] = 0

    elif test_mode == 2:
        # Define the waypoints as a list of tuples (x, y, z)
        waypoints = [
            (1.0, 2.0, 0.0),
            (3.0, 5.0, 3.0),
            (6.0, 8.0, 5.0),
            (9.0, 12.0, 10.0)
        ]

        # Waypoint finding fitness test
        total_waypoint_distance = 0

        for waypoint in waypoints:
            distance_to_waypoint = np.linalg.norm(virtual_creature.position_xyz[:2] - waypoint[:2])
            total_waypoint_distance += distance_to_waypoint

            # Check if the creature reaches the waypoint (within a threshold)
            if distance_to_waypoint < globals.WAYPOINT_THRESHOLD:
                fitness += globals.WAYPOINT_REWARD
            else:
                fitness -= globals.WAYPOINT_PENALTY
        
        # Penalize for the total distance to all waypoints
        fitness -= total_waypoint_distance

        # Don't go too far down
        fitness -= virtual_creature.position_xyz[2]

        # If you go way too far down you're dead
        if virtual_creature.position_xyz[2] < globals.TOO_LOW_Z:
            fitness = globals.FITNESS_PENALTY_TOO_LOW
    
    else:
        raise ValueError(f"Unsupported fitness test mode '{test_mode}'")
    
    # We may want additional logging data, in
    # which case we return more than just the fitness
    if return_logging_data:
        return fitness, fitness_components, state_trajectory
    else:
        return fitness

def select_fittest_individuals(population, fitness_scores, num_parents, method):
    """
    population - list of individual VirtualCreature objects describing the population
    fitness_scores - list of floats describing the fitness of each corresponding VirtualCreature in the population
    num_parents - integer describing the number of parents to select from the list 
    returns - list of VirtualCreature objects that are the selected parents
    """    
    
    #task : return VirtualCreature objects that are the selected parents (type: Object)
    #input : population of VirtualCreature objects describing the population (type: Object),
    # number of parents to select from the list (type: int),
    # fitness score for each VirtualCreature object in population (type: floats)
    if method=="truncation":
        # Pair each individual with its fitness score and original index
        paired_population = list(enumerate(zip(population, fitness_scores)))
        
        # Sort the paired population based on the fitness scores in descending order
        sorted_population = sorted(paired_population, key=lambda x: x[1][1], reverse=True)
        
        # Select the top num_parents individuals and their indices
        selected_parents_with_indices = sorted_population[:num_parents]
        parents        = [individual for idx, (individual, score) in selected_parents_with_indices]
        parent_indices = [idx        for idx, (individual, score) in selected_parents_with_indices]  

        # sorted_indices = np.argsort(fitness_scores)[::-1] # Highest to lowest
        # top_parents = sorted_indices[:num_parents] # The highest fitness scores
        # #parents = [population[np.random.choice(top_parents)] for _ in range(num_parents)] # The best of those
        # parent_indices = np.random.choice(top_parents, num_parents, replace=False)
        # parents = [population[i] for i in parent_indices]   

    elif method=="tournament":
        subset_indices = np.random.choice(len(fitness_scores),num_parents,replace=False)
        best_index = subset_indices[np.argmin(fitness_scores[subset_indices])]
        parents.append(population[best_index])
        parent_indices = best_index
    elif method =="roulette":
        adjusted_fitness = np.max(fitness_scores) - fitness_scores
        probabilities = adjusted_fitness/ np.sum(adjusted_fitness)
        distribution = rv_discrete(values=(np.arange(len(fitness_scores)), probabilities))
        parent_indices = distribution.rvs(size=num_parents)# if want roulette from population len(population))
        parents = [population[i] for i in parent_indices]
    else:
        raise ValueError(f"Unsupported selection method '{method}'")
    return parents, parent_indices