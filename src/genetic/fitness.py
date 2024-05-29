import numpy as np
from scipy.stats import rv_discrete
from tqdm import tqdm

from dynamics import forward_step
import globals


def evaluate_fitness(virtual_creature, test_mode=1, return_logging_data=True):

    # Ensure the creature's state is reset
    virtual_creature.reset_state()

    # Start with no fitness
    fitness = 0

    # We'll log all this
    state_trajectory = []

    # forward_step(virtual_creature)
    if test_mode == 1:

        # Rollout a simulation
        # Simulation parameters
        simulation_time_seconds = 7.5
        dt = 0.05
        t = 0
        num_steps = int(simulation_time_seconds / dt)
        for i in tqdm(range(num_steps), desc="Evaluating virtual creature's fitness", leave=False):
            # Run the dynamics forward
            t = forward_step(virtual_creature, t, dt=dt)

            # Get the state vector and log
            state_vector = virtual_creature.get_state_vector(t)
            state_trajectory.append(state_vector)

        # Now evaluate the fitness based on the trajectory
        # A fitter creature will have:
        # - gone forward more in the +x direction
        # - gone down less in the +z direction
        # This is basically just a glide ratio
        final_x = state_trajectory[-1][0]
        final_z = state_trajectory[-1][2]
        glide_ratio = final_x / final_z
        fitness = glide_ratio

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
    
    if return_logging_data:
        return fitness, state_trajectory
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
    parents = []
    if method=="truncation":
        sorted_indices = np.argsort(fitness_scores)[::-1]
        top_parents = sorted_indices[:num_parents]
        parents = [population[np.random.choice(top_parents)] for _ in range(num_parents)]
    elif method=="tournament":
        subset_indices = np.random.choice(len(fitness_scores),num_parents,replace=False)
        best_index = subset_indices[np.argmin(fitness_scores[subset_indices])]
        parents.append(population[best_index])
    elif method =="roulette":
        adjusted_fitness = np.max(fitness_scores) - fitness_scores
        probabilities = adjusted_fitness/ np.sum(adjusted_fitness)
        distribution = rv_discrete(values=(np.arange(len(fitness_scores)), probabilities))
        parent_indices = distribution.rvs(size=num_parents)# if want roulette from population len(population))
        parents = [population[i] for i in parent_indices]
    else:
        raise ValueError(f"Unsupported selection method '{method}'")
    return parents