import os 
from tqdm import tqdm
import glob

from genetic.chromosome import Chromosome
from genetic.virtual_creature import VirtualCreature
from genetic.fitness import parallel_evaluate_fitness, evaluate_fitness, select_fittest_individuals
from dynamics import forward_step
import globals
import visuals2 as vis

import utils

if __name__ == "__main__":

    # Set up a log folder
    log_folder = utils.make_log_folder()
    
    # Make a creature
    creature = VirtualCreature(Chromosome([8.96499136927574, 0.40901405185041106, 1.0176496479144665, 0.801114588448173, 0.3857502961244982, 0.2486288884031518, 1, 0, 1.6331792420065518, 2.126760168558163, 0.3276203339668462, -2.8454612428225037, 0.0, 0.0, 0.27419642125231647, 0.0051021235756654425, -7.483627388702725, -0.000866248134828905, 3.2479770561464676, 1.0116350867154793, 0.27550557880742677, 0.5328702575989213, 0.0, 0.0, -1.2508924754248856, 0.004059034215754024, -2.284681214206225, 0.009458234867048049]))

    # Can reset the creature's state
    creature.reset_state()

    # Run the creature for some time
    # Simulation parameters
    simulation_time_seconds = 4
    dt = globals.DT
    t = 0
    num_steps = int(simulation_time_seconds / dt)
    
    # Evaluate the fitness of each individual in the population
    evaluate_fitness_results = parallel_evaluate_fitness([creature], test_mode=1)
    fitness_scores     = [result[0] for result in evaluate_fitness_results]
    # Fitness components is a list of dicts, but we want a dict of lists
    fitness_components = [result[1] for result in evaluate_fitness_results]
    fitness_components = {key: [dic[key] for dic in fitness_components] for key in fitness_components[0]}
    state_trajectories = [result[2] for result in evaluate_fitness_results]

    # Log the state trajectories for the best individual
    vis.plot_state_trajectory(
        f"{log_folder}/state_trajectory.png",
        state_trajectory=state_trajectories[0]
    )

