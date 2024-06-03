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
    creature = VirtualCreature(Chromosome([10.673677623851926, 0.7877161576012752, 1.0349316770790011, 0.6639940004540399, 0.3721394664060006, 0.24825283047335453, 0, 0, 1.0724937401154104, -0.13412098274150352, -1.2960001394273934, -1.569520314455652, 0.0, 0.0, -0.6956817397975144, -0.0032504523647152016, 7.5278990975356415, -0.00733121136508792, -4.031960468216258, -0.9396496272545036, 0.5461614052229473, 0.9550588311001693, 0.0, 0.0, -2.3697120609113185, 0.00442813335919905, 2.2013231124926858, 0.006261077264949214]))

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

