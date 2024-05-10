import os 
from tqdm import tqdm

from genetic.chromosome import Chromosome
from genetic.virtual_creature import VirtualCreature
from genetic.fitness import evaluate_fitness, select_fittest_individuals
from dynamics import forward_step
import visuals

import utils

if __name__ == "__main__":

    # Set up a log folder
    log_folder = utils.make_log_folder()
    
    # Make a random creature
    # creature = VirtualCreature.random_init()
    creature = VirtualCreature(Chromosome([10.0, 0.5, 1.0, 1.0, 1.0, 0.26, 0.0, 0.0]))

    # Plot what it looks like
    visuals.render_virtual_creature(
        f"{log_folder}/mesh.png",
        creature
    )

    # Can reset the creature's state
    creature.reset_state()

    print(creature)

    # Run the creature for a few steps
    simulation_time_seconds = 10
    dt = 0.01
    num_steps = int(simulation_time_seconds / dt)
    state_trajectory = []
    for _ in tqdm(range(num_steps), desc="Running forward dynamics"):

        # Run the dynamics forward
        forward_step(creature, dt=dt)

        # Get the state vector and log
        state_vector = creature.get_state_vector()
        state_trajectory.append(state_vector)
        
    # Plot the state trajectory
    visuals.plot_state_trajectory(
        f"{log_folder}/state_trajectory.png",
        state_trajectory,
        VirtualCreature.get_state_vector_labels()
    )