import numpy as np

import globals

def evaluate_fitness(virtual_creature):
    # TODO: team to dream up more complicated measures, but right now
    # we just want it to move a lot laterally and not go too far down

    # TODO rollout the forward dynamics for the virtual creature 
    # for some number of timesteps, then compute fitness based
    # on performance

    fitness = 0

    # Move a lot laterally (in xy plane)
    fitness += np.linalg.norm(virtual_creature.position_xyz[:2])

    # Don't go too far down
    fitness -= virtual_creature.position_xyz[2]

    # If you go way too far down you're dead
    if virtual_creature.position_xyz[2] < globals.TOO_LOW_Z:
        fitness = globals.FITNESS_PENALTY_TOO_LOW

    return fitness

def select_fittest_individuals(population, fitness_scores, num_parents):
    """
    population - list of individual VirtualCreature objects describing the population
    fitness_scores - list of floats describing the fitness of each corresponding VirtualCreature in the population
    num_parents - integer describing the number of parents to select from the list 
    returns - list of VirtualCreature objects that are the selected parents
    """
    # TODO: Aditi to implement this function and return
    # the individuals in the population that will be selected
    # as parents for the next generation based on their fitness
    # scores (e.g. tournament, roulette wheel, rank-based, etc. - might
    # want to compare a few)
    pass