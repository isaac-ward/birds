import numpy as np
from scipy.stats import rv_discrete

from dynamics import forward_step
import globals

def evaluate_fitness(virtual_creature):
    # TODO: team to dream up more complicated measures, but right now
    # we just want it to move a lot laterally and not go too far down

    # TODO rollout the forward dynamics for the virtual creature 
    # for some number of timesteps, then compute fitness based
    # on performance

    # Ensure the creature's state is reset
    virtual_creature.reset_state()

    # forward_step(virtual_creature)

    fitness = 0

    # Move a lot laterally (in xy plane)
    fitness += np.linalg.norm(virtual_creature.position_xyz[:2])

    # Don't go too far down
    fitness -= virtual_creature.position_xyz[2]

    # If you go way too far down you're dead
    if virtual_creature.position_xyz[2] < globals.TOO_LOW_Z:
        fitness = globals.FITNESS_PENALTY_TOO_LOW

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