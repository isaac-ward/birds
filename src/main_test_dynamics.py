import os 

from genetic.chromosome import Chromosome
from genetic.virtual_creature import VirtualCreature
from genetic.fitness import evaluate_fitness, select_fittest_individuals
from dynamics import forward_step

if __name__ == "__main__":
    
    # Make a random creature
    creature = VirtualCreature.random_init()

    # Can reset the creature's state
    creature.reset_state()

    print(creature)

    # Run the creature for a few steps
    num_steps = 8
    for _ in range(num_steps):
        print(f"step {_}:\n")
        print(creature)
        forward_step(creature)
    print(creature)