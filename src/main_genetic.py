import os 
import numpy as np

from genetic.chromosome import Chromosome
from genetic.virtual_creature import VirtualCreature
from genetic.selection import evaluate_fitness, select_fittest_individuals

def solve_ga(
    population_size=128,
    num_generations=64,
    num_parents_per_generation=64,
):
    # Start by initializing the population
    population = [ VirtualCreature.random_init() for _ in range(population_size) ]

    # Loop over some number of generations
    for generation_index in range(num_generations):

        # Evaluate the fitness of each individual in the population
        fitness_scores = [ evaluate_fitness(individual) for individual in population ]

        # Select the fittest individuals to be parents
        parents = select_fittest_individuals(population, fitness_scores, num_parents=num_parents_per_generation)

        # Create the next generation by crossing over and mutating the parents
        children = []
        for _ in range(population_size):
            parent1 = np.random.choice(parents)
            parent2 = np.random.choice(parents)
            # TODO: many strategies for this, need to explore
            child = parent1.crossover(parent2)
            child.mutate()
            children.append(child)

        # Replace the old population with the new generation
        population = children

    # The last generation is the final population and the most optimal
    # individual in the last generation is the most optimal solution
    best_individual = max(population, key=evaluate_fitness)
    return best_individual

if __name__ == "__main__":
    
    # Run the genetic algorithm to solve the problem
    best_individual = solve_ga()

    # Print the best individual
    print(best_individual.chromosome)