import os 
import numpy as np

from genetic.chromosome import Chromosome
from genetic.virtual_creature import VirtualCreature
from genetic.fitness import evaluate_fitness, select_fittest_individuals

import visuals

import utils

def solve_ga(
    population_size,
    num_generations,
    num_parents_per_generation,
    log_folder,
):
    # Will track the following
    fitness_scores_per_generation = []

    # Start by initializing the population
    population = [ VirtualCreature.random_init() for _ in range(population_size) ]
    fitness_test_mode = 1 #1 for far lateral and else for waypoint
    # Loop over some number of generations
    for generation_index in range(num_generations):
       
        # Evaluate the fitness of each individual in the population
        evaluate_fitness_results = [evaluate_fitness(individual, fitness_test_mode, return_logging_data=True) for individual in population ]
        fitness_scores = [result[0] for result in evaluate_fitness_results]
        state_trajectories = [result[1] for result in evaluate_fitness_results]
        fitness_scores_per_generation.append(fitness_scores)

        # Log the state trajectories for the best individual
        best_individual_state_trajectory = state_trajectories[population.index(max(population, key=evaluate_fitness))]
        visuals.plot_state_trajectory(
            f"{log_folder}/state_trajectory_fittest_of_gen{generation_index}.png",
            best_individual_state_trajectory
        )

        # Select the fittest individuals to be parents
        parents = select_fittest_individuals(population, fitness_scores, num_parents=num_parents_per_generation, method="truncation") #can be truncation, tournament, roulette for method

        # Create the next generation by crossing over and mutating the parents
        children = []
        for _ in range(population_size):
            parent1 = np.random.choice(parents)
            parent2 = np.random.choice(parents)
            child = parent1.crossover(parent2)
            child.mutate()
            children.append(child)

        # Replace the old population with the new generation
        population = children

    # Plot the average fitness and the best fitness over time
    

    # The last generation is the final population and the most optimal
    # individual in the last generation is the most optimal solution
    best_individual = max(population, key=evaluate_fitness) #not sure how this works with fitnesstest argument
    return best_individual

if __name__ == "__main__":

    # Set up a log folder
    log_folder = utils.make_log_folder()
    
    # Run the genetic algorithm to solve the problem
    best_individual = solve_ga(
        population_size=128,
        num_generations=32,
        num_parents_per_generation=64,
        log_folder=log_folder
    )

    # Print the best individual
    print(best_individual.chromosome)