import os 
import numpy as np
import copy
from tqdm import tqdm

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
    logging=True
):
    # Will track the following
    fitness_scores_per_generation = []

    # Start by initializing the population
    population = [ VirtualCreature.random_init() for _ in range(population_size) ]
    fitness_test_mode = 1 #1 for far lateral and else for waypoint
    # Loop over some number of generations
    pbar = tqdm(total=num_generations, desc=f"Evolving creatures over {num_generations} generations")
    for generation_index in range(num_generations):
       
        # Evaluate the fitness of each individual in the population
        evaluate_fitness_results = [evaluate_fitness(individual, fitness_test_mode, return_logging_data=True) for individual in population ]
        fitness_scores = [result[0] for result in evaluate_fitness_results]
        state_trajectories = [result[1] for result in evaluate_fitness_results]
        fitness_scores_per_generation.append(fitness_scores)

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

        # Do some logging
        if logging:
            # Create a folder for this generation to store logs
            generation_folder = f"{log_folder}/generation_{generation_index}"
            os.makedirs(generation_folder, exist_ok=True)

            # Who was the mightiest of them all?
            fittest_index = np.argmax(fitness_scores)

            # Log the fitness scores of the population as as box plot
            visuals.plot_fitnesses(
                f"{generation_folder}/fitnesses.png",
                fitness_scores
            )

            # Log the state trajectories for the best individual
            visuals.plot_state_trajectory(
                f"{generation_folder}/fittest_state_trajectory.png",
                state_trajectory=state_trajectories[fittest_index]
            )

            # Log what this virtual creature looks like
            fittest_copy = copy.deepcopy(population[fittest_index])
            fittest_copy.reset_state()
            visuals.render_3d_frame(
                f"{generation_folder}/fittest_mesh.png",
                fittest_copy
            )

            # And save the fittest individual as a string
            with open(f"{generation_folder}/fittest_individual.txt", "w") as f:
                f.write(str(population[fittest_index]))

            # Log the distribution of the chromosomes in the population and where the fittest individual is
            visuals.plot_chromosome_distributions(
                f"{generation_folder}/chromosome_distribution.png",
                population,
                fittest_index
            )

            # Log an animation for the best individual
            visuals.render_video_of_creature(
                generation_folder, 
                fittest_copy,
                simulation_time_seconds=5.0,
                dt=0.05,
                fps=25
            )

        # Replace the old population with the new generation
        population = children

        # Update the progress bar with the current best fitness
        # (change the description)
        pbar.set_postfix({"worst_fitness"  : min(fitness_scores)})
        pbar.set_postfix({"average_fitness": np.mean(fitness_scores)})
        pbar.set_postfix({"best_fitness"   : max(fitness_scores)})
        pbar.update(1)

    # Plot the average fitness and the best fitness over time
    visuals.plot_fitnesses_over_time(
        f"{log_folder}/fitnesses_over_time.png",
        fitness_scores_per_generation
    )

    # The last generation is the final population and the most optimal
    # individual in the last generation is the most optimal solution
    best_individual = max(population, key=evaluate_fitness) #not sure how this works with fitnesstest argument
    return best_individual

if __name__ == "__main__":

    # Set up a log folder
    log_folder = utils.make_log_folder()
    
    # Run the genetic algorithm to solve the problem
    best_individual = solve_ga(
        population_size=16,
        num_generations=16,
        num_parents_per_generation=6,
        log_folder=log_folder
    )

    # Print the best individual
    print(best_individual.chromosome)