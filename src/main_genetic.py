import os 
import numpy as np
import copy
import math
from tqdm import tqdm
import random
import pickle

from genetic.chromosome import Chromosome
from genetic.virtual_creature import VirtualCreature
from genetic.fitness import parallel_evaluate_fitness, evaluate_fitness, select_fittest_individuals

import visuals2 as vis

import utils

def solve_ga(
    population_size,
    num_generations,
    num_parents_per_generation,
    log_folder,
    logging=True,
    log_videos=False,
):
    
    # Set all the randoms for repeatability
    random.seed(0)
    np.random.seed(0)

    # Can't have more parents than population size
    if num_parents_per_generation > population_size:
        raise ValueError(f"Can't have more parents {num_parents_per_generation} than population size {population_size}")

    # Will track the following
    fitness_scores_per_generation = []
    fittest_creature_pkl_per_generation = []

    # Start by initializing the population
    population = [ VirtualCreature.random_init() for _ in range(population_size) ]
    fitness_test_mode = 1 #1 for far lateral and else for waypoint
    # Loop over some number of generations
    pbar = tqdm(total=num_generations, desc=f"Evolving creatures over {num_generations} generations")
    for generation_index in range(num_generations):
       
        # Evaluate the fitness of each individual in the population
        evaluate_fitness_results = parallel_evaluate_fitness(population, test_mode=fitness_test_mode)
        fitness_scores     = [result[0] for result in evaluate_fitness_results]
        # Fitness components is a list of dicts, but we want a dict of lists
        fitness_components = [result[1] for result in evaluate_fitness_results]
        fitness_components = {key: [dic[key] for dic in fitness_components] for key in fitness_components[0]}
        state_trajectories = [result[2] for result in evaluate_fitness_results]
        fitness_scores_per_generation.append(fitness_scores)

        # Select the fittest individuals to be parents
        parents, parent_indices = select_fittest_individuals(population, fitness_scores, num_parents=num_parents_per_generation, method="truncation") #can be truncation, tournament, roulette for method

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
            print("")
            print(f"The highest fitness encountered this generation was creature {fittest_index} with fitness {fitness_scores[fittest_index]:.2f}")
            print("")

            # Save the fittest individual as a string
            with open(f"{generation_folder}/fittest_individual.txt", "w") as f:
                # Might get this:
                # UnicodeEncodeError: 'charmap' codec can't encode character '\u03c9' in position 110: character maps to <undefined>
                # Because of weird characters in the chromosome
                string = str(population[fittest_index])
                # So we'll just encode it to ascii
                f.write(string.encode("ascii", "ignore").decode())
            # And pickle it
            filepath_fittest_individual = f"{generation_folder}/fittest_individual.pkl"
            fittest_creature_pkl_per_generation.append(filepath_fittest_individual)
            with open(filepath_fittest_individual, "wb") as f:
                pickle.dump(population[fittest_index], f)

            # Log the fitness scores of the population as as box plot
            vis.plot_fitnesses(
                f"{generation_folder}/fitnesses.png",
                fitness_scores,
                fitness_components,
                highlight_indices=parent_indices
            )

            # Log the state trajectories for the best individual
            vis.plot_state_trajectory(
                f"{generation_folder}/fittest_state_trajectory.png",
                state_trajectory=state_trajectories[fittest_index]
            )

            # Log what this virtual creature looks like
            fittest_copy = copy.deepcopy(population[fittest_index])
            fittest_copy.reset_state()
            # Save as object files
            fittest_copy.save_as_obj(f"{generation_folder}/fittest_individual.obj", t=0)
            fittest_copy.save_as_obj(f"{generation_folder}/fittest_individual_zero_wing_angle.obj", t=-1)
            # And point cloud list
            fittest_copy.save_as_point_cloud(f"{generation_folder}/fittest_individual_point_cloud.obj", t=0)
            fittest_copy.save_as_point_cloud(f"{generation_folder}/fittest_individual_point_cloud_zero_wing_angle.obj", t=-1)

            # Log the distribution of the chromosomes in the population and where the fittest individual is
            vis.plot_chromosome_distributions(
                f"{generation_folder}/chromosome_distribution.png",
                population,
                fittest_index
            )

            # Log an animation for the best individual
            if log_videos:
                vis.render_simulation_of_creature(
                    generation_folder, 
                    fittest_copy,
                    state_trajectory=state_trajectories[fittest_index],
                    #test_mode=fitness_test_mode
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
    vis.plot_fitnesses_over_time(
        f"{log_folder}/fitnesses_over_time.png",
        fitness_scores_per_generation
    )

    # # Plot the evolution of the creatures over time
    # if log_videos:
    #     vis_legacy.render_evolution_of_creatures(
    #         log_folder,
    #         fittest_creature_pkl_per_generation
    #     )

    # The last generation is the final population and the most optimal
    # individual in the last generation is the most optimal solution
    best_individual = population[np.argmax(fitness_scores)]
    return best_individual

if __name__ == "__main__":

    # Set up a log folder
    log_folder = utils.make_log_folder()
    
    # Run the genetic algorithm to solve the problem
    population_size = 16
    num_parents_per_generation = math.ceil(0.2 * population_size)
    best_individual = solve_ga(
        population_size=population_size,
        num_generations=16,
        num_parents_per_generation=num_parents_per_generation,
        log_folder=log_folder,
        logging=True,
        log_videos=False,
    )

    # Print the best individual
    print(best_individual.chromosome)