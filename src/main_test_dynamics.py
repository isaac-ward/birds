import os 
from tqdm import tqdm
import glob

from genetic.chromosome import Chromosome
from genetic.virtual_creature import VirtualCreature
from genetic.fitness import evaluate_fitness, select_fittest_individuals
from dynamics import forward_step
import visuals

import utils

if __name__ == "__main__":

    # Set up a log folder
    log_folder = utils.make_log_folder()
    
    # Make a creature
    # creature = VirtualCreature.random_init()
    creature = VirtualCreature(Chromosome([10.0, 0.5, 1.0, 1.0, 1.0, 0.26, 0.0, 0.0]))

    # Plot what it looks like at the start
    visuals.render_3d_frame(
        f"{log_folder}/mesh.png",
        creature
    )

    # Can reset the creature's state
    creature.reset_state()

    print(creature)

    # Run the creature for some time
    # Simulation parameters
    simulation_time_seconds = 2.5
    dt = 0.05
    num_steps = int(simulation_time_seconds / dt)
    # Video parameters
    render_video = True
    fps = 25
    # Note the state trajectory
    state_trajectory = []
    for i in tqdm(range(num_steps), desc="Running forward dynamics"):

        # Run the dynamics forward
        forward_step(creature, dt=dt)

        # Get the state vector and log
        state_vector = creature.get_state_vector()
        state_trajectory.append(state_vector)

        # Plot every frame so that we can see what's going on
        if render_video:
            visuals.render_3d_frame(
                f"{log_folder}/frames/{i}.png",
                creature,
                extents=[(-5,20), (-5,5), (-5,5)],
                past_3d_positions=state_trajectory,
                current_time_s=i*dt,
                total_time_s=simulation_time_seconds
            )
        
    # Plot the state trajectory
    visuals.plot_state_trajectory(
        f"{log_folder}/state_trajectory.png",
        state_trajectory,
        VirtualCreature.get_state_vector_labels()
    )

    # Plot a video of the creature from the frames
    if render_video:
        # Files are called *.png, so we need to sort them by the number in the filename,
        # using os 
        files = glob.glob(f"{log_folder}/frames/*.png")
        files_sorted = sorted(files, key=lambda x: int(os.path.basename(x).split(".")[0]))
        visuals.sequence_frames_to_video(
            files_sorted,
            f"{log_folder}/video.mp4",
            fps=fps
        )
