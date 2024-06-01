import wandb 
import os 
import datetime
import numpy as np
from glob import glob

import keys

# Needed for wandb
os.environ['WANDB_API_KEY'] = keys.WANDB_API_KEY

def get_logs_folder():
    # We are currently in project/src/wandb_logging.py and
    # we want project/logs
    logs_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    return logs_folder

def get_timestamp():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")  
    return timestamp

def init_new_wandb_run(project_name, name_prefix):
    run_name = f"{name_prefix}-{get_timestamp()}"
    # Finish old run, start new one
    wandb.finish()
    # and setup pytorch lightning logger (we use standard wandb
    # to log test images, and pytorch lightning's wrapper for 
    # use in other pytorch lightning callback functions).
    # It is important that they are logging to the same place
    run = wandb.init(
        entity=keys.WANDB_ENTITY,
        project=project_name,
        name=run_name,
        #mode="offline",
    )
    return run

def log_image(key, filepath):
    wandb.log({key: wandb.Image(filepath)})

def log_video(key, filepath):
    wandb.log({key: wandb.Video(filepath)})

def log_obj(key, filepath):
    wandb.log({key: [wandb.Object3D(filepath)]})

def log_main_genetic_run_to_wandb(project_name, run_name, log_folder):
    """
    Need to log all the images and videos per generation 

    Structure is like this

    log_folder:
        generation_0:
            chromosome_distribution.png
            fitnesses.png
            fittest_individual.png
            fittest_state_trajectory.png
            fittest_individual.obj
            fittest_individual_no_wing_angle.obj
            simulation.mp4
        generation_1:
            ...
        generation_2:
            ...
        evolution.mp4
        fitnesses_over_time.png
    """

    def log_helper(folder, filename, suffix="", prefix=""):
        # Check if the thing exists and then log it if it does
        filepath = os.path.join(folder, filename)
        print(f"Attempting to log: {filepath}")
        if os.path.exists(filepath):
            # If it ends in .mp4 log it as a video
            # If it ends in .png log it as an image
            if filepath.endswith(".mp4"):
                log_video(f"{prefix}{filename}{suffix}", filepath)
            elif filepath.endswith(".png"):
                log_image(f"{prefix}{filename}{suffix}", filepath)
            elif filepath.endswith(".obj"):
                # Load the list of points from the obj file
                points = []
                for line in open(filepath, 'r'):
                    points.append(list(map(float, line.split())))
                points = np.array(points)
                log_obj(f"{prefix}{filename}{suffix}", points)
            else:
                print(f"Unknown file type for logging: {filepath}")
    
    # Initialize the W&B run
    run = init_new_wandb_run(project_name, run_name)

    # List all directories in log_folder
    generations = [folder for folder in os.listdir(log_folder) if os.path.isdir(os.path.join(log_folder, folder))]

    # Ensure that they are sorted by index
    generations = sorted(generations, key=lambda x: int(x.split('_')[1]))

    # Enumerate this
    for i, generation in enumerate(generations):
        generation_folder = os.path.join(log_folder, generation)

        def generation_log_helper(filename):
            log_helper(generation_folder, filename, prefix="generations/")
        
        # Log images
        generation_log_helper('chromosome_distribution.png')
        generation_log_helper('fitnesses.png')
        generation_log_helper('fittest_individual.png')
        generation_log_helper('fittest_state_trajectory.png')

        # Log video
        generation_log_helper('simulation.mp4')  

        # 3d point clouds are supported by W&B
        # generation_log_helper('fittest_individual_point_cloud.obj')
        # generation_log_helper('fittest_individual_point_cloud_zero_wing_angle.obj')

    # Log evolution video
    log_helper(log_folder, 'evolution.mp4', prefix="overall/")
    # And the fitnesses over time
    log_helper(log_folder, 'fitnesses_over_time.png', prefix="overall/")

    # Finish the run
    wandb.finish()

# If this is ran as a script get the latest run from the project and fo a full log
if __name__ == "__main__":
    project_name = "birds"
    name_prefix = "main-genetic"
    log_folder = get_logs_folder()

    # Get the latest run from the logs folder
    run_folders = glob(os.path.join(log_folder, "*"))
    run_folders.sort(key=os.path.getmtime)
    latest_run = run_folders[-1]

    print(f"Logging the latest run: {latest_run}")

    log_main_genetic_run_to_wandb(project_name, name_prefix, latest_run)