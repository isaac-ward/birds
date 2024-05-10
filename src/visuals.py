import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_state_trajectory(filepath, state_trajectory, state_element_labels):
    """
    A state trajectory is an iterable where each iterate
    is a vector representing the state at that time step

    We want to render how these evolve over time as a bunch of plots
    """

    # Create as subplots in matplotlib (rows each with 3 columns)
    # and then save to filepath at 600+ dpi
    
    num_states = len(state_element_labels)
    num_time_steps = len(state_trajectory)

    # Calculate the number of rows and columns for subplots
    num_rows = num_states // 3 + (1 if num_states % 3 != 0 else 0)
    num_cols = min(3, num_states)

    # Create subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))
    
    # Flatten axs if it's a 1D array
    if num_rows == 1:
        axs = [axs]

    for i in range(num_states):
        row = i // 3
        col = i % 3
        axs[row][col].plot(range(num_time_steps), [state[i] for state in state_trajectory])
        axs[row][col].set_title(state_element_labels[i])

    # Save the plot
    plt.tight_layout()
    plt.savefig(filepath, dpi=600)
    plt.close()
    
def render_state_trajectory():
    """
    """
    
    # +x is forward, +y is right, +z is up

    # TODO

def render_realtime():
    # TODO: irw + aditi to use pygame to create a real-time simulation 
    # visualization of genetic algorithm solution
    pass

def render_from_log():
    # TODO: irw + aditi to use pygame to create a visualization of
    # the genetic algorithm solution from a log file
    # TODO: irw + aditi to pull in blender scripts
    pass