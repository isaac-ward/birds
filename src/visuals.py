import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_state_trajectory(filepath, state_trajectory, state_element_labels):
    """
    A state trajectory is an iterable where each iterate
    is a vector representing the state at that time step

    We want to render how these evolve over time as a bunch of plots
    """

    # +x is forward, +y is up, +z is right
    state_element_labels[0] += " (+forward, -backward)"
    state_element_labels[1] += " (+up, -down)"
    state_element_labels[2] += " (+right, -left)"

    # label rotations with roll, yaw, pitch
    state_element_labels[9]  += " (roll)"
    state_element_labels[10] += " (yaw)"
    state_element_labels[11] += " (pitch)"

    # Create subplots in matplotlib (rows each with 3 columns)
    # and then save to filepath at a nice dpi

    num_states = len(state_element_labels)
    num_time_steps = len(state_trajectory)

    # Calculate the number of rows and columns for subplots
    num_rows = 3
    num_cols = math.ceil(num_states / num_rows)

    # Create subplots
    scale = 3
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(scale*num_cols, scale*num_rows))

    for i in range(num_states):
        # Calculate the row and column, we will fill
        # go down row by row and then move to the next
        # column 
        col = i // num_rows
        row = i % num_rows

        # Get the state element
        state_element = [state[i] for state in state_trajectory]

        # Plot the state element
        axs[row, col].plot(state_element)
        axs[row, col].set_title(state_element_labels[i])

    # Save the plot
    plt.tight_layout()
    plt.savefig(filepath, dpi=600)
    plt.close()

def render_virtual_creature(virtual_creature):
    """
    Renders a virtual creature in 3D space. Basically need
    to use the chromosome information to create a 3d mesh
    representing the 3D structure of the bird
    """
    pass

def render_realtime():
    # TODO: irw + aditi to use pygame to create a real-time simulation 
    # visualization of genetic algorithm solution
    pass

def render_from_log():
    # TODO: irw + aditi to use pygame to create a visualization of
    # the genetic algorithm solution from a log file
    # TODO: irw + aditi to pull in blender scripts
    pass