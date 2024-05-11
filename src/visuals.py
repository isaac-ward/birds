import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

import moviepy.editor as mpy

def plot_state_trajectory(filepath, state_trajectory, state_element_labels):
    """
    A state trajectory is an iterable where each iterate
    is a vector representing the state at that time step

    We want to render how these evolve over time as a bunch of plots
    """

    # Add three labels for top down, right side view, and behind view, 
    # and insert at the start
    to_add = ["top down view", "right side view", "behind view"]
    view_data = [
        # horizontal axis is x, vertical axis is z
        [[state[0], state[2]] for state in state_trajectory],
        # horizontal axis is x, vertical axis is y
        [[state[0], state[1]] for state in state_trajectory],
        # horizontal axis is z, vertical axis is y
        [[state[2], state[1]] for state in state_trajectory]
    ]
    view_data_x_labels = ["x (forward/back)", "x (forward/back)", "z (right/left)"]
    view_data_y_labels = ["z (right/left)", "y (up/down)", "y (up/down)"]

    # Create subplots in matplotlib (rows each with 3 columns)
    num_states = len(state_element_labels)
    num_time_steps = len(state_trajectory)

    # Calculate the number of rows and columns for subplots
    num_rows = 3
    num_cols = math.ceil(num_states / num_rows) + 1 # +1 for the view data

    # Create subplots
    scale = 3
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(scale*num_cols, scale*num_rows))

    # Start with the view data
    num_added = len(to_add)
    cmap = plt.get_cmap('winter')
    for i in range(num_added):
        # Calculate the row and column, we will fill
        # go down row by row and then move to the next
        # column 
        col = i // num_rows
        row = i % num_rows

        # Set the title
        axs[row, col].set_title(to_add[i])
        # And labels
        axs[row, col].set_xlabel(view_data_x_labels[i])
        axs[row, col].set_ylabel(view_data_y_labels[i])

        # Get the view data and plot it
        view = view_data[i]
        # Color map so that time progression is clear
        horiz = [v[0] for v in view]
        vert = [v[1] for v in view]
        colors = [cmap(i / num_time_steps)[:3] for i in range(num_time_steps)]
        axs[row, col].scatter(horiz, vert, color=colors, s=1, marker='.')

    # Then the rest
    for i in range(num_states):

        # Calculate the row and column, we will fill
        # go down row by row and then move to the next
        # column 
        col = i // num_rows + 1 # +1 for the view data
        row = i % num_rows

        # Set the title and time as x axis
        axs[row, col].set_title(state_element_labels[i])
        axs[row, col].set_xlabel("time step")

        # Get the state element as a vector and plot it
        state_element = [state[i] for state in state_trajectory]
        axs[row, col].plot(state_element)

    # Save the plot
    plt.tight_layout()
    plt.savefig(filepath, dpi=600)
    plt.close()

def get_3d_translation_matrix(translation):
    return np.array([
        [1, 0, 0, translation[0]],
        [0, 1, 0, translation[1]],
        [0, 0, 1, translation[2]],
        [0, 0, 0, 1],
    ])

def get_3d_rotation_matrix(euler_angles):
    # TODO could use a quaternion
    rot_x = np.array([
        [1, 0, 0, 0],
        [0, math.cos(euler_angles[0]), -math.sin(euler_angles[0]), 0],
        [0, math.sin(euler_angles[0]), math.cos(euler_angles[0]), 0],
        [0, 0, 0, 1],
    ])
    rot_y = np.array([
        [math.cos(euler_angles[1]), 0, math.sin(euler_angles[1]), 0],
        [0, 1, 0, 0],
        [-math.sin(euler_angles[1]), 0, math.cos(euler_angles[1]), 0],
        [0, 0, 0, 1],
    ])
    rot_z = np.array([
        [math.cos(euler_angles[2]), -math.sin(euler_angles[2]), 0, 0],
        [math.sin(euler_angles[2]), math.cos(euler_angles[2]), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    return rot_x @ rot_y @ rot_z

def render_3d_frame(
    filepath, 
    virtual_creature, 
    render_as_mesh=True,
    if_mesh_render_verts=True,
    if_mesh_render_cog=True,
    extents=[],
    past_3d_positions=[],
    current_time_s=-1,
    total_time_s=-1,
):
    """
    Renders a virtual creature in 3D space. Basically need
    to use the chromosome information to create a 3d mesh
    representing the 3D structure of the bird

    Saves the plot of the virtual creature to a file at filepath
    
    Optionally renders as a mesh with vertex points and cog specifically
    rendered if desired

    Optionally can specify the position and rotation of the creature

    Optionally can specify the extents of the plot, otherwise they'll be made
    to fit the creature
    """

    # Make the plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Ensure the folder exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # We might rendedr as a mesh
    if render_as_mesh:
    
        # Get the vertices and faces and show in matplotlib
        vertices, faces = virtual_creature.get_mesh_verts_and_faces()

        # Need to apply the translation and rotation to the vertices
        # before rendering. Position is x,y,z and rotation is euler
        # angles in radians about x,y,z axes
        position = virtual_creature.position_xyz
        rotation = virtual_creature.rotation_xyz
        transformation = get_3d_translation_matrix(position) @ get_3d_rotation_matrix(rotation)
        # Do the transformation in 4d and then back to 3d
        vertices = [(transformation @ np.array([*v, 1]))[:3] for v in vertices]

        # Render the mesh with trisurf
        ax.plot_trisurf(
            [v[0] for v in vertices],
            [v[1] for v in vertices],
            [v[2] for v in vertices],
            triangles=faces,
            cmap='spring',
            alpha=0.5,
            edgecolor='k'
        )

        # Annunciate the vertices with a 3d scatter plot
        # with little markers at each vertex
        if if_mesh_render_verts:
            ax.scatter(
                [v[0] for v in vertices],
                [v[1] for v in vertices],
                [v[2] for v in vertices],
                color='black',
                marker='o',
                s=1,
            )  

        # Draw a downwards pointing vector at the CoG position
        # Ensure that this has an arrow head
        if if_mesh_render_cog:

            # Where is the CoG?
            norm_COG_position = virtual_creature.chromosome.norm_COG_position
            COG_position = norm_COG_position * virtual_creature.chromosome.wing_root_chord
            cog_x = (1 - COG_position) * virtual_creature.chromosome.wing_root_chord
            cog_world_position = (transformation @ np.array([cog_x, 0, 0, 1]))[:3]

            ax.quiver(
                # Start at the CoG position
                cog_world_position[0], cog_world_position[1], cog_world_position[2],
                # dXYZ components of the arrow
                0, -2, 0,
                color='red',
            )
            # Scatter with an x
            ax.scatter([cog_world_position[0]], [cog_world_position[1]], [cog_world_position[2]], color='red', marker='x')

    else:
        # Just render a vertex at the given position
        ax.scatter([position[0]], [position[1]], [position[2]], color='black', marker='o')
    
    # Edit the axes so that +x is forward, +y is up, +z is right
    ax.set_xlabel("x (+forward/-back)")
    ax.set_ylabel("y (+up/-down)")
    ax.set_zlabel("z (+right/-left)")

    # Set axis limits so that everything is visible, unless we
    # were given extents in which case just use those
    if extents == []:
        extents = [
            (min([v[0] for v in vertices]), max([v[0] for v in vertices])),
            (min([v[1] for v in vertices]), max([v[1] for v in vertices])),
            (min([v[2] for v in vertices]), max([v[2] for v in vertices]))
        ]

        def modify_extent_if_too_close(extent):
            epsilon = 0.1
            if abs(extent[0] - extent[1]) < epsilon:
                return (extent[0] - epsilon, extent[1] + epsilon)
            return extent
        extents = [modify_extent_if_too_close(extent) for extent in extents]

        wingspan = extents[2]

        # Scale up bounds by scale factor
        sf = 2
        extents = [ (sf * extent[0], sf * extent[1]) for extent in extents]

        # Set z ticks to show wingspan
        ax.set_zticks([0, *wingspan])
        # Hide x and y axis ticks
        ax.set_xticks([])
        ax.set_yticks([])

    # Set the bounding box for the render
    ax.set_xlim3d(*extents[0])
    ax.set_ylim3d(*extents[1])
    ax.set_zlim3d(*extents[2]) 
    # Compute ratios if given
    size_x = extents[0][1] - extents[0][0]
    size_y = extents[1][1] - extents[1][0]
    size_z = extents[2][1] - extents[2][0]
    ax.set_box_aspect([size_x, size_y, size_z], zoom=0.8) 

    # Set azimuth elevation and roll on plot
    rot = 30
    ax.view_init(azim=-90 - rot, elev=110, roll=-rot)

    # Render previous positions as a line if given
    if len(past_3d_positions) > 0:
        # Get the x,y,z components of the past positions
        x = [p[0] for p in past_3d_positions]
        y = [p[1] for p in past_3d_positions]
        z = [p[2] for p in past_3d_positions]

        # Plot the line
        ax.plot(x, y, z, color='black', alpha=0.5)

    # Render the time information if given in the top right
    if current_time_s != -1 and total_time_s != -1:
        def plot_text_y_down(y, text):
            ax.text2D(0.95, y, text, transform=ax.transAxes, fontsize=12, color='black', ha='right')
        plot_text_y_down(1 - 0.05, f"t={current_time_s:.4f} s")
        plot_text_y_down(1 - 0.07, f"T={total_time_s:.4f} s")   

    #plt.show()

    # Save the plot
    plt.tight_layout()
    plt.savefig(filepath, dpi=600)
    plt.close()

def sequence_frames_to_video(filepaths_in, filepath_out, fps=30):
    # Use moviepy to accomplish this
    clip = mpy.ImageSequenceClip(filepaths_in, fps=fps)
    # Write the video
    clip.write_videofile(filepath_out, fps=fps)

def render_realtime():
    # TODO: irw + aditi to use pygame to create a real-time simulation 
    # visualization of genetic algorithm solution
    pass

def render_from_log():
    # TODO: irw + aditi to use pygame to create a visualization of
    # the genetic algorithm solution from a log file
    # TODO: irw + aditi to pull in blender scripts
    pass