import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import copy
import pickle

import moviepy.editor as mpy
from tqdm import tqdm

from dynamics import forward_step
from genetic.virtual_creature import VirtualCreature
import globals

def plot_state_trajectory(filepath, state_trajectory, state_element_labels=VirtualCreature.get_state_vector_labels()):
    """
    A state trajectory is an iterable where each iterate
    is a vector representing the state at that time step

    We want to render how these evolve over time as a bunch of plots
    """

    # Add three labels for top down, right side view, and behind view, 
    # and insert at the start
    # +x is forward, +y is right, +z is down
    to_add = ["top down view", "right side view", "behind view"]
    view_data = [
        # top down is forward v right
        [[state[0], state[1]] for state in state_trajectory],
        # right side is forward v down
        [[state[0], state[2]] for state in state_trajectory],
        # behind view is right v down
        [[state[1], state[2]] for state in state_trajectory]
    ]
    labels = ["x (-back/+forward)", "y (+right/-left)", "z (+down/-up)"]
    view_data_x_labels = [labels[0], labels[0], labels[1]]
    view_data_y_labels = [labels[1], labels[2], labels[2]]

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
        # Draw the line segment by segment
        for t in range(num_time_steps - 1):
            axs[row, col].plot(
                [horiz[t], horiz[t+1]],
                [vert[t], vert[t+1]],
                color=colors[t],
                alpha=0.5
            )

        # Want the origin for these plots to be at the top left
        axs[row, col].invert_yaxis()

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
        # print(state_element_labels[i])
        # print(state_element)
        axs[row, col].plot(state_element)

    # Save the plot
    plt.tight_layout()
    plt.savefig(filepath, dpi=600)
    plt.close()

def plot_fitnesses_over_time(filepath, fitness_scores_per_generation):
    """
    Plot the fitness scores of the population over time
    """

    # Create the plot
    fig = plt.figure(figsize=(6, 6), dpi=600)
    ax = fig.add_subplot(111)
    
    # Want to plot the worst (red), best (green), and average (blue) fitness scores
    num_generations = len(fitness_scores_per_generation)
    x = list(range(num_generations))
    worst_fitnesses = [min(fitness_scores) for fitness_scores in fitness_scores_per_generation]
    best_fitnesses = [max(fitness_scores) for fitness_scores in fitness_scores_per_generation]
    # If the best/worst are -inf, then forget them
    worst_fitnesses = [fitness if fitness != -np.inf else np.nan for fitness in worst_fitnesses]
    best_fitnesses  = [fitness if fitness != -np.inf else np.nan for fitness in best_fitnesses]
    # Average fitness
    average_fitnesses = [np.mean(fitness_scores) for fitness_scores in fitness_scores_per_generation]

    # Plot the fitness scores as a fillebetween around the average fitness
    ax.fill_between(x, worst_fitnesses, best_fitnesses, color='blue', alpha=0.3, label='best/worst fitness')
    ax.plot(x, average_fitnesses, color='blue', label='average fitness')

    # Add a legend
    ax.legend()

    # Save the plot
    plt.tight_layout()
    plt.savefig(filepath, dpi=600)
    plt.close()

def plot_fitnesses(filepath, fitness_scores):
    """
    Plot the fitness scores of the population
    """

    # Create the plot
    fig = plt.figure(figsize=(6, 6), dpi=600)
    ax = fig.add_subplot(111)

    # Ignore -inf values
    fitness_scores = [fitness for fitness in fitness_scores if fitness != -np.inf]

    # Fitnesses may be widely spread, with few overlapping, so
    # a good choice of plot is a ...
    # histogram of fitness scores
    ax.hist(fitness_scores, bins=30, edgecolor='black', alpha=0.7)
    ax.set_title('Distribution of Fitness Scores')
    ax.set_xlabel('Fitness')
    ax.set_ylabel('Frequency')

    # Save the plot
    plt.tight_layout()
    plt.savefig(filepath, dpi=600)
    plt.close()

def plot_chromosome_distributions(filepath, population, fittest_index):
    """
    We have a population of virtual creatures that each have some value
    for each chromosome. We want to visualize the distribution of each
    of them with a box plot, and then highlight the fittest individual
    """

    # What are the cutoffs for the chromosome values? Each will be a subplot
    names = []
    min_values = []
    max_values = []
    values_per_gene = {}

    # Get the chromosome values for each individual
    for gene in range(len(globals.CHROMOSOME_DEFINITION)):
        name = globals.CHROMOSOME_DEFINITION[gene].name
        names.append(name)
        min_values.append(globals.CHROMOSOME_DEFINITION[gene].min_val)
        max_values.append(globals.CHROMOSOME_DEFINITION[gene].max_val)
        values_per_gene[name] = [getattr(creature.chromosome, name) for creature in population]

    # Here's how we're going to plot this. There will be 
    # 9 rows of plots, each row corresponding to some grouping.
    # In each row we'll have the following number of plots
    plots_per_row = [8, 6, 4, 1, 4, 6, 4, 1, 4]
    n_rows = len(plots_per_row)
    n_cols = max(plots_per_row)

    # Create the plot
    fig = plt.figure(figsize=(16, 16), dpi=600)
    
    # Create the subplots
    axes = [fig.add_subplot(n_rows, n_cols, i+1) for i in range(n_rows * n_cols)]

    # Plot row by row
    gene_index = 0
    for i in range(n_rows):
        for j in range(n_cols):
            if j < plots_per_row[i]:
                # Get the name of the gene
                gene_name = names[gene_index]
                # Get the values for this gene
                values = values_per_gene[gene_name]
                # Plot the box plot
                axes[i * n_cols + j].boxplot(values)
                axes[i * n_cols + j].set_title(gene_name)
                offset = (max_values[gene_index] - min_values[gene_index]) / 4
                low = min_values[gene_index] - offset
                high = max_values[gene_index] + offset
                axes[i * n_cols + j].set_ylim(low, high)
                # Add a grid
                axes[i * n_cols + j].grid(True)

                # Highlight the fittest individual
                axes[i * n_cols + j].axhline(getattr(population[fittest_index].chromosome, gene_name), color='green')
                # Plot the actual number just above the line, at the very left of the plot
                x_text = 0.35
                y_text = getattr(population[fittest_index].chromosome, gene_name)
                axes[i * n_cols + j].text(x_text, y_text, f"{y_text:.2f}", color='green')

                # Remove the xticks
                axes[i * n_cols + j].set_xticks([])
                # Make y ticks have 8 divisions from the gene (offset) min to max + some more
                axes[i * n_cols + j].set_yticks(np.linspace(low, high, 8))

                # Move to the next gene
                gene_index += 1
            else:
                # Hide the plot
                axes[i * n_cols + j].axis('off')

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

def get_3d_rotation_matrix(quaternion):
    # Quaternion has scalar as last term [q1, q2, q3, q0], return a 4x4 rotation matrix
    q1, q2, q3, q0 = quaternion

    # Compute the elements of the rotation matrix
    r00 = 1 - 2 * (q2**2 + q3**2)
    r01 = 2 * (q1*q2 - q0*q3)
    r02 = 2 * (q1*q3 + q0*q2)
    r10 = 2 * (q1*q2 + q0*q3)
    r11 = 1 - 2 * (q1**2 + q3**2)
    r12 = 2 * (q2*q3 - q0*q1)
    r20 = 2 * (q1*q3 - q0*q2)
    r21 = 2 * (q2*q3 + q0*q1)
    r22 = 1 - 2 * (q1**2 + q2**2)

    # Create the 4x4 rotation matrix
    rotation_matrix = np.array([
        [r00, r01, r02, 0],
        [r10, r11, r12, 0],
        [r20, r21, r22, 0],
        [0,   0,   0,   1]
    ])

    return rotation_matrix

def render_3d_frame(
    filepath, 
    virtual_creature, 
    override_vertices=[],
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

    # Make the plot and use gridspec to set up the following axes
    fig = plt.figure(figsize=(12, 10))

    # We'll have the 3d view large and on the left, taking up
    # 3x3 grid cells, and then the ortho views on the right, 
    # each taking up 1x1 for a total of 3x1
    gs = gridspec.GridSpec(4, 4, figure=fig)

    # Make a 2x3 grid of 3d subplots
    axes = [
        # overall view
        fig.add_subplot(gs[:3,:], projection='3d'), 

        # ortho views
        fig.add_subplot(gs[3,1], projection='3d'), 
        fig.add_subplot(gs[3,2], projection='3d'), 
        fig.add_subplot(gs[3,3], projection='3d'), 

        # closeup view
        fig.add_subplot(gs[3,0], projection='3d'),
    ]
    axes[1].set_proj_type('ortho')
    axes[2].set_proj_type('ortho')
    axes[3].set_proj_type('ortho')

    # Ensure the folder exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Render to a given axes
    def render_to_axes(ax, azim, elev, roll, zoom, hide_labels, closeup, tracking):
        """
        Closeup means we want to render the creature close up, without any transformations or past positions
        Tracking means we want to render the creature with the past positions just focused in on the creature
        """
        nonlocal extents
        
        # Get the vertices and faces and show in matplotlib
        vertices, faces = virtual_creature.get_mesh_verts_and_faces(current_time_s)
        if len(override_vertices) > 0:
            vertices = override_vertices

        # Need to apply the translation and rotation to the vertices
        # before rendering. Position is x,y,z and rotation is euler
        # angles in radians about x,y,z axes
        position = virtual_creature.position_xyz
        rotation = virtual_creature.quaternions
        # For the close up we don't want to do the transformation
        if closeup:
            transformation = np.eye(4)
        else:
            transformation = get_3d_translation_matrix(position) @ get_3d_rotation_matrix(rotation)
        # Do the transformation in 4d and then back to 3d
        vertices = [(transformation @ np.array([*v, 1]))[:3] for v in vertices]

        # We might render as a mesh
        if render_as_mesh:            

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
                    0, 0, 2,
                    color='red',
                )
                # Scatter with an x
                ax.scatter([cog_world_position[0]], [cog_world_position[1]], [cog_world_position[2]], color='red', marker='x')

        else:
            # Just render a vertex at the given position
            ax.scatter([position[0]], [position[1]], [position[2]], color='black', marker='o')
        
        # Edit the axes so that +x is forward, +y is right, +z is down
        if hide_labels:
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_zlabel('')
            # Get rid of the numbers, but not the ticks
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
        else:
            ax.set_xlabel("x (-back/+forward)")
            ax.set_ylabel("y (+right/-left)")
            ax.set_zlabel("z (+down/-up)")

        # Set axis limits so that everything is visible, unless we
        # were given extents in which case just use those
        if extents == [] or closeup or tracking:
            extents = [
                (min([v[0] for v in vertices]), max([v[0] for v in vertices])),
                (min([v[1] for v in vertices]), max([v[1] for v in vertices])),
                (min([v[2] for v in vertices]), max([v[2] for v in vertices]))
            ]

            def modify_extent_if_too_close(extent):
                epsilon = 0.05
                if abs(extent[0] - extent[1]) < epsilon:
                    return (extent[0] - epsilon, extent[1] + epsilon)
                return extent
            extents = [modify_extent_if_too_close(extent) for extent in extents]

            # The biggest of the extents becomes the size for all extents so that we have 
            # a cube, 1:1:1 aspect ratio
            biggest_range_index = np.argmax([extent[1] - extent[0] for extent in extents])
            offset_from_center = 0.5 * (extents[biggest_range_index][1] - extents[biggest_range_index][0])

            # Scale up bounds by scale factor
            if tracking:
                sf = 2
            else:
                sf = 1.2
            offset_from_center *= sf            

            extent_means = [0.5 * (extent[1] + extent[0]) for extent in extents]
            extents = [(mean - offset_from_center, mean + offset_from_center) for mean in extent_means]

            # Set up the extents
            # ax.set_zticks([])
            # ax.set_xticks([])
            # ax.set_yticks([])

        # Set the bounding box for the render
        ax.set_xlim3d(*extents[0])
        ax.set_ylim3d(*extents[1])
        ax.set_zlim3d(*extents[2]) 
        # Compute ratios if given
        size_x = extents[0][1] - extents[0][0]
        size_y = extents[1][1] - extents[1][0]
        size_z = extents[2][1] - extents[2][0]
        ax.set_box_aspect([size_x, size_y, size_z], zoom=zoom) 

        # Set azimuth elevation and roll on plot
        ax.view_init(azim=azim, elev=elev, roll=roll)

        # Render previous positions as a line if given
        if len(past_3d_positions) > 0 and not closeup:
            # Get the x,y,z components of the past positions
            x = [p[0] for p in past_3d_positions]
            y = [p[1] for p in past_3d_positions]
            z = [p[2] for p in past_3d_positions]

            # Plot the line
            ax.plot(x, y, z, color='black', alpha=0.5)

    # Render to the axes
    render_to_axes(axes[0], azim=-115, elev=-165, roll=0, zoom=1.1, hide_labels=False, closeup=False, tracking=False)
    # Behind view
    render_to_axes(axes[1], azim=0, elev=180, roll=0, zoom=1.5, hide_labels=True, closeup=False, tracking=True)
    # Right side view
    render_to_axes(axes[2], azim=-90, elev=180, roll=0, zoom=1.5, hide_labels=True, closeup=False, tracking=True)
    # Top down
    render_to_axes(axes[3], azim=-90, elev=-90, roll=0, zoom=1.5, hide_labels=True, closeup=False, tracking=True)
    # Close up view
    render_to_axes(axes[4], azim=-115, elev=-165, roll=0, zoom=1.1, hide_labels=True, closeup=True, tracking=False)

    # Set titles
    axes[0].set_title("3D view")
    axes[1].set_title("Behind view")
    axes[2].set_title("Right side view")
    axes[3].set_title("Top down view")
    axes[4].set_title("Close up view")

    # Render the time information if given in the top left
    if current_time_s != -1 and total_time_s != -1:
        def plot_text_y_down(y, text):
            axes[0].text2D(0, y, text, transform=axes[0].transAxes, fontsize=12, color='black', ha='right')
        plot_text_y_down(1 - 0.05, f"t={current_time_s:.4f} s")
        plot_text_y_down(1 - 0.1, f"T={total_time_s:.4f} s")   

    #plt.show()

    # Save the plot
    plt.tight_layout()
    plt.savefig(filepath, dpi=600)
    plt.close()

def sequence_frames_to_video(filepaths_in, filepath_out, fps=25):
    # Use moviepy to accomplish this
    clip = mpy.ImageSequenceClip(filepaths_in, fps=fps)
    # Write the videofilepaths_in
    clip.write_videofile(filepath_out, fps=int(fps), threads=math.ceil(os.cpu_count() * 0.25), bitrate='2M')

def render_simulation_of_creature(log_folder, creature, fps=25):
    # Make a copy of the creature and reset its state
    creature = copy.deepcopy(creature)
    creature.reset_state()
    
    # Run the creature for some time
    t = 0
    fps = 10
    # We don't want to render every step of the simulation
    # so we instead playback at this target fps^
    num_sim_steps = int(globals.SIMULATION_T / globals.DT)
    num_render_steps = int(globals.SIMULATION_T * fps)
    # Log the following for rendering frames
    creature_at_time = []
    running_state_trajectory = []
    state_trajectory_at_time = []
    times = []

    # Do the simulation
    # TODO this should come from fitness.py
    for i in tqdm(range(num_sim_steps), desc=f"Running forward dynamics to generate video, num_sim_steps={num_sim_steps}"):

        # Run the dynamics forward
        forward_step(creature, t, dt=globals.DT)

        # Get the state vector
        state_vector = creature.get_state_vector()
        running_state_trajectory.append(state_vector)
        
        # Log everyrthing over time
        creature_at_time.append(copy.deepcopy(creature))
        state_trajectory_at_time.append(copy.deepcopy(running_state_trajectory))
        times.append(t)

        t += globals.DT

    # Now we know the extents
    extents = [
        (min([state[0] for state in running_state_trajectory]), max([state[0] for state in running_state_trajectory])),
        (min([state[1] for state in running_state_trajectory]), max([state[1] for state in running_state_trajectory])),
        (min([state[2] for state in running_state_trajectory]), max([state[2] for state in running_state_trajectory])),
    ]
    wingspan = creature.chromosome.wingspan
    # If the wingspan is bigger than the y extent, then make the y extent bigger
    if wingspan > (extents[1][1] - extents[1][0]):
        extents[1] = (-wingspan/2, wingspan/2)
    # Add a buffer 
    scale_factor = 1.5
    extents = [ (scale_factor * extent[0], scale_factor * extent[1]) for extent in extents]

    # Do the rendering of the frames
    filepaths = []
    for r in tqdm(range(num_render_steps), desc=f"Rendering frames for video, num_render_steps={num_render_steps}"):
        # Get the closest index i in the simulation to the time i / fps
        i = int(r * num_sim_steps / num_render_steps)
        filepath = f"{log_folder}/frames_simulation/{r}.png"
        render_3d_frame(
            filepath,
            creature_at_time[i],
            extents=extents,
            past_3d_positions=state_trajectory_at_time[i],
            current_time_s=times[i],
            total_time_s=globals.SIMULATION_T
        )
        filepaths.append(filepath)

    # Plot a video of the creature from the frames
    sequence_frames_to_video(
        filepaths,
        f"{log_folder}/simulation.mp4",
        fps=fps
    )

def render_evolution_of_creatures(log_folder, filepaths_virtual_creatures):
    """
    Given a list of filepaths to virtual creature pickles, 
    render the evolution of the creatures (i.e., render a video
    of the creatures evolving over time)

    This will work like so:
    - Load the virtual creatures from the filepaths as keyframes
    - Interpolate between the keyframes to get the inbetween frames
    - Render the frames to a video at fps=25
    """

    # Load the pickle files
    creatures = []
    for filepath in filepaths_virtual_creatures:
        with open(filepath, 'rb') as f:
            creatures.append(pickle.load(f))
    # Reset the state of the creatures,
    # And remove any rotational offset
    for creature in creatures:
        creature.reset_state()
        creature.update_state(
            position_xyz=np.zeros(3),
            velocity_xyz=np.array([10.0, 0.0, 0.0]),
            acceleration_xyz=np.zeros(3),
            quaternions=np.array([0.0, 0.0, 0.0, 1.0]),
            angular_velocity=np.zeros(3),
            # This is key to making nice plots
            wing_angle_left=0,
            wing_angle_right=0
        )
    
    # Get the vertices and faces for each creature
    # At the keyframes
    vertices_list = []
    faces_list = []
    for creature in creatures:
        # Get what the creature looks like at time 0
        vertices, faces = creature.get_mesh_verts_and_faces(0)
        vertices_list.append(vertices)
        faces_list.append(faces)
    
    # We can just use the same faces for all creatures
    # because they won't change
    faces = faces_list[0]

    # Each vertices list is a list of vertices for each creature
    # We want to interpolate between these
    def interpolate_vertices(v1, v2, t):
        """
        Interpolate between two sets of vertices
        t is a value between 0 and 1
        """
        v1 = np.array(v1)
        v2 = np.array(v2)
        return (1 - t) * v1 + t * v2

    # Now how many frames do we want?
    video_length_s_per_creature = 1
    fps = 25
    num_frames_per_creature = int(video_length_s_per_creature * fps)

    # Get the max wingspans from the creatures
    max_y_extent = max([max([v[1] for v in vertices]) for vertices in vertices_list])
    min_y_extent = min([min([v[1] for v in vertices]) for vertices in vertices_list])
    middle_y = 0.5 * (max_y_extent + min_y_extent)
    range_y = max_y_extent - min_y_extent
    extent = (middle_y - 1.5 * range_y, middle_y + 1.5 * range_y)

    # Create the frames
    frame_filepaths = []
    for i in tqdm(range(len(creatures) - 1), desc="Rendering frames for evolution video"):
        initial_vertices = vertices_list[i]
        final_vertices   = vertices_list[i + 1]
        for j in range(num_frames_per_creature):
            t = j / num_frames_per_creature
            vertices = interpolate_vertices(initial_vertices, final_vertices, t)
            filepath = f"{log_folder}/frames_evolution/{i*num_frames_per_creature + j}.png"
            render_3d_frame(
                filepath,
                creature,
                override_vertices=vertices,
                render_as_mesh=True,
                if_mesh_render_verts=False,
                if_mesh_render_cog=False,
                extents=[extent, extent, extent],
                past_3d_positions=[],
                current_time_s=-1,
                total_time_s=-1,
            )
            frame_filepaths.append(filepath)

    # Pausea t the start and end for 1 second
    num_frames_pause = int(fps * 1)
    # Copy the first frame
    for i in range(num_frames_pause):
        frame_filepaths.insert(0, frame_filepaths[0])
    # Copy the last frame
    for i in range(num_frames_pause):
        frame_filepaths.append(frame_filepaths[-1])

    # Create the video
    sequence_frames_to_video(
        frame_filepaths,
        f"{log_folder}/evolution.mp4",
        fps=fps
    )