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

def render_virtual_creature(filepath, virtual_creature):
    """
    Renders a virtual creature in 3D space. Basically need
    to use the chromosome information to create a 3d mesh
    representing the 3D structure of the bird
    """
    
    # These are the genes
    # Gene("wingspan", 0.0, 10.0),
    # Gene("norm_wrist_position", 0.0, 1.0),
    # Gene("wing_root_chord", 0.0, 1.0),
    # Gene("taper_armwing", 0.0, 1.0),
    # Gene("taper_handwing", 0.0, 1.0),
    # Gene("COG_position", 0.0, 1.0),
    # Gene("airfoil_armwing", 0.0, 1.0), #NOTE: Change later
    # Gene("airfoil_handwing", 0.0, 1.0) #NOTE: Change later

    # wingspan is the full length between wingtips (i.e. z axis length)
    # norm_wrist_position is the position of the from the root chord to the tip as a fraction 
    # wing_root_chord is the chord length at the root (i.e. x axis length)
    # taper_armwing is the length of the wing at the norm_wrist_position as a fraction of the root chord
    # taper_handwing is the length of the wing at the tip as a fraction of the length at the wrist
    # COG_position is the position of the center of gravity as a fraction of the root chord, FROM the front 
    # airfoils are just the types 

    # We can use the above information to create a 3D mesh of the bird
    def make_mesh(virtual_creature):
        # All the data that we need is in the chromosome
        chromosome = virtual_creature.chromosome
        wingspan = chromosome.wingspan
        norm_wrist_position = chromosome.norm_wrist_position
        wing_root_chord = chromosome.wing_root_chord
        taper_armwing = chromosome.taper_armwing
        taper_handwing = chromosome.taper_handwing
        norm_COG_position = chromosome.norm_COG_position
        COG_position = norm_COG_position * norm_wrist_position

        # Will represent the bird as a series of vertices and faces
        # (must be triangles)
        vertices = []
        faces = []

        # This version of the bird is essentially 4 quads, 2 for each wing, because
        # we have one either side of the wrist position

        # We  will construct about an 'origin' at the center, back of the craft
        # and then move it to the COG position afterwards
        right_handwing_vertices = [] # Distal part of arm
        right_armwing_vertices = [] # Proximal part of arm

        # Construct the right armwing, recall that:
        # +x is forward, +y is up, +z is right
        right_armwing_vertices.extend([
            # back at root
            (0, 0, 0), 
            # back at tip
            (0, 0, 0.5 * wingspan * norm_wrist_position), 
            # front at tip
            (wing_root_chord * taper_armwing, 0, 0.5 * wingspan * norm_wrist_position),
            # front at root
            (wing_root_chord, 0, 0)
        ])

        # Now the right handwing
        right_handwing_vertices.extend([
            # back at root (same as back at tip of armwing)
            (0, 0, 0.5 * wingspan * norm_wrist_position),
            # back at tip
            (0, 0, 0.5 * wingspan),
            # front at tip
            (wing_root_chord * taper_handwing, 0, 0.5 * wingspan),
            # front at root (same as front at tip of armwing)
            (wing_root_chord * taper_armwing, 0, 0.5 * wingspan * norm_wrist_position)
        ])

        # The left wing is just the right handwing reflected across z
        left_armwing_vertices = [(x, y, -z) for x, y, z in right_armwing_vertices]
        left_handwing_vertices = [(x, y, -z) for x, y, z in right_handwing_vertices]

        # Now we can construct the full vertices and faces
        vertices.extend(right_armwing_vertices)
        vertices.extend(right_handwing_vertices)
        vertices.extend(left_armwing_vertices)
        vertices.extend(left_handwing_vertices)
        
        def helper_quad_to_tri(quad):
            """
            Given a quad, returns two triangles
            """
            return [(quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])]

        # Right armwing
        faces.extend(helper_quad_to_tri([0, 1, 2, 3]))
        # Right handwing
        faces.extend(helper_quad_to_tri([4, 5, 6, 7]))
        # Left armwing
        faces.extend(helper_quad_to_tri([8, 9, 10, 11]))
        # Left handwing
        faces.extend(helper_quad_to_tri([12, 13, 14, 15]))

        return vertices, faces
    
    # Get the vertices and faces and show in matplotlib
    vertices, faces = make_mesh(virtual_creature)

    # Our convention is +x is forward, +y is up, +z is right
    # But mpl has +x is forward, +y is left, +z is up
    # TODO need to do something about this

    # Render the mesh with trisurf
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(
        [v[0] for v in vertices],
        [v[1] for v in vertices],
        [v[2] for v in vertices],
        triangles=faces,
        cmap='viridis',
        alpha=0.5,
        edgecolor='k'
    )

    # Annunciate the vertices with a 3d scatter plot
    # with little markers at each vertex
    ax.scatter(
        [v[0] for v in vertices],
        [v[1] for v in vertices],
        [v[2] for v in vertices],
        color='black',
        marker='.',
        s=0.2,
    )    

    # Draw a downwards pointing vector at the CoG position
    norm_COG_position = virtual_creature.chromosome.norm_COG_position
    COG_position = norm_COG_position * virtual_creature.chromosome.wing_root_chord
    cog_x = (1 - COG_position) * virtual_creature.chromosome.wing_root_chord
    # Ensure that this has an arrow head
    ax.quiver(
        # Start at the CoG position
        cog_x, 0, 0,
        # dXYZ components of the arrow
        0, -0.1, 0,
        color='red',
    )
    # Scatter with an x
    ax.scatter([cog_x], [0], [0], color='red', marker='x')

    # Edit the axes so that +x is forward, +y is up, +z is right
    ax.set_xlabel("+x forward")
    ax.set_ylabel("+y up")
    ax.set_zlabel("+z right")

    # Set axis limits so that everything is visible
    extents = [
        (min([v[0] for v in vertices]), max([v[0] for v in vertices])),
        (min([v[1] for v in vertices]), max([v[1] for v in vertices])),
        (min([v[2] for v in vertices]), max([v[2] for v in vertices]))
    ]
    ax.set_xlim3d(*extents[0])
    ax.set_xlim3d(*extents[1])
    ax.set_xlim3d(*extents[2]) 

    # Set aspect ratoi
    ax.set_box_aspect([1, 1, 5])

    # Set ticks to show wingspan
    ax.set_zticks([*extents[2]])
    # Hide x and y axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Set azimuth elevation and roll on plot
    rot = 30
    ax.view_init(azim=-90 - rot, elev=110, roll=-rot)

    #plt.show()

    # Save the plot
    plt.tight_layout()
    plt.savefig(filepath, dpi=600)
    plt.close()

def render_realtime():
    # TODO: irw + aditi to use pygame to create a real-time simulation 
    # visualization of genetic algorithm solution
    pass

def render_from_log():
    # TODO: irw + aditi to use pygame to create a visualization of
    # the genetic algorithm solution from a log file
    # TODO: irw + aditi to pull in blender scripts
    pass