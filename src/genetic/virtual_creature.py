import numpy as np 

from genetic.chromosome import Chromosome

class VirtualCreature:
    """
    Describes a virtual creature with a chromosome. A virtual
    creature furthers this by being an instantiation of a creature
    in a simulation environment (e.g. it has kinematic parameters).
    Also contains methods for mutating and crossing over
    individuals, and other utility methods.
    """

    def __init__(
        self,
        chromosome,
    ):
        # A chromosome directly describes a virtual creature
        self.chromosome = chromosome

        # Kinematic parameters (describe it in the world)
        self.reset_state()

        # TODO: Kusal please feel free to add other kinematic parameters
        self.get_mass_parameters()

    @classmethod
    def random_init(cls):
        """
        Initializes a virtual creature with a random chromosome
        """
        chromosome = Chromosome.random_init()
        return cls(chromosome)
    
    def update_state(
        self,
        position_xyz,
        velocity_xyz,
        acceleration_xyz,
        quaternions,
        angular_velocity,
        wing_angle_left,
        wing_angle_right
    ):
        """
        Updates the state of the virtual creature by one step
        """
        # +x is forward, +y is up, +z is right
        self.position_xyz = position_xyz         # m
        self.velocity_xyz = velocity_xyz         # m/s
        self.acceleration_xyz = acceleration_xyz # m/s^2
        self.quaternions = quaternions           # scalar is last term [q1, q2, q3, q0]
        self.angular_velocity = angular_velocity # rad/s (pqr in bird frame)
        self.wing_angle_left = wing_angle_left   # rad (angle between left wing and bird body)
        self.wing_angle_right = wing_angle_right # rad (angle between right wing and bird body)
    
    def reset_state(self):
        """
        Resets the state of the virtual creature
        """
        self.update_state(
            position_xyz=np.zeros(3),
            velocity_xyz=np.array([10.0, 0.0, 0.0]),
            acceleration_xyz=np.zeros(3),
            quaternions=np.array([0.0, 0.0, 0.0, 1.0]),
            angular_velocity=np.zeros(3),
            wing_angle_left=0.035,
            wing_angle_right=0.0175
        )

    def get_state_vector(self):
        """
        Returns the state of the virtual creature as a vector
        """
        return np.concatenate([
            self.position_xyz,
            self.velocity_xyz,
            self.acceleration_xyz,
            self.quaternions,
            self.angular_velocity,
            np.array([self.wing_angle_left]),
            np.array([self.wing_angle_right])
        ])
    
    def get_mass_parameters(self):
        # Get chromosome data
        wingspan, norm_wrist_position, wing_root_chord = self.chromosome.wingspan, self.chromosome.norm_wrist_position, self.chromosome.wing_root_chord
        taper_armwing, taper_handwing, norm_COG_position = self.chromosome.taper_armwing, self.chromosome.taper_handwing, self.chromosome.norm_COG_position
        airfoil_armwing, airfoil_handwing = self.chromosome.airfoil_armwing, self.chromosome.airfoil_handwing
        bird_density = globals.BIRD_DENSITY
        COG_position = norm_COG_position * wing_root_chord

        airfoil_armwing, airfoil_handwing = "NACA 0012", "NACA 0012" #NOTE: This will need to be changed

        # Wing characteristics
        span_aw = wingspan * norm_wrist_position
        span_hw = wingspan - span_aw
        area = lambda b, cr, gamma: (1+gamma)/2*cr*b
        area_aw = area(span_aw, wing_root_chord, taper_armwing)
        area_hw = area(span_hw, wing_root_chord*taper_armwing, taper_handwing)
        AR_aw = span_aw / area_aw
        AR_hw = span_hw / area_hw

        chord_avg_aw = (1+taper_armwing)/2 * wing_root_chord
        chord_avg_hw = (1+taper_handwing)/2 * taper_armwing * wing_root_chord
        COL_position = 1/4 * wing_root_chord

        # Wing volume
        chord_thickness_aw = globals.AIRFOIL_DATABASE[airfoil_armwing][-1]
        chord_thickness_hw = globals.AIRFOIL_DATABASE[airfoil_handwing][-1]
        wing_volume_aw = chord_avg_aw**2 * span_aw * chord_thickness_aw
        wing_volume_hw = chord_avg_hw**2 * span_hw * chord_thickness_hw
        bird_volume = wing_volume_aw + wing_volume_hw
        bird_mass = bird_volume * bird_density

        # Moment of Inertia (simplified)
        span_avg = (area_aw + area_hw) / wing_root_chord
        Ix = bird_mass*(span_avg**2)/12 + bird_mass*(COG_position-0.5*wing_root_chord)**2
        Iy = bird_mass*(span_avg**2 + wing_root_chord**2)/12
        Iz = bird_mass*(wing_root_chord**2)/12

        self.AR_aw, self.AR_hw, self.area_aw, self.area_hw = AR_aw, AR_hw, area_aw, area_hw
        self.COG_position, self.COL_position, self.bird_mass = COG_position, COL_position, bird_mass
        self.Ix, self.Iy, self.Iz = Ix, Iy, Iz

    @staticmethod
    def get_state_vector_labels():
        """
        Returns the labels for the state vector
        """
        
        # +x is forward, +y is up, +z is right
        # label rotations with roll, yaw, pitch
        # everything is metres, seconds, radians
        
        return [
            "px (+forward, -backward)", "py (+up, -down)", "pz (+right, -left)",
            "vx", "vy", "vz",
            "ax", "ay", "az",
            "rx (roll)", "ry (yaw)", "rz (pitch)",
            "ωx", "ωy", "ωz",
            "wing_angle"
        ]

    def mutate(self, mutation_rate):
        """
        Mutates the chromosome of the virtual creature
        with a given mutation rate
        """

        # Create a new chromosome with Chromosome(...)
        # and then return a new VirtualCreature with the new chromosome
        # like return VirtualCreature(new_chromosome)
        # TODO 
        # self.chromosome.cog_position
        pass

    def crossover(self, other):
        """
        Crosses over the chromosome of the virtual creature
        with another virtual creature
        """
        # TODO
        # self.chromosome.wingspan
        # other.chromosome.wingspan

        # again return a new VirtualCreature with the new chromosome
        # new_chromosome <- other.chromosome mixed with self.chromosome
        # like return VirtualCreature(new_chromosome)
        pass

    def __str__(self):
        """
        Returns a string representation of the virtual creature
        """
        s = ""
        # Write out the state
        s += "----------------\n"
        s += "state:\n"
        s += f"p  = {self.position_xyz}\n"
        s += f"v  = {self.velocity_xyz}\n"
        s += f"a  = {self.acceleration_xyz}\n"
        s += f"r  = {self.quaternions}\n"
        s += f"ω  = {self.angular_velocity}\n"
        s += f"wa = {self.wing_angle_left}\n"
        s += f"wa = {self.wing_angle_right}\n"
        s += "\n"
        # Write out the chromosome
        s += "chromosome:\n"
        s += str(self.chromosome)
        s += "----------------\n"
        return s
    
    def get_mesh_verts_and_faces(self):
        
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
            
        # All the data that we need is in the chromosome
        chromosome = self.chromosome
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
    
