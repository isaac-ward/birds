import numpy as np 
import random

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
        # here (note that these are not genes! they are the state of the
        # creature in the world, not the parameters that define the creature)

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
        rotation_xyz,
        wing_angle,
        angular_velocity
    ):
        """
        Updates the state of the virtual creature by one step
        """
        # +x is forward, +y is up, +z is right
        self.position_xyz = position_xyz         # m
        self.velocity_xyz = velocity_xyz         # m/s
        self.acceleration_xyz = acceleration_xyz # m/s^2
        self.rotation_xyz = rotation_xyz         # rad
        self.angular_velocity = angular_velocity # rad/s
        self.wing_angle = wing_angle             # rad (angle between wing and bird body)
    
    def reset_state(self):
        """
        Resets the state of the virtual creature
        """
        self.update_state(
            position_xyz=np.zeros(3),
            velocity_xyz=np.array([10, 0, 0]),
            acceleration_xyz=np.zeros(3),
            rotation_xyz=np.zeros(3),
            angular_velocity=np.zeros(3),
            wing_angle=0
        )

    def get_state_vector(self):
        """
        Returns the state of the virtual creature as a vector
        """
        return np.concatenate([
            self.position_xyz,
            self.velocity_xyz,
            self.acceleration_xyz,
            self.rotation_xyz,
            self.angular_velocity,
            np.array([self.wing_angle])
        ])

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
        
        #still in progress not able to access individual genes  
        #Gaussian mutation by adding zero mean Gaussian Noise (Alg 9.9)
        sigma = mutation_rate
        mutated_wingspan = np.clip(self.chromosome.wingspan + np.random.normal(0, sigma), 0.0, 10.0)
        mutated_norm_wrist_position = np.clip(self.chromosome.norm_wrist_position + np.random.normal(0, sigma), 0.0, 1.0)
        mutated_wing_root_chord = np.clip(self.chromosome.wing_root_chord + np.random.normal(0, sigma), 0.0, 1.0)
        mutated_taper_armwing = np.clip(self.chromosome.taper_armwing + np.random.normal(0, sigma), 0.0, 1.0)
        mutated_taper_handwing = np.clip(self.chromosome.taper_handwing + np.random.normal(0, sigma), 0.0, 1.0)
        mutated_norm_COG_position = np.clip(self.chromosome.norm_COG_position + np.random.normal(0, sigma), 0.0, 1.0)
        mutated_airfoil_armwing = np.clip(self.chromosome.airfoil_armwing + np.random.normal(0, sigma), 0.0, 1.0)
        mutated_airfoil_handwing = np.clip(self.chromosome.airfoil_handwing + np.random.normal(0, sigma), 0.0, 1.0)
        new_chromosome = Chromosome(
            wingspan=mutated_wingspan,
            norm_wrist_position=mutated_norm_wrist_position,
            wing_root_chord=mutated_wing_root_chord,
            taper_armwing=mutated_taper_armwing,
            taper_handwing=mutated_taper_handwing,
            norm_COG_position=mutated_norm_COG_position,
            airfoil_armwing=mutated_airfoil_armwing,
            airfoil_handwing=mutated_airfoil_handwing,
         )
        # gene_definitions = self.chromosome.get_gene_definitions()
        # new_vector = np.clip(
        #     self.chromosome.vector + np.random.normal(0, sigma, self.chromosome.vector.shape),
        #     [gene.min_val for gene in CHROMOSOME_DEFINITION],
        #     [gene.max_val for gene in CHROMOSOME_DEFINITION]
        # )
        # new_chromosome = Chromosome(new_vector)
        return VirtualCreature(new_chromosome)
        #pass

    def crossover(self, other):
        """
        Crosses over the chromosome of the virtual creature
        with another virtual creature
        """
       
        # from textbook single point cross over and will choose a random index to split and stitch the chromosome 
        crossover_index = random.randint(0, 5) # There are 6 traits, indexed 0 to 5
        # new_chromosome = Chromosome(
        wingspan=self.chromosome.wingspan if crossover_index >= 0 else other.chromosome.wingspan,
        norm_wrist_position=self.chromosome.norm_wrist_position if crossover_index >= 1 else other.chromosome.norm_wrist_position,
        wing_root_chord=self.chromosome.wing_root_chord if crossover_index >= 2 else other.chromosome.wing_root_chord,
        taper_armwing=self.chromosome.taper_armwing if crossover_index >= 3 else other.chromosome.taper_armwing,
        taper_handwing=self.chromosome.taper_handwing if crossover_index >= 4 else other.chromosome.taper_handwing,
        norm_COG_position=self.chromosome.norm_COG_position if crossover_index >= 5 else other.chromosome.norm_COG_position,
        airfoil_armwing=self.chromosome.airfoil_armwing if crossover_index >= 3 else other.chromosome.airfoil_armwing,
        airfoil_handwing=self.chromosome.airfoil_handwing if crossover_index >= 4 else other.chromosome.airfoil_handwing,
        # )
        
        # crossover_point = random.randint(1, len(self.chromosome.vector) - 1)
        # new_vector = np.concatenate((
        #     self.chromosome.vector[:crossover_point],
        #     other.chromosome.vector[crossover_point:]
        # ))
        # new_chromosome = Chromosome(new_vector)
        return VirtualCreature(wingspan,norm_wrist_position,wing_root_chord,taper_armwing,taper_handwing,norm_COG_position,airfoil_armwing,airfoil_handwing)

        

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
        s += f"r  = {self.rotation_xyz}\n"
        s += f"ω  = {self.angular_velocity}\n"
        s += f"wa = {self.wing_angle}\n"
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
    
