import numpy as np 
import random
import copy

import globals
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

        # Set the mass and basis parameters
        self.get_mass_parameters()
        self.get_basis_functions()

        # Kinematic parameters (describe it in the world)
        self.reset_state()

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
        # +x is forward, +y is right, +z is down
        self.position_xyz = position_xyz         # m
        self.velocity_xyz = velocity_xyz         # m/s
        self.acceleration_xyz = acceleration_xyz # m/s^2
        self.quaternions = quaternions           # scalar is last term [q1, q2, q3, q0]
        self.angular_velocity = angular_velocity # rad/s (pqr in bird frame)

        # Wing angles
        self.wing_angle_left = wing_angle_left
        self.wing_angle_right = wing_angle_right
    
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
            wing_angle_left=self.calc_wing_angle(0, "left"),
            wing_angle_right=self.calc_wing_angle(0, "right")
        )

    def get_state_vector(self):
        """
        Returns the state of the virtual creature as a vector
        """
        vector = np.concatenate([
            self.position_xyz,
            self.velocity_xyz,
            self.acceleration_xyz,
            self.quaternions,
            self.angular_velocity,
            [self.wing_angle_left, self.wing_angle_right]
        ])
        return vector
    
    def get_dynamics_state_vector(self):
        """
        Returns the state used in dynamics of the virtual creature as a vector
        """
        vector = np.concatenate([
            self.position_xyz,
            self.velocity_xyz,
            self.quaternions,
            self.angular_velocity,
        ])
        return vector
    
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

        # Center of lift
        chord_avg_aw = (1+taper_armwing)/2 * wing_root_chord
        chord_avg_hw = (1+taper_handwing)/2 * taper_armwing * wing_root_chord
        chord_avg = (norm_wrist_position*chord_avg_aw + (1-norm_wrist_position)*chord_avg_hw)
        x_COL_position = 1/4 * wing_root_chord

        z_COL_position = wingspan/4

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
        Iz = bird_mass*(span_avg**2 + wing_root_chord**2)/12
        Iy = bird_mass*(wing_root_chord**2)/12
        I_mat = np.diag([Ix, Iy, Iz])

        self.AR_aw, self.AR_hw, self.area_aw, self.area_hw = AR_aw, AR_hw, area_aw, area_hw
        self.COG_position, self.x_COL_position, self.z_COL_position = COG_position, x_COL_position, z_COL_position
        self.bird_mass, self.I_mat = bird_mass, I_mat
    
    def get_basis_functions(self):
        left_basis = [self.chromosome.basis_left_const, self.chromosome.basis_left_poly1, self.chromosome.basis_left_poly2,
                      self.chromosome.basis_left_poly3, self.chromosome.basis_left_poly4, self.chromosome.basis_left_poly5,
                      self.chromosome.basis_left_sinamp1, self.chromosome.basis_left_sinfreq1, self.chromosome.basis_left_sinamp2,
                      self.chromosome.basis_left_sinfreq2, self.chromosome.basis_left_sawtooth, self.chromosome.basis_left_expamp1,
                      self.chromosome.basis_left_exppwr1, self.chromosome.basis_left_expamp2, self.chromosome.basis_left_exppwr2]
        right_basis = [self.chromosome.basis_right_const, self.chromosome.basis_right_poly1, self.chromosome.basis_right_poly2,
                      self.chromosome.basis_right_poly3, self.chromosome.basis_right_poly4, self.chromosome.basis_right_poly5,
                      self.chromosome.basis_right_sinamp1, self.chromosome.basis_right_sinfreq1, self.chromosome.basis_right_sinamp2,
                      self.chromosome.basis_right_sinfreq2, self.chromosome.basis_right_sawtooth, self.chromosome.basis_right_expamp1,
                      self.chromosome.basis_right_exppwr1, self.chromosome.basis_right_expamp2, self.chromosome.basis_right_exppwr2]
        self.left_basis, self.right_basis = left_basis, right_basis

    def calc_wing_angle(self, t, wing_choice):
        if wing_choice == "left":
            basis = self.left_basis
        elif wing_choice == "right":
            basis = self.right_basis
        else:
            raise TypeError("Input either left or right for wing_choice")
        
        # For readability let's pull out all the basis functions values
        const, poly1, poly2, poly3, poly4, poly5, sinamp1, sinfreq1, sinamp2, sinfreq2, sawtooth, expamp1, exppwr1, expamp2, exppwr2 = basis
        
        # Compute the value for the wing angle
        polynomial  = const + poly1*t + poly2*t**2 + poly3*t**3 + poly4*t**4 + poly5*t**5
        sinusoid    = sinamp1*np.sin(sinfreq1*t) + sinamp2*np.sin(sinfreq2*t)
        # TODO add rate of increase (e.g. X*sawtooth)
        # sawtooth    = t % sawtooth
        exponential = expamp1*np.exp(exppwr1*t) + expamp2*np.exp(exppwr2*t)
        total = polynomial + sinusoid + sawtooth + exponential

        # TODO this can allow discontinuities

        # Modulo it into the range [-pi, pi]
        result = (total + np.pi) % (2*np.pi) - np.pi

        # Now scale it into the range [-25deg, +25deg]
        rad25deg = 25 * np.pi / 180
        result = result * rad25deg / np.pi

        # print(f"poly: {polynomial:.2f}, sin: {sinusoid:.2f}, saw: {sawtooth:.2f}, exp: {exponential:.2f}, total: {total:.2f}, result: {result:.2f}")

        return result        

    @staticmethod
    def get_state_vector_labels():
        """
        Returns the labels for the state vector
        """
        
        # label rotations with roll, yaw, pitch
        # everything is metres, seconds, radians
        
        return [
            "px (+forward, -backward)", "py (+right, -left)", "pz (+down, -up)",
            "vx", "vy", "vz",
            "ax", "ay", "az",
            #"rx (roll)", "ry (yaw)", "rz (pitch)",
            "qx", "qy", "qz", "qw",
            "ωx", "ωy", "ωz",
            "wing_angle_left", "wing_angle_right"
        ]
    
    def crossover(self, other):
        """
        Crosses over the chromosome of the virtual creature
        with another virtual creature
        """
        # from textbook single point cross over and will choose a random index to split and stitch the chromosome 
        return VirtualCreature(self.chromosome.crossover(other.chromosome))

    def mutate(self):
        """
        Mutates the chromosome of the virtual creature
        with a given mutation rate
        """
        #Gaussian mutation by adding zero mean Gaussian Noise (Alg 9.9)
        return VirtualCreature(self.chromosome)

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
        s += "\n"
        # Write out the chromosome
        s += "chromosome:\n"
        s += str(self.chromosome)
        s += "----------------\n"
        return s
    
    def get_mesh_verts_and_faces(self, t):
        
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

        # Basically going to define the origin as the back center of the bird

        # Construct the right armwing, recall that:
        # +x is forward, +y is right, +z is down
        right_armwing_vertices.extend([
            # back at root
            (0, 0, 0), 
            # back at tip
            (0, 0.5 * wingspan * norm_wrist_position, 0), 
            # front at tip
            (wing_root_chord * taper_armwing, 0.5 * wingspan * norm_wrist_position, 0),
            # front at root
            (wing_root_chord, 0, 0)
        ])

        # Now the right handwing
        right_handwing_vertices.extend([
            # back at root (same as back at tip of armwing)
            (0, 0.5 * wingspan * norm_wrist_position, 0),
            # back at tip
            (0, 0.5 * wingspan, 0),
            # front at tip
            (wing_root_chord * taper_handwing, 0.5 * wingspan, 0),
            # front at root (same as front at tip of armwing)
            (wing_root_chord * taper_armwing, 0.5 * wingspan * norm_wrist_position, 0)
        ])

        # The left wing is just the right handwing reflected across y
        left_armwing_vertices = [(x, -y, z) for x, y, z in right_armwing_vertices]
        left_handwing_vertices = [(x, -y, z) for x, y, z in right_handwing_vertices]

        # Group into left and right
        left_wing_vertices  = np.array(left_armwing_vertices + left_handwing_vertices)
        right_wing_vertices = np.array(right_armwing_vertices + right_handwing_vertices)

        # We need to move the vertices to the COG position
        # translation_cog = np.array([COG_position * wing_root_chord, 0, 0])
        # left_wing_vertices  += translation_cog
        # right_wing_vertices += translation_cog

        # We need to rotate the vertices in each wing based on the wing angle
        # state parameter, which rotates them in the xz plane (i.e. angle of attack)
        def get_wing_angle_rot(angle):
            # Negative because we want to rotate the other
            # way (positive wing angle should rotate up)
            angle = -angle 
            return np.array([
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)]
            ])

        left_wing_vertices  = left_wing_vertices  @ get_wing_angle_rot(self.calc_wing_angle(t, "left"))
        right_wing_vertices = right_wing_vertices @ get_wing_angle_rot(self.calc_wing_angle(t, "right"))

        # Now we can construct the full vertices and faces
        vertices.extend(left_wing_vertices)
        vertices.extend(right_wing_vertices)
        
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
    
    def __getstate__(self):
        return self.__dict__
    def __setstate__(self, state):
        self.__dict__.update(state)
