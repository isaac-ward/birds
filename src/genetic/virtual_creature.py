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
            wing_angle=0.0873
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
        s += f"r  = {self.rotation_xyz}\n"
        s += f"ω  = {self.angular_velocity}\n"
        s += f"wa = {self.wing_angle}\n"
        s += "\n"
        # Write out the chromosome
        s += "chromosome:\n"
        s += str(self.chromosome)
        s += "----------------\n"
        return s