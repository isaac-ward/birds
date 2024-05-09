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
            velocity_xyz=np.zeros(3),
            acceleration_xyz=np.zeros(3),
            rotation_xyz=np.zeros(3),
            angular_velocity=np.zeros(3),
            wing_angle=np.zeros(3)
        )

    def mutate(self, mutation_rate):
        """
        Mutates the chromosome of the virtual creature
        with a given mutation rate
        """
        # TODO 

    def crossover(self, other):
        """
        Crosses over the chromosome of the virtual creature
        with another virtual creature
        """
        # TODO

    def render(self):
        """
        Renders the virtual creature as an image for quick
        visualization
        """
        # TODO: irw/aditi

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
        s += f"r' = {self.angular_velocity}\n"
        s += f"wa = {self.wing_angle}\n"
        s += "\n"
        # Write out the chromosome
        s += "chromosome:\n"
        s += str(self.chromosome)
        s += "----------------\n"
        return s