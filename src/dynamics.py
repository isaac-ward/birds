import numpy as np 

# TODO: Kusal to implement how the state evolves at every step 
# starting with a single virtual creature in a simple aerodynamic
# environment/model
def forward_step(virtual_creature):
    """
    Takes a virtual creature and updates its state by one step based on
    it's current state and the aerodynamic environment
    """
    # +x is forward, +y is right, +z is up
    p, v, a, r = virtual_creature.position_xyz, virtual_creature.velocity_xyz, virtual_creature.acceleration_xyz, virtual_creature.rotation_xyz

    # TODO: Do all the funky physics here and compute new state
    # Can access the genes of the chromosome like this:
    wingspan = virtual_creature.chromosome.wingspan
    p = p + 0.1 * np.array([1, 0, 0])  # Move forward 0.1m
    # etc.

    # Finish by updating the state of the virtual creature
    virtual_creature.update_state(
        position_xyz=p,
        velocity_xyz=v,
        acceleration_xyz=a,
        rotation_xyz=r,
    )