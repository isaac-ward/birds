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
    position_xyz = virtual_creature.position_xyz
    velocity_xyz = virtual_creature.velocity_xyz
    acceleration_xyz = virtual_creature.acceleration_xyz
    rotation_xyz = virtual_creature.rotation_xyz

    # TODO: Do all the funky physics here and compute new state
    # Can access the genes of the chromosome like this:
    wingspan = virtual_creature.wingspan

    # Finish by updating the state of the virtual creature
    virtual_creature.update_state(
        position_xyz=new_position_xyz,
        velocity_xyz=new_velocity_xyz,
        acceleration_xyz=new_acceleration_xyz,
        rotation_xyz=new_rotation_xyz,
    )