import numpy as np

from phi import struct
from phi.physics.world import World
from phi.physics import Physics
from phi.physics.collective import CollectivePhysics
from phi.struct import VARIABLES
from phi.struct.functions import mappable
from .util import placeholder


def tf_bake_graph(world, session):
    # --- Build placeholder state ---
    with VARIABLES:
        state_in = placeholder(world.state.staticshape, dtype=world.state.dtype)
    dt = placeholder(())
    # --- Build graph ---
    state_out = world.physics.step(state_in, dt=dt)
    world.physics = BakedWorldPhysics(world.physics, session, state_in, state_out, dt, world=world)
    for name, sysstate in world.state.states.items():
        sysstate_in = state_in[name]
        sysstate_out = state_out[name]
        baked_physics = BakedPhysics(session, sysstate_in, sysstate_out, dt)
        world.physics.add(name, baked_physics)


def tf_bake_subgraph(tracker, session):
    tfworld = World()
    tfworld.add(tracker.state)
    # --- Build placeholder state ---
    with VARIABLES:
        state_in = placeholder(tracker.state.staticshape, dtype=tracker.state.dtype)
    dt = placeholder(())
    # --- Build graph ---
    state_out = tracker.world.physics.substep(state_in, tracker.world.state, dt)
    tracker.physics = BakedPhysics(session, state_in, state_out, dt)
    return tfworld


class BakedPhysics(Physics):

    def __init__(self, session, state_in, state_out, dt):
        Physics.__init__(self, {})
        self.state_in = state_in
        self.state_out = state_out
        self.session = session
        self.dt = dt

    def step(self, state, dt=1.0, **dependent_states):
        for key, value in dependent_states:
            assert not value, 'Baked subgraph can only be executed without dependencies'
        return self.session.run(self.state_out, {self.state_in: state, self.dt: dt})


class BakedWorldPhysics(CollectivePhysics):

    def __init__(self, physics, session, state_in, state_out, dt, world):
        CollectivePhysics.__init__(self)
        self._physics = physics.physics
        self.state_in = state_in
        self.state_out = state_out
        self.session = session
        self.dt = dt
        self.world = world

    def step(self, world_state, dt=1.0, **dependent_states):
        return self.run(self.state_out, dt, world_state)

    def run(self, fetches, dt=1.0, world_state=None):
        if world_state is None:
            world_state = self.world.state
        feed_dict = {self.state_in: world_state, self.dt: dt}
        return self.session.run(fetches, feed_dict)
