import numpy as np

from phi.physics.world import World
from phi.physics import Physics
from phi.physics.collective import CollectivePhysics
from phi import math, struct
from phi.struct.functions import mappable
from .util import placeholder


def tf_bake_graph(world, session):
    # --- Build placeholder state ---
    shape = world.state.shape
    dtype = _32_bit(math.types(world.state))
    state_in = placeholder(shape, dtype=dtype)
    dt = placeholder(())
    # --- Build graph ---
    state_out = world.physics.step(state_in, dt=dt)
    world.physics = BakedWorldPhysics(world.physics, session, state_in, state_out, dt)
    for name, sysstate in world.state.states.items():
        sysstate_in = state_in[name]
        sysstate_out = state_out[name]
        baked_physics = BakedPhysics(session, sysstate_in, sysstate_out, dt)
        world.physics.add(name, baked_physics)


def tf_bake_subgraph(tracker, session):
    tfworld = World()
    tfworld.add(tracker.state)
    # --- Build placeholder state ---
    dtype = _32_bit(math.types(tracker.state))
    shape = tracker.state.shape
    state_in = placeholder(shape, dtype=dtype)
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

    def __init__(self, physics, session, state_in, state_out, dt):
        CollectivePhysics.__init__(self)
        self._physics = physics.physics
        self.state_in = state_in
        self.state_out = state_out
        self.session = session
        self.dt = dt

    def step(self, state_collection, dt=1.0, **dependent_states):
        result = self.session.run(self.state_out, {self.state_in: state_collection, self.dt: dt})
        return result


@mappable(item_condition=None, unsafe_context=True)
def _32_bit(dtype):
    if dtype == np.float64:
        return np.float32
    if dtype == np.int64:
        return np.int32
    return dtype
