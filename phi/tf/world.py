from phi.physics.world import *
from .util import *


def tf_bake_graph(world, session):
    state_in = placeholder_like(world.state)
    dt = tf.placeholder(tf.float32, ())
    state_out = world.physics.step(state_in, dt=dt)
    world.physics = BakedWorldPhysics(world.physics, session, state_in, state_out, dt)
    for sysstate in world.state.states:
        sysstate_in = state_in[sysstate.trajectorykey]
        sysstate_out = state_out[sysstate.trajectorykey]
        baked_physics = BakedPhysics(session, sysstate_in, sysstate_out, dt)
        world.physics.add(sysstate.trajectorykey, baked_physics)


def tf_bake_subgraph(tracker, session):
    tfworld = World()
    tfworld.add(tracker.state)
    state_in = placeholder_like(tracker.state)
    dt = tf.placeholder(tf.float32, ())
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
        self._physics = physics._physics
        self.state_in = state_in
        self.state_out = state_out
        self.session = session
        self.dt = dt

    def step(self, collectivestate, dt=1.0, **dependent_states):
        result = self.session.run(self.state_out, {self.state_in: collectivestate, self.dt: dt})
        return result
