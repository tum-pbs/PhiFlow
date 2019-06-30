from phi.physics.world import *
from .util import *


def tf_bake_graph(session, world=world):
    tf_bake_subgraph(StateTracker(world, world.state.states[0].trajectorykey), session)  # TODO bake whole world


def tf_bake_subgraph(tracker, session):
    tfworld = World()
    tfworld.add(tracker.state)
    state_in = placeholder_like(tracker.state)
    dt = tf.placeholder(tf.float32, ())
    state_out = tracker.world.physics.substep(state_in, tracker.world.state, dt)
    tracker.physics = CallTFPhysics(session, state_in, state_out, dt)
    return tfworld


class CallTFPhysics(Physics):

    def __init__(self, session, state_in, state_out, dt):
        Physics.__init__(self, {})
        self.state_in = state_in
        self.state_out = state_out
        self.session = session
        self.dt = dt

    def step(self, state, dt=1.0, **dependent_states):
        return self.session.run(self.state_out, {self.state_in: state, self.dt: dt})