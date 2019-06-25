from phi.physics.world import *
from .util import *



def shadow_world_tf(tracker, session):
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

    def step(self, state, dependent_states, dt=1.0):
        return self.session.run(self.state_out, {self.state_in: state, self.dt: dt})