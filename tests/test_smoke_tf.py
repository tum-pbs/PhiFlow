from unittest import TestCase
from phi.tf.flow import *


class TestSmokeTF(TestCase):
    def test_smoke_tf(self):
        world.reset()
        smoke = world.Smoke(Domain([16, 16]))
        world.Inflow(Sphere((8, 8), radius=4))
        world.Obstacle(box[4:16, 0:8])
        state = smoke.initial_state()
        session = Session(Scene.create('data'))
        state_in = placeholder(smoke.shape())
        state_out = smoke.step(state_in)  # depends on Session() which calls load_tensorflow()
        state = session.run(state_out, {state_in: state})
        state = session.run(state_out, {state_in: state})
