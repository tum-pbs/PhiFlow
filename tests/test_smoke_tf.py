from unittest import TestCase
from phi.tf.flow import *


class TestSmokeTF(TestCase):
    def test_smoke_tf(self):
        world = World()
        smoke = world.Smoke(Domain([16, 16])).state
        world.Inflow(Sphere((8, 8), radius=4))
        world.Obstacle(box[4:16, 0:8])
        smoke_in = smoke.copied_with(density=placeholder, velocity=placeholder)
        smoke_out = world.step(smoke_in)
        session = Session(Scene.create('data'))
        smoke = session.run(smoke_out, {smoke_in: smoke})
        smoke = session.run(smoke_out, {smoke_in: smoke})
        self.assertIsInstance(smoke_out, Smoke)

    def test_tf_world(self):
        world = World()
        smoke = world.Smoke(Domain([16, 16]))
        shadow_world_tf(smoke, Session(Scene.create('data')))
        world.step()
        self.assertIsInstance(smoke.state, Smoke)
        self.assertIsInstance(smoke.state.density, np.ndarray)