from unittest import TestCase

import numpy

from phi.data.fluidformat import Scene
from phi.geom import Sphere, box
from phi.physics.domain import Domain
from phi.physics.field.effect import Inflow
from phi.physics.obstacle import Obstacle
from phi.physics.smoke import Smoke
from phi.physics.world import World
from phi.tf.session import Session
from phi.tf.util import placeholder
from phi.tf.world import tf_bake_subgraph, tf_bake_graph


class TestSmokeTF(TestCase):

    def test_smoke_tf(self):
        world = World()
        smoke = Smoke(Domain([16, 16]))
        world.add(smoke)
        world.add(Inflow(Sphere((8, 8), radius=4)))
        world.add(Obstacle(box[4:16, 0:8]))
        smoke_in = smoke.copied_with(density=placeholder, velocity=placeholder)
        smoke_out = world.step(smoke_in)
        self.assertIsInstance(smoke_out, Smoke)
        session = Session(Scene.create('data', copy_calling_script=False))
        smoke = session.run(smoke_out, {smoke_in: smoke})
        smoke = session.run(smoke_out, {smoke_in: smoke})
        self.assertIsInstance(smoke, Smoke)

    def test_tf_subgraph(self):
        world = World()
        smoke = world.add(Smoke(Domain([16, 16])))
        tf_bake_subgraph(smoke, Session(Scene.create('data', copy_calling_script=False)))
        world.step()
        self.assertIsInstance(smoke.state, Smoke)
        self.assertIsInstance(smoke.state.density.data, numpy.ndarray)

    def test_tf_worldgraph(self):
        world = World()
        smoke = world.add(Smoke(Domain([16, 16])))
        tf_bake_graph(world, Session(Scene.create('data', copy_calling_script=False)))
        world.step()
        self.assertIsInstance(smoke.state, Smoke)
        self.assertIsInstance(smoke.state.density.data, numpy.ndarray)
