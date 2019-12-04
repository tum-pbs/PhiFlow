from unittest import TestCase

import numpy

from phi.data.fluidformat import Scene
from phi.geom import Sphere, box
from phi.physics.domain import Domain
from phi.physics.field.effect import Inflow
from phi.physics.obstacle import Obstacle
from phi.physics.fluid import Fluid
from phi.physics.world import World
from phi.tf.session import Session
from phi.tf.util import placeholder
from phi.tf.world import tf_bake_subgraph, tf_bake_graph


class TestFluidTF(TestCase):

    def test_fluid_tf(self):
        world = World()
        fluid = Fluid(Domain([16, 16]))
        world.add(fluid)
        world.add(Inflow(Sphere((8, 8), radius=4)))
        world.add(Obstacle(box[4:16, 0:8]))
        fluid_in = fluid.copied_with(density=placeholder, velocity=placeholder)
        fluid_out = world.step(fluid_in)
        self.assertIsInstance(fluid_out, Fluid)
        session = Session(Scene.create('data', copy_calling_script=False))
        fluid = session.run(fluid_out, {fluid_in: fluid})
        fluid = session.run(fluid_out, {fluid_in: fluid})
        self.assertIsInstance(fluid, Fluid)

    def test_tf_subgraph(self):
        world = World()
        fluid = world.add(Fluid(Domain([16, 16])))
        tf_bake_subgraph(fluid, Session(Scene.create('data', copy_calling_script=False)))
        world.step()
        self.assertIsInstance(fluid.state, Fluid)
        self.assertIsInstance(fluid.state.density.data, numpy.ndarray)

    def test_tf_worldgraph(self):
        world = World()
        fluid = world.add(Fluid(Domain([16, 16])))
        tf_bake_graph(world, Session(Scene.create('data', copy_calling_script=False)))
        world.step()
        self.assertIsInstance(fluid.state, Fluid)
        self.assertIsInstance(fluid.state.density.data, numpy.ndarray)
