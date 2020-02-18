from unittest import TestCase

import numpy

from phi.data.fluidformat import Scene
from phi.geom import Sphere, box
from phi.physics.domain import Domain
from phi.physics.field.effect import Inflow
from phi.physics.material import CLOSED
from phi.physics.obstacle import Obstacle
from phi.physics.fluid import Fluid, IncompressibleFlow
from phi.physics.world import World
from phi.tf.flow import tf, Session, placeholder, variable, tf_bake_subgraph, tf_bake_graph


class TestFluidTF(TestCase):

    def test_fluid_tf(self):
        tf.reset_default_graph()
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
        tf.reset_default_graph()
        world = World()
        fluid = world.add(Fluid(Domain([16, 16])))
        tf_bake_subgraph(fluid, Session(Scene.create('data', copy_calling_script=False)))
        world.step()
        self.assertIsInstance(fluid.state, Fluid)
        self.assertIsInstance(fluid.state.density.data, numpy.ndarray)

    def test_tf_worldgraph(self):
        tf.reset_default_graph()
        world = World()
        fluid = world.add(Fluid(Domain([16, 16])))
        tf_bake_graph(world, Session(Scene.create('data', copy_calling_script=False)))
        world.step()
        self.assertIsInstance(fluid.state, Fluid)
        self.assertIsInstance(fluid.state.density.data, numpy.ndarray)

    def test_gradient_batch_independence(self):
        session = Session(None)  # Used to run the TensorFlow graph

        world = World()
        fluid = world.add(Fluid(Domain([40, 32], boundaries=CLOSED), buoyancy_factor=0.1, batch_size=2), physics=IncompressibleFlow())
        world.add(Inflow(Sphere(center=numpy.array([[5, 4], [5, 8]]), radius=3), rate=0.2))
        fluid.velocity = variable(fluid.velocity)  # create TensorFlow variable
        # fluid.velocity *= 0
        initial_state = fluid.state  # Remember the state at t=0 for later visualization
        session.initialize_variables()

        for frame in range(3):
            world.step(dt=1.5)

        target = session.run(fluid.density).data[0, ...]

        loss = tf.nn.l2_loss(fluid.density.data[1, ...] - target)
        self_loss = tf.nn.l2_loss(fluid.density.data[0, ...] - target)
        # loss = self_loss
        optim = tf.train.GradientDescentOptimizer(learning_rate=0.2).minimize(loss)
        session.initialize_variables()

        for optim_step in range(3):
            _, loss_value, sl_value = session.run([optim, loss, self_loss])

        staggered_velocity = session.run(initial_state.velocity).staggered_tensor()
        numpy.testing.assert_equal(staggered_velocity[0, ...], 0)
        assert numpy.all(~numpy.isnan(staggered_velocity))
