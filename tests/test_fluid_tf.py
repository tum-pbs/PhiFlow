from unittest import TestCase

import numpy

from phi.tf.flow import tf, Session, placeholder, variable, tf_bake_subgraph, tf_bake_graph, Noise, constant, struct, OPEN, PERIODIC, STICKY, SLIPPERY, World, Fluid, IncompressibleFlow, Obstacle, CLOSED, Inflow, Domain, Sphere, box, Scene, math


class TestFluidTF(TestCase):

    def test_fluid_tf(self):
        tf.reset_default_graph()
        world = World()
        fluid = Fluid(Domain([16, 16]))
        world.add(fluid, physics=IncompressibleFlow())
        world.add(Inflow(Sphere((8, 8), radius=4)))
        world.add(Obstacle(box[4:16, 0:8]))
        fluid_in = fluid.copied_with(density=placeholder, velocity=placeholder)
        fluid_out = world.step(fluid_in)
        self.assertIsInstance(fluid_out, Fluid)
        session = Session(Scene.create('data', copy_calling_script=False))
        fluid = session.run(fluid_out, {fluid_in: fluid})
        fluid = session.run(fluid_out, {fluid_in: fluid})
        self.assertIsInstance(fluid, Fluid)

    def test_fluid_tf_equality(self):
        tf.reset_default_graph()
        _sess = tf.InteractiveSession()
        for domain in [
            Domain([8, 6], boundaries=OPEN),
            Domain([8, 6], boundaries=STICKY),
            Domain([8, 6], boundaries=SLIPPERY),
            Domain([8, 6], boundaries=PERIODIC),
            Domain([8, 6], boundaries=[PERIODIC, [OPEN, STICKY]])
        ]:
            print('Comparing on domain %s' % (domain.boundaries,))
            np_fluid = Fluid(domain, density=Noise(), velocity=Noise(), batch_size=10)
            tf_fluid = constant(np_fluid)
            physics = IncompressibleFlow(conserve_density=False)
            for _ in range(3):
                np_fluid = physics.step(np_fluid, 1.0)
                tf_fluid = physics.step(tf_fluid, 1.0)
                for np_tensor, tf_tensor in zip(struct.flatten(np_fluid), struct.flatten(tf_fluid)):
                    tf_eval = tf_tensor.eval()
                    numpy.testing.assert_almost_equal(np_tensor, tf_eval, decimal=5)

    def test_tf_subgraph(self):
        tf.reset_default_graph()
        world = World()
        fluid = world.add(Fluid(Domain([16, 16])), physics=IncompressibleFlow())
        tf_bake_subgraph(fluid, Session(Scene.create('data', copy_calling_script=False)))
        world.step()
        self.assertIsInstance(fluid.state, Fluid)
        self.assertIsInstance(fluid.state.density.data, numpy.ndarray)

    def test_tf_worldgraph(self):
        tf.reset_default_graph()
        world = World()
        fluid = world.add(Fluid(Domain([16, 16])), physics=IncompressibleFlow())
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

    def test_precision_64(self):
        try:
            math.set_precision(64)
            fluid = Fluid(Domain([16, 16]), density=math.maximum(0, Noise()))
            fluid = variable(fluid)
            self.assertEqual(fluid.density.data.dtype.as_numpy_dtype, numpy.float64)
            self.assertEqual(fluid.velocity.unstack()[0].data.dtype.as_numpy_dtype, numpy.float64)
            fluid = IncompressibleFlow().step(fluid, dt=1.0)
            self.assertEqual(fluid.density.data.dtype.as_numpy_dtype, numpy.float64)
            self.assertEqual(fluid.velocity.unstack()[0].data.dtype.as_numpy_dtype, numpy.float64)
        finally:
            math.set_precision(32)  # Reset environment

    def test_precision_16(self):
        try:
            math.set_precision(16)
            fluid = Fluid(Domain([16, 16]), density=math.maximum(0, Noise()))
            fluid = variable(fluid)
            self.assertEqual(fluid.density.data.dtype.as_numpy_dtype, numpy.float16)
            self.assertEqual(fluid.velocity.unstack()[0].data.dtype.as_numpy_dtype, numpy.float16)
            fluid = IncompressibleFlow().step(fluid, dt=1.0)
            self.assertEqual(fluid.density.data.dtype.as_numpy_dtype, numpy.float16)
            self.assertEqual(fluid.velocity.unstack()[0].data.dtype.as_numpy_dtype, numpy.float16)
        finally:
            math.set_precision(32)  # Reset environment
