from unittest import TestCase

import numpy

from phi import struct, math
from phi.geom import Sphere
from phi.physics.domain import Domain
from phi.physics.field import StaggeredGrid
from phi.physics.field.effect import Fan, Inflow
from phi.physics.material import CLOSED, OPEN
from phi.physics.fluid import Fluid, INCOMPRESSIBLE_FLOW
from phi.physics.world import World


class TestFluid(TestCase):

    def test_direct_fluid(self):
        fluid = Fluid(Domain([16, 16]))
        assert fluid.default_physics() == INCOMPRESSIBLE_FLOW
        fluid2 = INCOMPRESSIBLE_FLOW.step(fluid)
        assert fluid2.age == 1.0
        assert fluid.age == 0.0
        assert fluid2.name == fluid.name

    def test_simpleplume(self):
        world = World()
        world.batch_size = 3
        fluid = world.add(Fluid(Domain([16, 16])))
        inflow = world.add(Inflow(Sphere((8, 8), radius=4)))
        world.step()
        world.step(fluid)
        self.assertAlmostEqual(fluid.age, 2.0)
        self.assertAlmostEqual(inflow.age, 1.0)

    def test_varying_boundaries(self):
        fluid = Fluid(Domain([16, 16], boundaries=[(CLOSED, OPEN), CLOSED]))
        INCOMPRESSIBLE_FLOW.step(fluid)

    def test_fluid_initializers(self):
        def typetest(fluid):
            self.assertIsInstance(fluid, Fluid)
            self.assertIsInstance(fluid.velocity, StaggeredGrid)
            numpy.testing.assert_equal(fluid.density.data.shape, [1, 4, 4, 1])
            numpy.testing.assert_equal(fluid.velocity.resolution, [4, 4])
            numpy.testing.assert_equal(fluid.velocity.data[0].resolution, [5, 4])
        typetest(Fluid(Domain([4, 4]), density=0.0, velocity=0.0))
        typetest(Fluid(Domain([4, 4]), density=1.0, velocity=1.0))
        typetest(Fluid(Domain([4, 4]), density=0, velocity=math.zeros))
        typetest(Fluid(Domain([4, 4]), density=lambda s: math.randn(s), velocity=lambda s: math.randn(s)))
        typetest(Fluid(Domain([4, 4]), density=numpy.zeros([1, 4, 4, 1]), velocity=numpy.zeros([1, 5, 5, 2])))
        typetest(Fluid(Domain([4, 4]), density=numpy.zeros([1, 4, 4, 1]), velocity=numpy.zeros([1, 5, 5, 2])))
        typetest(Fluid(Domain([4, 4])))

    def test_effects(self):
        world = World()
        world.add(Fluid(Domain([16, 16])))
        world.add(Fan(Sphere((10, 8), 5), [-1, 0]))
        world.step()
        world.step()

    def test_properties_dict(self):
        world = World()
        world.add(Fluid(Domain([16, 16])))
        world.add(Inflow(Sphere((8, 8), radius=4)))
        # world.add(ConstantDensity(box[0:2, 6:10], 1.0))
        world.add(Fan(Sphere((10, 8), 5), [-1, 0]))
        struct.properties_dict(world.state)

    def test_new_grids(self):
        fluid = Fluid(Domain([16, 16]), batch_size=3)
        centered_ones = fluid.centered_grid('f', 1)
        numpy.testing.assert_equal(centered_ones.data, 1)
        staggered_ones = fluid.staggered_grid('v', 1)
        numpy.testing.assert_equal(staggered_ones.data[0].data, 1)