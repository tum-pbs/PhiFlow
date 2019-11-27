from unittest import TestCase

import numpy

from phi import struct, math
from phi.geom import Sphere
from phi.physics.domain import Domain
from phi.physics.field import StaggeredGrid
from phi.physics.field.effect import Fan, Inflow
from phi.physics.smoke import Smoke, SMOKE
from phi.physics.world import World


class TestSmoke(TestCase):

    def test_direct_smoke(self):
        smoke = Smoke(Domain([16, 16]))
        assert smoke.default_physics() == SMOKE
        smoke2 = SMOKE.step(smoke)
        assert smoke2.age == 1.0
        assert smoke.age == 0.0
        assert smoke2.name == smoke.name

    def test_simpleplume(self):
        world = World()
        world.batch_size = 3
        smoke = world.add(Smoke(Domain([16, 16])))
        inflow = world.add(Inflow(Sphere((8, 8), radius=4)))
        world.step()
        world.step(smoke)
        self.assertAlmostEqual(smoke.age, 2.0)
        self.assertAlmostEqual(inflow.age, 1.0)
        # self.assertEqual(smoke._batch_size, 3)

    def test_smoke_initializers(self):
        def typetest(smoke):
            self.assertIsInstance(smoke, Smoke)
            self.assertIsInstance(smoke.velocity, StaggeredGrid)
            numpy.testing.assert_equal(smoke.density.data.shape, [1, 4, 4, 1])
            numpy.testing.assert_equal(smoke.velocity.resolution, [4, 4])
            numpy.testing.assert_equal(smoke.velocity.data[0].resolution, [5, 4])
        typetest(Smoke(Domain([4, 4]), density=0.0, velocity=0.0))
        typetest(Smoke(Domain([4, 4]), density=1.0, velocity=1.0))
        typetest(Smoke(Domain([4, 4]), density=0, velocity=math.zeros))
        typetest(Smoke(Domain([4, 4]), density=lambda s: math.randn(s), velocity=lambda s: math.randn(s)))
        typetest(Smoke(Domain([4, 4]), density=numpy.zeros([1, 4, 4, 1]), velocity=numpy.zeros([1, 5, 5, 2])))
        typetest(Smoke(Domain([4, 4]), density=numpy.zeros([1, 4, 4, 1]), velocity=numpy.zeros([1, 5, 5, 2])))
        typetest(Smoke(Domain([4, 4])))

    def test_effects(self):
        world = World()
        world.add(Smoke(Domain([16, 16])))
        world.add(Fan(Sphere((10, 8), 5), [-1, 0]))
        world.step()
        world.step()

    def test_properties_dict(self):
        world = World()
        world.add(Smoke(Domain([16, 16])))
        world.add(Inflow(Sphere((8, 8), radius=4)))
        # world.add(ConstantDensity(box[0:2, 6:10], 1.0))
        world.add(Fan(Sphere((10, 8), 5), [-1, 0]))
        struct.properties_dict(world.state)

    def test_new_grids(self):
        smoke = Smoke(Domain([16, 16]), batch_size=3)
        centered_ones = smoke.centered_grid('f', 1)
        numpy.testing.assert_equal(centered_ones.data, 1)
        staggered_ones = smoke.staggered_grid('v', 1)
        numpy.testing.assert_equal(staggered_ones.data[0].data, 1)