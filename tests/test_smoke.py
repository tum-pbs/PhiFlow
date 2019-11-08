from unittest import TestCase

from phi.flow import *


class TestSmoke(TestCase):
    
    def test_direct_smoke(self):
        smoke = Smoke(Domain([16, 16]))
        assert smoke.default_physics() == SMOKE
        smoke2 = SMOKE.step(smoke)
        assert(smoke2.age == 1.0)
        assert(smoke.age == 0.0)
        assert(smoke2.trajectorykey == smoke.trajectorykey)

    def test_simpleplume(self):
        world = World()
        world.batch_size = 3
        smoke = world.add(Smoke(Domain([16, 16])))
        inflow = world.add(Inflow(Sphere((8, 8), radius=4)))
        world.step()
        world.step(smoke)
        self.assertAlmostEqual(world.state.age, 2.0)
        self.assertAlmostEqual(smoke.age, 2.0)
        self.assertAlmostEqual(inflow.age, 1.0)
        self.assertEqual(smoke._batch_size, 3)

    def test_smoke_initializers(self):
        def typetest(smoke):
            self.assertIsInstance(smoke, Smoke)
            self.assertIsInstance(smoke.velocity, StaggeredGrid)
            np.testing.assert_equal(smoke.density.data.shape, [1, 4, 4, 1])
            np.testing.assert_equal(smoke.velocity.resolution, [4, 4])
            np.testing.assert_equal(smoke.velocity.data[0].resolution, [5, 4])
        typetest(Smoke(Domain([4, 4]), density=0.0, velocity=0.0))
        typetest(Smoke(Domain([4, 4]), density=1.0, velocity=1.0))
        typetest(Smoke(Domain([4, 4]), density=0, velocity=math.zeros))
        typetest(Smoke(Domain([4, 4]), density=lambda s: math.randn(s), velocity=lambda s: math.randn(s)))
        typetest(Smoke(Domain([4, 4]), density=np.zeros([1, 4, 4, 1]), velocity=np.zeros([1, 5, 5, 2])))
        typetest(Smoke(Domain([4, 4]), density=np.zeros([1, 4, 4, 1]), velocity=np.zeros([1, 5, 5, 2])))
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
