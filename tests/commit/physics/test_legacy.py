from unittest import TestCase

from phi import struct, math
from phi.geom import Sphere, Box
from phi.physics import Domain, CLOSED, OPEN, Obstacle
from phi.physics._effect import Fan, Inflow
from phi.physics._fluid_legacy import Fluid, IncompressibleFlow
from phi.physics._world import World


class TestLegacyPhysics(TestCase):

    def test_direct_fluid(self):
        fluid = Fluid(Domain(x=16, y=16))
        fluid2 = IncompressibleFlow().step(fluid)
        assert fluid2.age == 1.0
        assert fluid.age == 0.0
        assert fluid2.name == fluid.name

    def test_smoke_plume(self):
        world = World()
        world.batch_size = 3
        fluid = world.add(Fluid(Domain(x=16, y=16)), physics=IncompressibleFlow())
        inflow = world.add(Inflow(Sphere((8, 8), radius=4)))
        world.step()
        world.step(fluid)
        self.assertAlmostEqual(fluid.age, 2.0)
        self.assertAlmostEqual(inflow.age, 1.0)

    def test_varying_boundaries(self):
        fluid = Fluid(Domain(x=16, y=16, boundaries=[(CLOSED, OPEN), CLOSED]))
        IncompressibleFlow().step(fluid)

    def test_effects(self):
        world = World()
        fluid = world.add(Fluid(Domain(x=16, y=16)), physics=IncompressibleFlow())
        fan = world.add(Fan(Sphere((10, 8), 5), [-1, 0]))
        obstacle = world.add(Obstacle(Box[0:1, 0:1]))
        world.step(dt=1)
        world.step(dt=0.5)
        assert fluid.age == fan.age == obstacle.age == 1.5

    def test_properties_dict(self):
        world = World()
        world.add(Fluid(Domain(x=16, y=16)), physics=IncompressibleFlow())
        world.add(Inflow(Sphere((8, 8), radius=4)))
        world.add(Fan(Sphere((10, 8), 5), [-1, 0]))
        struct.properties_dict(world.state)

    def test_batch_independence(self):
        def simulate(centers):
            world = World()
            fluid = world.add(Fluid(Domain(x=5, y=4, boundaries=CLOSED, bounds=Box(0, [40, 32])),
                                    buoyancy_factor=0.1,
                                    batch_size=centers.shape[0]),
                              physics=IncompressibleFlow())
            world.add(Inflow(Sphere(center=centers, radius=3), rate=0.2))
            world.add(Fan(Sphere(center=centers, radius=5), acceleration=[1.0, 0]))
            world.step(dt=1.5)
            world.step(dt=1.5)
            world.step(dt=1.5)
            assert not math.close(fluid.density.values, 0)
            print()
            return fluid.density.values.batch[0], fluid.velocity.values.batch[0]

        d1, v1 = simulate(math.tensor([[5, 16], [5, 4]], names='batch,vector'))
        d2, v2 = simulate(math.tensor([[5, 16], [5, 16]], names='batch,vector'))

        math.assert_close(d1, d2)
        math.assert_close(v1, v2)
