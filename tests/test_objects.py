from unittest import TestCase

from phi.geom import Sphere, box
from phi.physics.field.effect import Inflow
from phi.physics.obstacle import Obstacle, GeometryMovement
from phi.physics.world import World


def geometry_at(time):
    return Sphere([time], radius=5)


class TestObjects(TestCase):

    def test_geometry_movement(self):
        obstacle = Obstacle(geometry_at(0))
        phys = GeometryMovement(geometry_at)

        obstacle = phys.step(obstacle, dt=2.0)
        self.assertIsInstance(obstacle, Obstacle)
        self.assertAlmostEqual(obstacle.age, 2.0)
        self.assertAlmostEqual(obstacle.geometry.center[0], 2.0)
        self.assertAlmostEqual(obstacle.velocity[0], 1.0)

        obstacle = phys.step(obstacle, dt=2.0)
        self.assertIsInstance(obstacle, Obstacle)
        self.assertAlmostEqual(obstacle.age, 4.0)
        self.assertAlmostEqual(obstacle.geometry.center[0], 4.0)
        self.assertAlmostEqual(obstacle.velocity[0], 1.0)

    def test_collective_step(self):
        world = World()
        obstacle = world.add(Obstacle(geometry_at(0)), physics=GeometryMovement(geometry_at))
        inflow = world.add(Inflow(geometry_at(0)))
        inflow.physics = world.get_physics(obstacle)
        static_obstacle = world.add(Obstacle(box[0:1]))
        world.step()
        self.assertAlmostEqual(obstacle.age, 1.0)
        self.assertAlmostEqual(obstacle.geometry.center[0], 1.0)
        self.assertAlmostEqual(inflow.age, 1.0)
        self.assertAlmostEqual(inflow.field.geometries[0].center[0], 1.0)
        self.assertAlmostEqual(static_obstacle.age, 1.0)
