from unittest import TestCase
from phi.flow import *


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
        obstacle = world.Obstacle(geometry_at(0))
        obstacle.physics = GeometryMovement(geometry_at)
        inflow = world.Inflow(geometry_at(0))
        inflow.physics = obstacle.physics
        static_obstacle = world.Obstacle(box[0:1])
        world.step()
        self.assertAlmostEqual(obstacle.age, 1.0)
        self.assertAlmostEqual(obstacle.geometry.center[0], 1.0)
        self.assertAlmostEqual(inflow.age, 1.0)
        self.assertAlmostEqual(inflow.field.bounds.center[0], 1.0)
        self.assertAlmostEqual(static_obstacle.age, 1.0)
