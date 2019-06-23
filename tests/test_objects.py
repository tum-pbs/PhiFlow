from unittest import TestCase
from phi.flow import *


def geometry_at(time):
    return Sphere([time], radius=5)


class TestObjects(TestCase):
    def test_geometry(self):
        obstacle = Obstacle(geometry_at(0), physics=GeometryMovement(geometry_at))

        obstacle = step(obstacle, dt=2.0)
        self.assertIsInstance(obstacle, Obstacle)
        self.assertAlmostEqual(obstacle.age, 2.0)
        self.assertEqual(obstacle.step_count, 1)
        self.assertAlmostEqual(obstacle.geometry.center[0], 2.0)
        self.assertAlmostEqual(obstacle.velocity[0], 1.0)

        obstacle = step(obstacle, dt=2.0)
        self.assertIsInstance(obstacle, Obstacle)
        self.assertAlmostEqual(obstacle.age, 4.0)
        self.assertEqual(obstacle.step_count, 2)
        self.assertAlmostEqual(obstacle.geometry.center[0], 4.0)
        self.assertAlmostEqual(obstacle.velocity[0], 1.0)

    def test_collective_step(self):
        objects = [ Obstacle(box[0:1]) ]
        r = step(objects)
        self.assertIsInstance(r, list)
        self.assertIsInstance(r[0], Obstacle)

    def test_multi_step(self):
        objects = [Obstacle(box[0:1]), (Inflow(box[1:2]), Obstacle(box[2:3]))]
        r = step(objects)
        self.assertIsInstance(r, list)
        self.assertIsInstance(r[0], Obstacle)
        self.assertIsInstance(r[1], tuple)
        self.assertIsInstance(r[1][0], Inflow)
        self.assertIsInstance(r[1][1], Obstacle)
        self.assertNotEqual(r[0], objects[0])
        self.assertNotEqual(r[1][0], objects[1][0])
        self.assertNotEqual(r[1][1], objects[1][1])
        self.assertEqual(r[1][0].step_count, 1)
        self.assertEqual(objects[1][0].step_count, 1)