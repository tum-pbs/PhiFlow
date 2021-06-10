from unittest import TestCase

from phi import math
from phi.geom import Box, Sphere, stack
from phi.math import batch, channel, spatial


class TestGeom(TestCase):

    def test_box_constructor(self):
        box = Box(0, (1, 1))
        math.assert_close(box.size, 1)
        self.assertEqual(math.spatial(x=1, y=1), box.shape)

    def test_box_batched(self):
        box = Box(math.tensor([(0, 0), (1, 1)], batch('boxes'), channel('vector')), 1)
        self.assertEqual(math.batch(boxes=2) & spatial(x=1, y=1), box.shape)

    def test_box_volume(self):
        box = Box(math.tensor([(0, 0), (1, 1)], batch('boxes'), channel('vector')), 1)
        math.assert_close(box.volume, [1, 0])

    def test_sphere_volume(self):
        sphere = Sphere(math.tensor([(0, 0), (1, 1)], batch('batch'), channel('vector')), radius=math.tensor([1, 2], batch('batch')))
        math.assert_close(sphere.volume, [4/3 * math.PI, 4/3 * math.PI * 8])

    def test_stack_volume(self):
        u = stack([Box[0:1, 0:1], Box[0:2, 0:2]], batch('batch'))
        math.assert_close(u.volume, [1, 4])
