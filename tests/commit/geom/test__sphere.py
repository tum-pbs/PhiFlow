from unittest import TestCase


from phi import math
from phi.geom import Box, union, Cuboid, embed, Sphere
from phi.math import batch, channel
from phi.math.magic import Shaped, Sliceable, Shapable


class TestSphere(TestCase):

    def test_interfaces(self):
        sphere = Sphere(x=0, radius=5)
        self.assertIsInstance(sphere, Shaped)
        self.assertIsInstance(sphere, Sliceable)
        self.assertIsInstance(sphere, Shapable)

    def test_circle_area(self):
        sphere = Sphere(math.tensor([(0, 0), (1, 1)], batch('batch'), channel(vector='x,y')), radius=math.tensor([1, 2], batch('batch')))
        math.assert_close(sphere.volume, [math.PI, 4 * math.PI])

    def test_sphere_volume(self):
        sphere = Sphere(math.tensor([(0, 0, 0), (1, 1, 1)], batch('batch'), channel(vector='x,y,z')), radius=math.tensor([1, 2], batch('batch')))
        math.assert_close(sphere.volume, [4/3 * math.PI, 4/3 * 8 * math.PI])

    def test_sphere_constructor_kwargs(self):
        s = Sphere(x=0.5, y=2, radius=1.)
        self.assertEqual(s.center.shape.get_item_names('vector'), ('x', 'y'))

    def test_stack_type(self):
        spheres = math.stack([Sphere(x=1, radius=1), Sphere(x=2, radius=1)], batch('batch'))
        self.assertIsInstance(spheres, Sphere)

    def test_slice(self):
        s1, s2 = Sphere(x=0, radius=1), Sphere(x=5, radius=1)
        u = union(s1, s2)
        self.assertEqual(s1, u.union[0])
        self.assertEqual(s2, u.union[1])

    def test_project(self):
        sphere = Sphere(x=4, y=3, radius=1)
        self.assertEqual(Sphere(x=4, radius=1), sphere.vector['x'])
        self.assertEqual(sphere, sphere.vector['x,y'])
        self.assertEqual(Sphere(x=4, radius=1), sphere['x'])

    # def test_box_constructor(self):
    #     try:
    #         Box(0, (1, 1))
    #         raise RuntimeError
    #     except AssertionError:
    #         pass
    #     math.assert_close(Box(x=1, y=1).size, 1)
    #
    # def test_box_batched(self):
    #     lower = math.tensor([(0, 0), (1, 1)], batch('boxes'), channel(vector='x,y'))
    #     upper = math.wrap((1, 1), channel(vector='x,y'))
    #     box = Box(lower, upper)
    #     self.assertEqual(batch(boxes=2) & channel(vector='x,y'), box.shape)
    #
    # def test_slice(self):
    #     b1, b2 = Box(x=4, y=3), Box(x=2, y=1)
    #     u = union(b1, b2)
    #     self.assertEqual(b1, u.union[0])
    #     self.assertEqual(b2, u.union[1])
    #
    # def test_without(self):
    #     box = Box(x=4, y=3)
    #     self.assertEqual(Box(x=4), box.without(('y',)))
    #     self.assertEqual(Box(), box.without(('x', 'y')))
    #     self.assertEqual(box, box.without(()))
    #
    # def test_embed(self):
    #     self.assertEqual(Box(x=4, y=3, z=None), embed(Box(x=4, y=3), 'z'))
    #     self.assertEqual(Box(x=4, y=3, z=None), embed(Box(x=4, y=3), 'x,z'))
    #
    # def test_box_product(self):
    #     a = Box(x=4)
    #     b = Box(y=3).shifted(math.wrap(1))
    #     ab = a * b
    #     self.assertEqual(2, ab.spatial_rank)
    #     math.assert_close(ab.size, (4, 3))
    #     math.assert_close(ab.lower, (0, 1))
    #
    # def test_union_same(self):
    #     u = union(Box(x=1, y=1), Box(x=(2, 3), y=1))
    #     self.assertIsInstance(u, Box)
    #     math.assert_close(u.approximate_signed_distance((0, 0)), u.approximate_signed_distance((3, 1)), 0)
    #     math.assert_close(u.approximate_signed_distance((1.5, 0)), 0.5)
    #
    # def test_stack_volume(self):
    #     u = math.stack([Box(x=1, y=1), Box(x=2, y=2)], batch('batch'))
    #     math.assert_close(u.volume, [1, 4])
    #
    # def test_shape_type(self):
    #     box = Box(x=1, y=2)
    #     self.assertEqual(box.rotated(0.1).shape_type, 'rotB')
    #
    # def test_box_eq(self):
    #     self.assertNotEqual(Box(x=1, y=1), Box(x=1))
    #     self.assertEqual(Box(x=1, y=1), Box(x=1, y=1))
    #
    # def test_cuboid_constructor_kwargs(self):
    #     c = Cuboid(x=2., y=1.)
    #     math.assert_close(c.lower, -c.upper, (-1, -.5))
