from unittest import TestCase

from phi import math
from phi.geom import union, Sphere
from phi.math import stack, vec, instance, expand, rename_dims, unpack_dim, pack_dims, spatial, flatten, batch, channel
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

    def test_reshaping(self):
        s = stack([Sphere(vec(x=0, y=0), radius=1)] * 50, instance('points'))
        s = expand(s, batch(b=100))
        s = rename_dims(s, 'b', 'bat')
        s = unpack_dim(s, 'points', spatial(x=10, y=5))
        assert batch(bat=100) & spatial(x=10, y=5) & channel(vector='x,y') == s.shape
        s = pack_dims(s, 'x,y', instance('particles'))
        assert batch(bat=100) & instance(particles=50) & channel(vector='x,y') == s.shape
        s = flatten(s)
        assert batch(bat=100) & instance(flat=50) & channel(vector='x,y') == s.shape

    def test_reshaping_const_radius(self):
        s = Sphere(stack([vec(x=0, y=0)] * 50, instance('points')), radius=1)
        s = expand(s, batch(b=100))
        s = rename_dims(s, 'b', 'bat')
        s = unpack_dim(s, 'points', spatial(x=10, y=5))
        assert not s.radius.shape
        assert batch(bat=100) & spatial(x=10, y=5) & channel(vector='x,y') == s.shape
        s = pack_dims(s, 'x,y', instance('particles'))
        assert not s.radius.shape
        assert batch(bat=100) & instance(particles=50) & channel(vector='x,y') == s.shape
        s = flatten(s)
        assert batch(bat=100) & instance(flat=50) & channel(vector='x,y') == s.shape
