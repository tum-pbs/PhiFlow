from unittest import TestCase

from phi import math, geom
from phi.geom import Box, Sphere
from phi.geom._box import Cuboid
from phi.math import batch, channel, spatial


class TestGeom(TestCase):

    def test_circle_area(self):
        sphere = Sphere(math.tensor([(0, 0), (1, 1)], batch('batch'), channel('vector')), radius=math.tensor([1, 2], batch('batch')))
        math.assert_close(sphere.volume, [math.PI, 4 * math.PI])

    def test_sphere_volume(self):
        sphere = Sphere(math.tensor([(0, 0, 0), (1, 1, 1)], batch('batch'), channel('vector')), radius=math.tensor([1, 2], batch('batch')))
        math.assert_close(sphere.volume, [4/3 * math.PI, 4/3 * 8 * math.PI])

    def test_sphere_constructor_kwargs(self):
        s = Sphere(x=0.5, y=2, radius=1.)
        self.assertEqual(s.center.shape.get_item_names('vector'), ('x', 'y'))

    def test_box_constructor_kwargs(self):
        b = Box(x=3.5, y=4)
        math.assert_close(b.lower, 0)
        math.assert_close(b.upper, (3.5, 4))
        b = Box(x=(1, 2), y=None)
        math.assert_close(b.lower, (1, -math.INF))
        math.assert_close(b.upper, (2, math.INF))
        b = Box(x=(None, None))
        math.assert_close(b.lower, -math.INF)
        math.assert_close(b.upper, math.INF)

    def test_union_varying(self):
        box = Box(x=1, y=1)
        sphere = Sphere((0, 0), radius=1)
        union = geom.union(box, sphere)
        math.assert_close(union.approximate_signed_distance((1, 1)), union.approximate_signed_distance((0, -1)), 0)

    def test_infinite_cylinder(self):
        cylinder = geom.infinite_cylinder(x=.5, y=.5, radius=.5, inf_dim=math.spatial('z'))
        self.assertEqual(cylinder.spatial_rank, 3)
        cylinder = geom.infinite_cylinder(x=.5, y=.5, radius=.5, inf_dim='z')
        loc = math.wrap([(0, 0, 0), (.5, .5, 0), (1, 1, 0), (0, 0, 100), (.5, .5, 100)], math.instance('points'), math.channel(vector='x,y,z'))
        inside = math.wrap([False, True, False, False, True], math.instance('points'))
        math.assert_close(cylinder.lies_inside(loc), inside)
        loc = math.wrap([(0, 0, 0), (.5, 1, 0), (1, 1, 0), (0, 0, 100), (.5, 1, 100)], math.instance('points'), math.channel(vector='x,y,z'))
        corner_distance = math.sqrt(2) / 2 - .5
        distance = math.wrap([corner_distance, 0, corner_distance, corner_distance, 0], math.instance('points'))
        math.assert_close(cylinder.approximate_signed_distance(loc), distance)
