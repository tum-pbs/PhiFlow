from unittest import TestCase

from phi import math, geom
from phi.math import stack, vec, instance, expand, rename_dims, unpack_dim, pack_dims, spatial, flatten, batch, channel
from phi.geom import Box, Sphere


class TestGeom(TestCase):

    def test_union_varying(self):
        box = Box(x=1, y=1)
        sphere = Sphere(x=0, y=0, radius=1)
        union = geom.union(box, sphere)
        math.assert_close(union.approximate_signed_distance((1, 1)), union.approximate_signed_distance((0, -1)), 0)

    def test_infinite_cylinder(self):
        cylinder = geom.infinite_cylinder(x=.5, y=.5, radius=.5, inf_dim=math.spatial('z'))
        self.assertEqual(3, cylinder.spatial_rank)
        self.assertEqual(('x', 'y', 'z'), cylinder.shape.get_item_names('vector'))
        cylinder = geom.infinite_cylinder(x=.5, y=.5, radius=.5, inf_dim='z')
        loc = math.wrap([(0, 0, 0), (.5, .5, 0), (1, 1, 0), (0, 0, 100), (.5, .5, 100)], math.instance('points'), math.channel(vector='x,y,z'))
        inside = math.wrap([False, True, False, False, True], math.instance('points'))
        math.assert_close(cylinder.lies_inside(loc), inside)
        loc = math.wrap([(0, 0, 0), (.5, 1, 0), (1, 1, 0), (0, 0, 100), (.5, 1, 100)], math.instance('points'), math.channel(vector='x,y,z'))
        corner_distance = math.sqrt(2) / 2 - .5
        distance = math.wrap([corner_distance, 0, corner_distance, corner_distance, 0], math.instance('points'))
        math.assert_close(cylinder.approximate_signed_distance(loc), distance)

    def test_point_reshaping(self):
        s = stack([geom.Point(vec(x=0, y=0))] * 50, instance('points'))
        s = expand(s, batch(b=100))
        s = rename_dims(s, 'b', 'bat')
        s = unpack_dim(s, 'points', spatial(x=10, y=5))
        self.assertEqual(set(batch(bat=100) & spatial(x=10, y=5) & channel(vector='x,y')), set(s.shape))
        s = pack_dims(s, 'x,y', instance('particles'))
        self.assertEqual(set(batch(bat=100) & instance(particles=50) & channel(vector='x,y')), set(s.shape))
        s = flatten(s)
        self.assertEqual(set(batch(bat=100) & instance(flat=50) & channel(vector='x,y')), set(s.shape))
