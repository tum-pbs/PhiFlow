from unittest import TestCase

from phi.geom import Box, Sphere, sdf_from_geometry, SDFGrid
from phiml import math
from phiml.math import channel, vec


class TestSDFGrid(TestCase):

    def test_sdf_from_geometry(self):
        sphere = Sphere(x=1, y=1, radius=.8)
        sdf = sdf_from_geometry(sphere, Box(x=3, y=3), x=100, y=100)
        self.assertEqual(channel(vector='x,y'), sdf.shape)
        math.assert_close(sdf.center, sphere.center)
        math.assert_close(sdf.volume, sphere.volume)
        math.assert_close(sdf.bounding_radius(), sphere.bounding_radius())

    def test_sdf_creation(self):
        sphere = Sphere(x=1, y=1, radius=.8)
        sdf1 = sdf_from_geometry(sphere, Box(x=3, y=3), x=100, y=100)
        sdf = SDFGrid(sdf1.values, sdf1.bounds)
        self.assertEqual(channel(vector='x,y'), sdf.shape)
        math.assert_close(sdf.center, sphere.center, abs_tolerance=.1)
        math.assert_close(sdf.volume, sphere.volume, abs_tolerance=.1)
        math.assert_close(sdf.bounding_radius(), sphere.bounding_radius(), abs_tolerance=.2)

    def test_lies_inside(self):
        sphere = Sphere(x=1, y=1, radius=.8)
        sdf = sdf_from_geometry(sphere, Box(x=3, y=3), x=100, y=100)
        math.assert_close(sphere.lies_inside(sdf.points), sdf.lies_inside(sdf.points))

    def test_signed_distance(self):
        sphere = Sphere(x=1, y=1, radius=.8)
        sdf = sdf_from_geometry(sphere, Box(x=3, y=3), x=100, y=100)
        math.assert_close(sphere.approximate_signed_distance(sdf.points), sdf.approximate_signed_distance(sdf.points), abs_tolerance=.1)
        sdf = sdf_from_geometry(sphere, Box(x=(2, 3), y=3), x=100, y=100)  # signed_distance inside sphere only approximate
        math.assert_close(sphere.approximate_signed_distance(sdf.points), sdf.approximate_signed_distance(sdf.points))

    def test_closest_surface(self):
        sphere = Sphere(x=1, y=1, radius=.8)
        sdf = sdf_from_geometry(sphere, Box(x=(2, 3), y=3), x=100, y=100)
        sgn_dist_sph, delta_sph, normal_sph, offset_sph, _ = sphere.approximate_closest_surface(sdf.points)
        sgn_dist_sdf, delta_sdf, normal_sdf, offset_sdf, _ = sdf.approximate_closest_surface(sdf.points)
        math.assert_close(sgn_dist_sph, sgn_dist_sdf, abs_tolerance=.1)
        math.assert_close(delta_sph, delta_sdf, abs_tolerance=.1)
        math.assert_close(normal_sph, normal_sdf, abs_tolerance=.1)
        math.assert_close(offset_sph, offset_sdf, abs_tolerance=.1)

    def test_sdf_rebuild(self):
        spheres = Sphere(vec(x=[1, 2], y=1), radius=.8)
        bounds = Box(x=3, y=2)
        sdf = sdf_from_geometry(spheres, bounds, x=64, y=64, rebuild='from-surface')

