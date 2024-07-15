from unittest import TestCase

from phi import math
from phi.geom import Box, build_mesh, Sphere, mesh_from_numpy
from phiml.math import spatial, vec


class TestGrid(TestCase):

    def test_build_mesh(self):
        build_mesh(Box(x=1, y=1), x=2, y=2)
        build_mesh(x=[0, 1, 3], y=[0, 1, 2])
        build_mesh(x=math.linspace(0, 1, spatial(x=10)), y=[0, 1, 4, 5])
        # --- with obstacles ---
        obs = Sphere(x=.5, y=0, radius=.5)
        build_mesh(Box(x=1, y=1), x=10, y=10, obstacles=obs)

    def test_lies_inside(self):
        points = [(0, 0), (1, 0), (0, 1)]
        polygons = [(0, 1, 2)]
        boundaries = {'outer': [(0, 1), (1, 2), (2, 0)]}
        mesh = mesh_from_numpy(points, polygons, boundaries)
        math.assert_close(True, mesh.lies_inside(vec(x=.4, y=.4)))
        math.assert_close(False, mesh.lies_inside(vec(x=1, y=1)))
        math.assert_close(False, mesh.lies_inside(vec(x=-.1, y=.5)))
        math.assert_close([False, True], mesh.lies_inside(vec(x=[-.1, .1], y=.5)))

    def test_closest_distance(self):
        points = [(0, 0), (1, 0), (0, 1)]
        polygons = [(0, 1, 2)]
        boundaries = {'outer': [(0, 1), (1, 2), (2, 0)]}
        mesh = mesh_from_numpy(points, polygons, boundaries)
        math.assert_close(.1, mesh.approximate_signed_distance(vec(x=-.1, y=.5)))
        math.assert_close(-.1, mesh.approximate_signed_distance(vec(x=.1, y=.5)))
        math.assert_close(-.1, mesh.approximate_signed_distance(vec(x=.5, y=.1)))
        math.assert_close(.1, mesh.approximate_signed_distance(vec(x=.5, y=-.1)))
