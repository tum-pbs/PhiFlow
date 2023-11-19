from unittest import TestCase

from phiml import math
from phiml.math import spatial, wrap, vec

from phi.geom import Box, Heightmap
from phi.field import CenteredGrid

class TestHeightmap(TestCase):

    def test_1d_creation(self):
        x = math.range(spatial(x=11))
        height = math.exp(-.5 * (x - 5) ** 2)
        bounds = Box(x=(0, 10), y=1)
        Heightmap(height, bounds, max_dist=.1)

    def test_important_faces_1d(self):
        # --- simple case ---
        height = wrap([-1, 0, 0, 1], spatial('x'))
        bounds = Box(x=2, y=(-1, 1))
        heightmap = Heightmap(height, bounds, max_dist=.1)
        math.assert_close([1, 2, 1], heightmap._faces.index.consider['outside']['x'])
        math.assert_close([1, 0, 1], heightmap._faces.index.consider['inside']['x'])
        # --- complex case ---
        height = wrap([.1, .02, 0, 0, 1, .95, .8, .5, 0], spatial('x'))
        bounds = Box(x=2, y=1)
        heightmap = Heightmap(height, bounds, max_dist=.1)
        outside_idx = heightmap._faces.index.consider['outside']['x']
        math.assert_close([1, 0, 3, 2, 3, 4, 5, 6], outside_idx)  # [- 0 3 - - 4 5 6] wanted    [ -1 1    -1-1-1] wanted shifts

    def test_is_inside_1d(self):
        height = wrap([-1, 0, 0, 1], spatial('x'))
        bounds = Box(x=2, y=(-1, 1))
        heightmap = Heightmap(height, bounds, max_dist=.1)
        math.assert_close([False, True], heightmap.lies_inside(vec(x=[.5, 1.5], y=[0, 0])))

    def test_closest_surface_1d(self):
        height = wrap([-1, 0, 0, 1], spatial('x'))
        bounds = Box(x=2, y=(-1, 1))
        heightmap = Heightmap(height, bounds, max_dist=.1)
        sgn_dist, delta, normals, offsets, face_idx = heightmap.approximate_closest_surface(vec(x=[.5, 1.5], y=[0, 0]))
        math.assert_close([[0], [2]], face_idx)

    def test_creation_2d(self):
        bounds = Box(x=2, y=2, z=1)
        height = CenteredGrid(lambda pos: math.exp(-math.vec_squared(pos - 1) * 3), 0, bounds['x,y'], x=10, y=10).values
        Heightmap(height, bounds, max_dist=.1)

    def test_is_inside_2d(self):
        bounds = Box(x=2, y=2, z=1)
        height = CenteredGrid(lambda pos: math.exp(-math.vec_squared(pos - 1) * 3), 0, bounds['x,y'], x=10, y=10).values
        heightmap = Heightmap(height, bounds, max_dist=.1)
        math.assert_close([False, True], heightmap.lies_inside(vec(x=[0, 1], y=[0, 1], z=[.1, .9])))

    def test_closest_surface_2d(self):
        bounds = Box(x=2, y=2, z=1)
        height = CenteredGrid(lambda pos: math.exp(-math.vec_squared(pos - 1) * 3), 0, bounds['x,y'], x=10, y=10).values
        heightmap = Heightmap(height, bounds, max_dist=.1)
        sgn_dist, delta, normals, offsets, face_idx = heightmap.approximate_closest_surface(vec(x=[0, 1], y=[0, 1], z=[.1, .9]))
        math.assert_close([(0, 0), (4, 4)], face_idx)

