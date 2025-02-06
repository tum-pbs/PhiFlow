from unittest import TestCase

from phi import math
from phi.field import CenteredGrid, PointCloud
from phi.math import spatial, channel, instance
from phi.vis._vis_base import to_field


class TestVisBase(TestCase):

    def test_tensor_as_field(self):
        # --- Grid ---
        t = math.random_normal(spatial(x=4, y=3))
        grid = to_field(t)
        self.assertTrue(grid.is_grid)
        self.assertTrue(grid.is_centered)
        math.assert_close(grid.dx, 1)
        math.assert_close(grid.points.x[0].y[0], 0)
        # --- PointCloud ---
        t = math.random_normal(instance(points=5), channel(vector='x,y'))
        points = to_field(t)
        self.assertTrue(points.is_point_cloud)
        # --- Arbitrary lines ---
        t = math.random_normal(spatial(points=5), channel(vector='x,y'))
        points = to_field(t)
        self.assertTrue(points.is_point_cloud)
