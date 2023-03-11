from unittest import TestCase

from phi import math
from phi.field import CenteredGrid, PointCloud
from phi.math import spatial, channel, instance
from phi.vis._vis_base import to_field


class TestVisBase(TestCase):

    def test_tensor_as_field(self):
        # --- Grid ---
        t = math.random_normal(spatial(x=4, y=3), channel(vector='x,y'))
        grid = to_field(t)
        self.assertIsInstance(grid, CenteredGrid)
        math.assert_close(grid.dx, 1)
        math.assert_close(grid.points.x[0].y[0], 0)
        # --- PointCloud ---
        t = math.random_normal(instance(points=5), channel(vector='x,y'))
        points = to_field(t)
        self.assertIsInstance(points, PointCloud)
        # --- Arbitrary lines ---
        t = math.random_normal(spatial(points=5), channel(vector='x,y'))
        points = to_field(t)
        self.assertIsInstance(points, PointCloud)
