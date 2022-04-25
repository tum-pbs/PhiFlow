from unittest import TestCase

from phi import math, geom
from phi import field
from phi.field import Noise, CenteredGrid, StaggeredGrid
from phi.geom import Box, Sphere
from phi.math import extrapolation, spatial, channel, batch


class GridTest(TestCase):

    def test_create_grid_int_resolution(self):
        g = CenteredGrid(0, 0, Box(x=4, y=3), resolution=10)
        self.assertEqual(g.shape, spatial(x=10, y=10))
        g = StaggeredGrid(0, 0, Box(x=4, y=3), resolution=10)
        self.assertEqual(spatial(g), spatial(x=10, y=10))

    def test_staggered_grid_sizes_by_extrapolation(self):
        s = spatial(x=20, y=10)
        for initializer in [0, Noise(vector=2), (0, 1), Sphere((0, 0), 1)]:
            g_const = StaggeredGrid(initializer, extrapolation.ZERO, resolution=s)
            self.assertEqual(g_const.shape, spatial(x=20, y=10) & channel(vector=2))
            self.assertEqual(g_const.values.vector[0].shape, spatial(x=19, y=10))
            g_periodic = StaggeredGrid(initializer, extrapolation.PERIODIC, resolution=s)
            self.assertEqual(g_periodic.shape, spatial(x=20, y=10) & channel(vector=2))
            self.assertEqual(g_periodic.values.vector[0].shape, spatial(x=20, y=10))
            g_boundary = StaggeredGrid(initializer, extrapolation.BOUNDARY, resolution=s)
            self.assertEqual(g_boundary.shape, spatial(x=20, y=10) & channel(vector=2))
            self.assertEqual(g_boundary.values.vector[0].shape, spatial(x=21, y=10))

    def test_slice_staggered_grid_along_vector(self):
        v = StaggeredGrid(Noise(batch(batch=10)), x=10, y=20)
        x1 = v[{'vector': 0}]
        x2 = v.vector[0]
        x3 = v.vector['x']
        x4 = field.unstack(v, 'vector')[0]
        self.assertIsInstance(x1, CenteredGrid)
        field.assert_close(x1, x2, x3, x4)

    def test_slice_staggered_grid_along_batch(self):
        v = StaggeredGrid(Noise(batch(batch=10)), x=10, y=20)
        b1 = v[{'batch': 1}]
        b2 = v.batch[1]
        b3 = field.unstack(v, 'batch')[1]
        self.assertIsInstance(b1, StaggeredGrid)
        field.assert_close(b1, b2, b3)

    # def test_slice_staggered_grid_along_spatial(self):
    #     v = StaggeredGrid(Noise(batch(batch=10)), x=10, y=20)
    #     x1 = v[{'x': 1}]
    #     x2 = v.x[1]
    #     x3 = field.unstack(v, 'x')[1]
    #     self.assertIsInstance(x1, StaggeredGrid)
    #     field.assert_close(x1, x2, x3)
    #     self.assertEqual(x1.bounds, Box[1:2, 0:20])

    def test_slice_centered_grid(self):
        g = CenteredGrid(Noise(batch(batch=10), channel(vector=2)), x=10, y=20)
        s1 = g[{'vector': 0, 'batch': 1, 'x': 1}]
        s2 = g.vector[0].batch[1].x[1]
        self.assertIsInstance(s1, CenteredGrid)
        self.assertEqual(s1.bounds, Box[1:2, 0:20])
        field.assert_close(s1, s2)

    def test_staggered_grid_with_extrapolation(self):
        grid = StaggeredGrid(Noise(vector='x,y'), extrapolation.BOUNDARY, x=20, y=10)
        grid_0 = grid.with_extrapolation(extrapolation.ZERO)
        self.assertEqual(grid.resolution, grid_0.resolution)
        grid_ = grid_0.with_extrapolation(extrapolation.BOUNDARY)
        self.assertEqual(grid.resolution, grid_.resolution)
        math.assert_close(grid_.values.vector['x'].x[0], 0)
        math.assert_close(grid_.values.vector['x'].x[-1], 0)
        math.assert_close(grid_.values.vector['y'].y[0], 0)
        math.assert_close(grid_.values.vector['y'].y[-1], 0)

    def test_grid_constant_extrapolation(self):
        grid = CenteredGrid(math.random_uniform(spatial(x=50, y=10)), 0., Box[0:1, 0:1])
        self.assertEqual(grid.extrapolation, extrapolation.ZERO)
        grid = CenteredGrid(0, 0, Box[0:1, 0:1], x=50, y=10)
        self.assertEqual(grid.extrapolation, extrapolation.ZERO)
        grid = StaggeredGrid(0, 0, Box[0:1, 0:1], x=50, y=10)
        self.assertEqual(grid.extrapolation, extrapolation.ZERO)

    def test_infinite_cylinder_to_grid(self):
        cylinder = geom.infinite_cylinder(x=2, y=1.5, radius=.8, inf_dim='z')
        StaggeredGrid(cylinder, 0, x=4, y=3, z=2)

    def test_zero_vector_grid(self):
        for data in [(0, 0), Noise(vector='x,y'), Noise(vector=2), lambda x: x]:
            grid = CenteredGrid(data, 0, x=4, y=3)
            self.assertEqual(('x', 'y'), grid.values.vector.item_names)
            self.assertEqual(('x', 'y'), grid.dx.vector.item_names)

    def test_zero_staggered_grid(self):
        for data in [(0, 0), 0, Noise(), lambda x: x]:
            grid = StaggeredGrid(data, 0, x=4, y=3)
            self.assertEqual(('x', 'y'), grid.values.vector.item_names)
            self.assertEqual(('x', 'y'), grid.dx.vector.item_names)

    def test_iter_dim(self):
        slices = tuple(StaggeredGrid(0, x=4, y=3).vector)
        self.assertEqual(2, len(slices))
        self.assertFalse(slices[0].shape.non_spatial)
        self.assertEqual(('x', 'y'), slices[0].bounds.size.vector.item_names)
