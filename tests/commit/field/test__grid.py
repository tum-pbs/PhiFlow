from unittest import TestCase

from phi import math, geom
from phi import field
from phi.field import Noise, CenteredGrid, StaggeredGrid, Field
from phi.geom import Box, Sphere
from phiml.math import extrapolation, spatial, channel, batch
from phiml.math.magic import PhiTreeNode


class GridTest(TestCase):

    def test_create_grid_int_resolution(self):
        g = CenteredGrid(0, 0, Box(x=4, y=3), resolution=10)
        self.assertEqual(g.shape, spatial(x=10, y=10))
        g = StaggeredGrid(0, 0, Box(x=4, y=3), resolution=10)
        self.assertEqual(spatial(g), spatial(x=10, y=10))

    def test_create_grid_float_bounds(self):
        g = CenteredGrid(0, 0, 2, x=10, y=20)
        math.assert_close(2, g.bounds.size)
        self.assertEqual(('x', 'y'), g.bounds.vector.item_names)
        math.assert_close((0.2, 0.1), g.dx)

    def test_staggered_grid_sizes_by_extrapolation(self):
        s = spatial(x=20, y=10)
        for initializer in [0, Noise(vector=2), (0, 1), Sphere(x=0, y=0, radius=1)]:
            g_const = StaggeredGrid(initializer, extrapolation.ZERO, resolution=s)
            self.assertEqual(g_const.shape, spatial(x=20, y=10) & channel(vector='x,y'))
            self.assertEqual(spatial(x=19, y=10), g_const.vector[0].values.shape)
            g_periodic = StaggeredGrid(initializer, extrapolation.PERIODIC, resolution=s)
            self.assertEqual(g_periodic.shape, spatial(x=20, y=10) & channel(vector='x,y'))
            self.assertEqual(g_periodic.vector[0].values.shape, spatial(x=20, y=10))
            g_boundary = StaggeredGrid(initializer, extrapolation.BOUNDARY, resolution=s)
            self.assertEqual(g_boundary.shape, spatial(x=20, y=10) & channel(vector='x,y'))
            self.assertEqual(g_boundary.vector[0].values.shape, spatial(x=21, y=10))

    def test_slice_staggered_grid_along_vector(self):
        v = StaggeredGrid(Noise(batch(batch=10)), x=10, y=20)
        x1 = v[{'vector': 0}]
        x2 = v.vector[0]
        x3 = v.vector['x']
        x4 = field.unstack(v, 'vector')[0]
        self.assertTrue(x1.is_grid)
        self.assertTrue(x1.is_centered)
        field.assert_close(x1, x2, x3, x4)

    def test_slice_staggered_grid_along_batch(self):
        v = StaggeredGrid(Noise(batch(batch=10)), x=10, y=20)
        b1 = v[{'batch': 1}]
        b2 = v.batch[1]
        b3 = field.unstack(v, 'batch')[1]
        self.assertTrue(b1.is_grid)
        self.assertTrue(b1.is_staggered)
        field.assert_close(b1, b2, b3)

    def test_slice_staggered_grid_keeping_staggered(self):
        grid = StaggeredGrid(0, x=4, y=4, z=4)
        g1d = grid[{'z': 0, 'y': 0, 'vector': 'x'}]
        self.assertTrue(g1d.is_grid)
        self.assertTrue(g1d.is_staggered)
        self.assertEqual(spatial(x=4), g1d.resolution)
        g2d = grid[{'z': 0, 'vector': 'x,y'}]
        self.assertTrue(g2d.is_grid)
        self.assertTrue(g2d.is_staggered)
        self.assertEqual(spatial(x=4, y=4), g2d.resolution)

    # def test_slice_staggered_grid_along_spatial(self):
    #     v = StaggeredGrid(Noise(batch(batch=10)), x=10, y=20)
    #     x1 = v[{'x': 1}]
    #     x2 = v.x[1]
    #     x3 = field.unstack(v, 'x')[1]
    #     self.assertIsInstance(x1, StaggeredGrid)
    #     field.assert_close(x1, x2, x3)
    #     self.assertEqual(x1.bounds, Box['x,y', 1:2, 0:20])

    def test_slice_centered_grid(self):
        g = CenteredGrid(Noise(batch(batch=10), channel(vector=2)), x=10, y=20)
        s1 = g[{'vector': 0, 'batch': 1, 'x': 1}]
        s2 = g.vector[0].batch[1].x[1]
        self.assertTrue(s1.is_grid)
        self.assertEqual(Box(y=20), s1.bounds.vector['y'])
        field.assert_close(s1, s2)

    def test_staggered_grid_with_extrapolation(self):
        grid = StaggeredGrid(Noise(vector='x,y'), extrapolation.BOUNDARY, x=20, y=10)
        grid_0 = grid.with_extrapolation(extrapolation.ZERO)
        self.assertEqual(grid.resolution, grid_0.resolution)
        grid_ = grid_0.with_extrapolation(extrapolation.BOUNDARY)
        self.assertEqual(grid.resolution, grid_.resolution)
        math.assert_close(grid_.values.vector.dual['x'].x[0], 0)
        math.assert_close(grid_.values.vector.dual['x'].x[-1], 0)
        math.assert_close(grid_.values.vector.dual['y'].y[0], 0)
        math.assert_close(grid_.values.vector.dual['y'].y[-1], 0)

    def test_grid_constant_extrapolation(self):
        grid = CenteredGrid(math.random_uniform(spatial(x=50, y=10)), 0., Box['x,y', 0:1, 0:1])
        self.assertEqual(grid.extrapolation, extrapolation.ZERO)
        grid = CenteredGrid(0, 0, Box['x,y', 0:1, 0:1], x=50, y=10)
        self.assertEqual(grid.extrapolation, extrapolation.ZERO)
        grid = StaggeredGrid(0, 0, Box['x,y', 0:1, 0:1], x=50, y=10)
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
            self.assertEqual(('x', 'y'), grid.values.vector.dual.item_names)
            self.assertEqual(('x', 'y'), grid.dx.vector.item_names)

    def test_staggered_grid_from_uniform_values(self):
        for ext in [0, extrapolation.PERIODIC, extrapolation.BOUNDARY]:
            print(ext)
            grid = StaggeredGrid(1, ext, x=10, y=10)
            grid_ = StaggeredGrid(grid.uniform_values(), ext, x=10, y=10)
            self.assertEqual(grid.shape, grid_.shape)
            grid_ = StaggeredGrid(grid.uniform_values(), ext)
            self.assertEqual(grid.shape, grid_.shape)

    def test_iter_dim(self):
        slices = tuple(StaggeredGrid(0, x=4, y=3).vector)
        self.assertEqual(2, len(slices))
        self.assertFalse(slices[0].shape.non_spatial)
        self.assertEqual(('x', 'y'), slices[0].bounds.size.vector.item_names)

    def test_sample_function(self):
        self.assertEqual(spatial(x=5), CenteredGrid(lambda x: x, x=5).shape)
        self.assertEqual(spatial(x=5, y=4), CenteredGrid(lambda x, y=0: x+y, x=5, y=4).shape)
        self.assertEqual(spatial(x=5, y=4) & channel(vector='x,y'), CenteredGrid(lambda pos: pos, x=5, y=4).shape)
        try:
            self.assertEqual(spatial(x=5, y=4), CenteredGrid(lambda a, b: a+b, x=5, y=4).shape)
            raise RuntimeError
        except AssertionError:
            pass
        # Varargs
        grid = CenteredGrid(lambda *x: x[0], x=10, y=10)
        math.assert_close(grid.points['x'], grid.values)
        grid = CenteredGrid(lambda t, x, y: t, t=5, x=10, y=10)
        math.assert_close(grid.points['t'], grid.values)

    def test_is_phi_tree_node(self):
        self.assertTrue(issubclass(Field, PhiTreeNode))
        grid = CenteredGrid(0, x=4)
        self.assertTrue(isinstance(grid, PhiTreeNode))

    def test_reshape_centered_grid(self):
        grid = math.expand(CenteredGrid(1, x=10, y=10), batch(b=100))
        grid = math.rename_dims(grid, 'b', 'bat')
        self.assertEqual(batch(bat=100) & spatial(x=10, y=10), grid.shape)

    def test_reshape_staggered_grid(self):
        grid = math.expand(StaggeredGrid(1, x=10, y=10), batch(b=100))
        grid = math.rename_dims(grid, 'b', 'bat')
        self.assertEqual(batch(bat=100) & spatial(x=10, y=10) & channel(vector='x,y'), grid.shape)

    def test_with_values_staggered(self):
        grid = StaggeredGrid(Noise(), 1, x=10, y=10)
        grid0 = grid.with_values(0)
        self.assertEqual({'x', 'y', '~vector'}, set(grid0.values.shape.names))
        cgrid = grid.with_values(math.zeros(spatial(x=10, y=10)))
        self.assertEqual({'x', 'y'}, set(cgrid.values.shape.names))

