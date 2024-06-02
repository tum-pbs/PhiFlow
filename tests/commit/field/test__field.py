from unittest import TestCase

from phi import math, geom
from phi.field import CenteredGrid, Noise, assert_close, AngularVelocity, StaggeredGrid, resample, Field
from phi.geom import Box, Sphere, Point
from phiml.math import batch, spatial, vec, channel
from phiml.math.extrapolation import ZERO_GRADIENT


class TestField(TestCase):

    def test_concat(self):
        grids = [CenteredGrid(Noise(batch(b=1)), x=32, y=32) for _ in range(2)]
        concat = math.concat(grids, 'b')
        self.assertEqual(2, concat.b.size)

    def test_stack(self):
        grids = [CenteredGrid(Noise(), x=32, y=32) for _ in range(2)]
        b_stack = math.stack(grids, batch('b'))
        self.assertEqual(2, b_stack.b.size)
        z_stack = math.stack(grids, spatial('z'), bounds=Box(z=.2))
        math.assert_close(.2, z_stack.bounds.size.vector['z'])
        z_stack = math.stack(grids, spatial('z'))
        math.assert_close(2, z_stack.bounds.size.vector['z'])

    def test_embedding(self):
        grid = CenteredGrid(lambda x: math.vec_length(x), 1, x=4, y=4)
        math.print(grid.values, "Base Grid")
        inner = CenteredGrid(0, grid, x=2, y=2, bounds=Box(x=(1, 3), y=(1, 3)))
        resampled = inner @ grid
        math.assert_close(inner.values, resampled.values.x[1:-1].y[1:-1])
        math.assert_close(grid.values.x[:1].y[1:-1], resampled.values.x[:1].y[1:-1])
        math.assert_close(grid.values.x[-1:].y[1:-1], resampled.values.x[-1:].y[1:-1])
        math.assert_close(grid.values.x[1:-1].y[:1], resampled.values.x[1:-1].y[:1])
        math.assert_close(grid.values.x[1:-1].y[-1:], resampled.values.x[1:-1].y[-1:])

    def test_embedding_resample(self):
        p = CenteredGrid(Noise(), x=10, y=10, bounds=Box(x=100, y=100))
        p_emb_x0 = CenteredGrid(p, p, x=5, y=5, bounds=Box(x=(10, 60), y=(10, 60)))
        p_back = CenteredGrid(p_emb_x0, p.extrapolation, p.bounds, p.resolution)
        assert_close(0, p_back - p)

    def test_slice_str(self):
        vgrid = CenteredGrid(Noise(col='r,g,b'), x=4, y=3)
        math.assert_close(vgrid.col[0], vgrid['r'])
        matrix_grid = CenteredGrid(Noise(col='r,g,b', vec='x,y'), x=4, y=3)
        math.assert_close(matrix_grid.col[(0, 1)].vec['y'], matrix_grid['r,g', 'y'])

    def test_legacy_resampling(self):
        for obj in [AngularVelocity(location=vec(x=0, y=0)), Sphere(x=0, y=0, radius=1)]:
            resampled = obj >> CenteredGrid(0, x=4, y=3)
            self.assertTrue(resampled.is_grid)

    def test_boundary_slicing(self):
        v = StaggeredGrid(0, vec(x=1, y=-1), x=10, y=10)
        self.assertIn('vector', v.boundary.shape)
        components = math.unstack(v, 'vector')
        self.assertNotIn('vector', components[0].boundary.shape)

    def test_numpy(self):
        g = CenteredGrid(Noise(channel(vector='x,y')), 1, x=10, y=8)
        self.assertEqual((10, 8, 2), g.numpy().shape)
        g = StaggeredGrid(Noise(), 1, x=10, y=8)
        self.assertEqual((9, 8), g.numpy()[0].shape)
        self.assertEqual((10, 7), g.numpy()[1].shape)

    def test_stack_boundaries(self):
        b1 = CenteredGrid(1, 0, x=4)
        b2 = b1 + 1
        f1 = CenteredGrid(0, b1, x=2)
        f2 = CenteredGrid(0, b2, x=2)
        f = math.stack([f1, f2], batch('b'))
        full = resample(f, b1)
        math.assert_close([0, 0, 1, 1], full.values.b[0])
        math.assert_close([0, 0, 2, 2], full.values.b[1])

    def test_as_points(self):
        values = math.wrap([[1, 2], [3, 4]], spatial('x,y'))
        grid = CenteredGrid(values, 1)
        points = grid.as_points()
        math.assert_close([1, 2, 3, 4], points.values)
        self.assertIsInstance(points.geometry, Point)
        math.assert_close(math.flatten(grid.points), math.flatten(points.points))

    def test_as_spheres(self):
        values = math.wrap([[1, 2], [3, 4]], spatial('x,y'))
        grid = CenteredGrid(values, 1)
        spheres = grid.as_spheres()
        math.assert_close([1, 2, 3, 4], spheres.values)
        self.assertIsInstance(spheres.geometry, Sphere)
        math.assert_close(1, spheres.geometry.volume)
        self.assertEqual(math.EMPTY_SHAPE, spheres.geometry.volume.shape)
        math.assert_close(math.flatten(grid.points), math.flatten(spheres.points))

    def test_mesh_to_grid(self):
        domain = Box(x=2, y=1)
        resolution = spatial(x=30, y=10)
        mesh = geom.build_mesh(domain, resolution)
        v = Field(mesh, vec(x=0, y=0), {'x-': vec(x=1, y=0), 'x+': ZERO_GRADIENT, 'y': 0})
        grid = v.to_grid()
        self.assertEqual(resolution, grid.resolution)
        self.assertEqual(grid.bounds, mesh.bounds)
        grid = v.to_grid(x=10, y=10)
        self.assertEqual(resolution.with_sizes(10), grid.resolution)
        self.assertEqual(grid.bounds, mesh.bounds)
