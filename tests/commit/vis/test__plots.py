from unittest import TestCase

import plotly

from phi import geom, field, math
from phi.field import CenteredGrid, StaggeredGrid, PointCloud, Noise, SoftGeometryMask
from phi.geom import Sphere, Box
from phi.math import extrapolation, wrap, instance, channel, batch, spatial
from phi.vis import show, overlay, plot
import matplotlib.pyplot as plt


class TestPlots(TestCase):

    def _test_plot(self, *plottable, show_=True, **kwargs):
        fig = plot(*plottable, lib='matplotlib', **kwargs)
        self.assertIsInstance(fig, plt.Figure)
        fig = plot(*plottable, lib='plotly', **kwargs)
        self.assertIsInstance(fig, plotly.graph_objs.Figure)
        if show_:
            show(gui='matplotlib')
            show(gui='plotly')

    def test_plot_1d(self):
        self._test_plot(CenteredGrid(lambda x: math.sin(x), x=100, bounds=Box(x=2 * math.pi)))

    def test_plot_multi_1d(self):
        self._test_plot(CenteredGrid(lambda x: math.stack({'sin': math.sin(x), 'cos': math.cos(x)}, channel('curves')), x=100, bounds=Box(x=2 * math.pi)))

    def test_plot_scalar_grid_2d(self):
        self._test_plot(CenteredGrid(Noise(), 0, x=64, y=8, bounds=Box(0, [1, 1])))

    def test_plot_scalar_tensor_2d(self):
        self._test_plot(CenteredGrid(Noise(), 0, x=64, y=8, bounds=Box(0, [1, 1])).values)

    def test_plot_point_tensor(self):
        self._test_plot(wrap((1, 1)))

    def test_plot_collection_tensor(self):
        self._test_plot(math.wrap([(0, 0), (1, 1)], instance('points'), channel(vector='x,y')))

    def test_plot_scalar_2d_batch(self):
        self._test_plot(CenteredGrid(Noise(batch(b=2)), 0, x=64, y=8, bounds=Box(0, [1, 1])))
        self._test_plot(CenteredGrid(Noise(batch(b=2)), 0, x=64, y=8, bounds=Box(0, [1, 1])), row_dims='b', size=(2, 4))

    def test_plot_vector_grid_2d(self):
        self._test_plot(CenteredGrid(Noise(vector=2), extrapolation.ZERO, x=64, y=8, bounds=Box(0, [1, 1])) * 0.1)

    def test_plot_vector_2d_batch(self):
        self._test_plot(CenteredGrid(Noise(batch(b=2), vector=2), extrapolation.ZERO, bounds=Box[0:1, 0:1], x=10, y=10))

    def test_plot_staggered_grid_2d(self):
        self._test_plot(StaggeredGrid(Noise(), extrapolation.ZERO, x=16, y=10, bounds=Box(0, [1, 1])) * 0.1)

    def test_plot_spheres_2d(self):
        spheres = Sphere(wrap([(.2, .4), (.9, .8), (.7, .8)], instance('points'), channel('vector')), radius=.1)
        self._test_plot(spheres)

    def test_plot_point_cloud_2d(self):
        spheres = PointCloud(Sphere(wrap([(.2, .4), (.9, .8), (.7, .8)], instance('points'), channel('vector')), radius=.1))
        cells = PointCloud(geom.pack_dims(CenteredGrid(0, 0, x=3, y=3, bounds=Box[.4:.6, .2:.4]).elements, 'x,y', instance('points')))
        cloud = field.stack([spheres, cells], instance('stack'))
        self._test_plot(cloud)

    def test_plot_point_cloud_2d_large(self):
        spheres = PointCloud(Sphere(wrap([(2, 4), (9, 8), (7, 8)], instance('points'), channel('vector')), radius=1))
        cells = PointCloud(geom.pack_dims(CenteredGrid(0, 0, x=3, y=3, bounds=Box[4:6, 2:4]).elements, 'x,y', instance('points')))
        cloud = field.stack([spheres, cells], instance('stack'))
        self._test_plot(cloud)

    def test_plot_point_cloud_2d_points(self):
        self._test_plot(PointCloud(math.random_normal(instance(points=5), channel(vector='x,y'))))

    def test_plot_point_cloud_2d_bounded(self):
        points = wrap([(.2, .4), (.9, .8)], instance('points'), channel('vector'))
        self._test_plot(PointCloud(Sphere(points, radius=0.1), bounds=Box(0, [1, 1])))

    def test_plot_point_cloud_vector_field_2d_bounded(self):
        points = math.random_uniform(instance(points='a,b,c,d,e'), channel(vector='x,y'))
        velocity = PointCloud(Sphere(points, radius=.1), bounds=Box(x=1, y=1))
        self._test_plot(velocity * (.05, 0))

    def test_plot_multiple(self):
        grid = CenteredGrid(Noise(batch(b=2)), 0, Box[0:1, 0:1], x=50, y=10)
        grid2 = CenteredGrid(grid, 0, Box[0:2, 0:1], x=20, y=50)
        points = wrap([(.2, .4), (.9, .8)], instance('points'), channel('vector'))
        cloud = PointCloud(Sphere(points, radius=0.1), bounds=Box(0, [1, 1]))
        titles = math.wrap([['b=0', 'b=0', 'points'], ['b=1', 'b=1', '']], spatial('rows,cols'))
        self._test_plot(grid, grid2, cloud, row_dims='b', title=titles)

    def test_overlay(self):
        grid = CenteredGrid(Noise(), extrapolation.ZERO, x=64, y=8, bounds=Box(0, [1, 1]))
        points = wrap([(.2, .4), (.9, .8)], instance('points'), channel('vector'))
        cloud = PointCloud(Sphere(points, radius=.1))
        self._test_plot(overlay(grid, grid * (0.1, 0.02), cloud), title='Overlay')

    def test_plot_density_3d_batched(self):
        sphere = CenteredGrid(SoftGeometryMask(Sphere(x=.5, y=.5, z=.5, radius=.4)), x=10, y=10, z=10, bounds=Box(x=1, y=1, z=1))
        cylinder = CenteredGrid(geom.infinite_cylinder(x=16, y=16, inf_dim='z', radius=10), x=32, y=32, z=32)
        self._test_plot(sphere, cylinder)

    def test_plot_vector_3d_batched(self):
        sphere = CenteredGrid(SoftGeometryMask(Sphere(x=.5, y=.5, z=.5, radius=.4)), x=10, y=10, z=10, bounds=Box(x=1, y=1, z=1)) * (.1, 0, 0)
        cylinder = CenteredGrid(geom.infinite_cylinder(x=16, y=16, inf_dim='z', radius=10), x=32, y=32, z=32) * (0, 0, .1)
        self._test_plot(sphere, cylinder)

    def test_plot_point_cloud_3d(self):
        points = math.random_uniform(instance(points=50), channel(vector=3))
        cloud = PointCloud(Sphere(points, radius=.1), bounds=Box(x=2, y=1, z=1))
        self._test_plot(cloud)

    def test_plot_point_cloud_3d_points(self):
        self._test_plot(PointCloud(math.random_normal(instance(points=5), channel(vector='x,y,z'))))

    def test_animate(self):
        values = math.random_uniform(batch(time=3), spatial(x=32, y=32))
        anim = plot(values, animate='time', show_color_bar=False, frame_time=100, lib='matplotlib')
        anim.to_html5_video()
        # anim.save('animation.mp4')
