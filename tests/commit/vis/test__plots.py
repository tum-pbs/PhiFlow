from unittest import TestCase

import plotly

from phi.field import CenteredGrid, StaggeredGrid, PointCloud, Noise
from phi.geom import Sphere, Box
from phi.math import extrapolation, wrap, instance, channel, batch
from phi.vis import show, overlay, plot
import matplotlib.pyplot as plt


class TestMatplotlibPlots(TestCase):

    def _test_plot(self, *plottable, show_=True, down=''):
        fig = plot(*plottable, lib='matplotlib', down=down)
        self.assertIsInstance(fig, plt.Figure)
        fig = plot(*plottable, lib='plotly', down=down)
        self.assertIsInstance(fig, plotly.graph_objs.Figure)
        if show_:
            show(gui='matplotlib')
            show(gui='plotly')

    def test_plot_scalar_grid_2d(self):
        self._test_plot(CenteredGrid(Noise(), 0, x=64, y=8, bounds=Box(0, [1, 1])))

    def test_plot_scalar_tensor_2d(self):
        self._test_plot(CenteredGrid(Noise(), 0, x=64, y=8, bounds=Box(0, [1, 1])).values)

    def test_plot_scalar_2d_batch(self):
        self._test_plot(CenteredGrid(Noise(batch(b=2)), 0, x=64, y=8, bounds=Box(0, [1, 1])))
        self._test_plot(CenteredGrid(Noise(batch(b=2)), 0, x=64, y=8, bounds=Box(0, [1, 1])), down='b')

    def test_plot_vector_grid_2d(self):
        self._test_plot(CenteredGrid(Noise(vector=2), extrapolation.ZERO, x=64, y=8, bounds=Box(0, [1, 1])) * 0.1)

    def test_plot_vector_2d_batch(self):
        self._test_plot(CenteredGrid(Noise(batch(b=2), vector=2), extrapolation.ZERO, bounds=Box[0:1, 0:1], x=10, y=10))

    def test_plot_staggered_grid_2d(self):
        self._test_plot(StaggeredGrid(Noise(), extrapolation.ZERO, x=16, y=10, bounds=Box(0, [1, 1])) * 0.1)

    def test_plot_point_cloud_2d(self):
        points = wrap([(.2, .4), (.9, .8)], instance('points'), channel('vector'))
        self._test_plot(PointCloud(Sphere(points, radius=.1)))

    def test_plot_point_cloud_bounded(self):
        points = wrap([(.2, .4), (.9, .8)], instance('points'), channel('vector'))
        self._test_plot(PointCloud(Sphere(points, radius=0.1), bounds=Box(0, [1, 1])))

    def test_plot_multiple(self):
        grid = CenteredGrid(Noise(batch(b=2)), 0, Box[0:1, 0:1], x=50, y=10)
        grid2 = CenteredGrid(grid, 0, Box[0:2, 0:1], x=20, y=50)
        points = wrap([(.2, .4), (.9, .8)], instance('points'), channel('vector'))
        cloud = PointCloud(Sphere(points, radius=0.1), bounds=Box(0, [1, 1]))
        self._test_plot(grid, grid2, cloud, down='b')

    def test_overlay(self):
        grid = CenteredGrid(Noise(), extrapolation.ZERO, x=64, y=8, bounds=Box(0, [1, 1]))
        points = wrap([(.2, .4), (.9, .8)], instance('points'), channel('vector'))
        cloud = PointCloud(Sphere(points, radius=.1))
        self._test_plot(overlay(grid, grid * (0.1, 0.02), cloud))
