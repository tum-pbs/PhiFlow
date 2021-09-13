from unittest import TestCase

from phi.field import CenteredGrid, StaggeredGrid, PointCloud, Noise
from phi.geom import Sphere, Box
from phi.math import extrapolation, wrap, instance, channel, batch
from phi.vis._matplotlib._matplotlib_plots import plot
from phi.vis import show
import matplotlib.pyplot as plt


class TestMatplotlibPlots(TestCase):

    def test_plot_scalar_grid(self):
        plt.close()
        grid = CenteredGrid(Noise(), extrapolation.ZERO, x=64, y=8, bounds=Box(0, [1, 1]))
        fig = plot(grid)
        assert isinstance(fig, plt.Figure)
        plt.show()

    def test_plot_scalar_batch(self):
        plt.close()
        grid = CenteredGrid(Noise(batch(b=2)), extrapolation.ZERO, bounds=Box[0:1, 0:1], x=10, y=10)
        fig = plot(grid)
        assert isinstance(fig, plt.Figure)
        show()

    def test_plot_vector_grid(self):
        plt.close()
        grid = CenteredGrid(Noise(vector=2), extrapolation.ZERO, x=64, y=8, bounds=Box(0, [1, 1])) * 0.1
        fig = plot(grid)
        assert isinstance(fig, plt.Figure)
        plt.show()

    def test_plot_vector_batch(self):
        plt.close()
        grid = CenteredGrid(Noise(batch(b=2), vector=2), extrapolation.ZERO, bounds=Box[0:1, 0:1], x=10, y=10)
        fig = plot(grid * 0.1)
        assert isinstance(fig, plt.Figure)
        show()

    def test_plot_staggered_grid(self):
        plt.close()
        grid = StaggeredGrid(Noise(), extrapolation.ZERO, x=16, y=10, bounds=Box(0, [1, 1])) * 0.1
        fig = plot(grid)
        assert isinstance(fig, plt.Figure)
        plt.show()

    def test_plot_point_cloud(self):
        plt.close()
        points = wrap([(.2, .4), (.9, .8)], instance('points'), channel('vector'))
        cloud = PointCloud(Sphere(points, radius=.1))
        fig = plot(cloud)
        assert isinstance(fig, plt.Figure)
        plt.show()

    def test_plot_point_cloud_bounded(self):
        plt.close()
        points = wrap([(.2, .4), (.9, .8)], instance('points'), channel('vector'))
        cloud = PointCloud(Sphere(points, radius=0.1), bounds=Box(0, [1, 1]))
        fig = plot(cloud)
        assert isinstance(fig, plt.Figure)
        plt.show()

    def test_plot_multiple(self):
        plt.close()
        grid = CenteredGrid(Noise(), extrapolation.ZERO, Box[0:1, 0:1], x=50, y=10)
        grid2 = CenteredGrid(grid, extrapolation.ZERO, Box[0:2, 0:1], x=20, y=50)
        points = wrap([(.2, .4), (.9, .8)], instance('points'), channel('vector'))
        cloud = PointCloud(Sphere(points, radius=0.1), bounds=Box(0, [1, 1]))
        fig = plot([grid, grid2, cloud])
        assert isinstance(fig, plt.Figure)
        plt.show()
