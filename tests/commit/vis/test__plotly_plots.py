from unittest import TestCase

from phi.field import CenteredGrid, StaggeredGrid, PointCloud, Noise
from phi.geom import Sphere, Box
from phi.math import extrapolation, wrap, instance, channel
from phi.vis._dash._plotly_plots import plot
import plotly


class TestMatplotlibPlots(TestCase):

    def test_plot_scalar_grid(self):
        grid = CenteredGrid(Noise(), extrapolation.ZERO, x=64, y=8, bounds=Box(0, [1, 1]))
        fig = plot(grid)
        assert isinstance(fig, plotly.graph_objs.Figure)
        fig.show()

    def test_plot_vector_grid(self):
        grid = CenteredGrid(Noise(vector=2), extrapolation.ZERO, x=64, y=8, bounds=Box(0, [1, 1])) * 0.1
        fig = plot(grid)
        assert isinstance(fig, plotly.graph_objs.Figure)
        fig.show()

    def test_plot_staggered_grid(self):
        grid = StaggeredGrid(Noise(), extrapolation.ZERO, x=16, y=10, bounds=Box(0, [1, 1])) * 0.1
        fig = plot(grid)
        assert isinstance(fig, plotly.graph_objs.Figure)
        fig.show()

    def test_plot_point_cloud(self):
        points = wrap([(.2, .4), (.9, .8)], instance('points'), channel('vector'))
        cloud = PointCloud(Sphere(points, radius=0.1))
        fig = plot(cloud)
        assert isinstance(fig, plotly.graph_objs.Figure)
        fig.show()

    def test_plot_point_cloud_bounded(self):
        points = wrap([(.2, .4), (.9, .8)], instance('points'), channel('vector'))
        cloud = PointCloud(Sphere(points, radius=0.1), bounds=Box(0, [1, 1]))
        fig = plot(cloud)
        assert isinstance(fig, plotly.graph_objs.Figure)
        fig.show()
