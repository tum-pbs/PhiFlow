from unittest import TestCase

import plotly

from phi import geom, field, math
from phi.field import CenteredGrid, StaggeredGrid, PointCloud, Noise, resample, Field
from phi.geom import Sphere, Box
from phiml.math import extrapolation, wrap, instance, channel, batch, spatial, vec, stack
from phi.vis import show, overlay, plot, close
import matplotlib.pyplot as plt


class TestPlots(TestCase):

    def _test_plot(self, *plottable, show_=True, **kwargs):
        fig = plot(*plottable, lib='matplotlib', **kwargs)
        self.assertIsInstance(fig.native(), plt.Figure)
        fig = plot(*plottable, lib='plotly', **kwargs)
        self.assertIsInstance(fig.native(), plotly.graph_objs.Figure)
        if show_:
            show(gui='matplotlib')
            show(gui='plotly')

    def test_bar_chart(self):
        dim = instance(planets='Sun,Earth,Mars')
        data = math.random_uniform(dim, channel(variations='Mass, Size'))
        try:
            self._test_plot(data, err=data * .1, alpha=.2)
        except NotImplementedError:
            pass
        try:
            self._test_plot(PointCloud(math.range_tensor(dim, channel(vector='planets')), data))
        except NotImplementedError:
            pass

    def test_plot_1d(self):
        self._test_plot(CenteredGrid(lambda x: math.sin(x.vector[0]), input=100, bounds=Box(input=2 * math.pi)))

    def test_plot_multi_1d(self):
        self._test_plot(CenteredGrid(lambda x: math.stack({'sin': math.sin(x), 'cos': math.cos(x)}, channel('curves')), x=100, bounds=Box(x=2 * math.pi)))

    def test_plot_scalar_grid_2d(self):
        self._test_plot(CenteredGrid(Noise(), 0, x=64, y=8, bounds=Box(x=1, y=1)))

    def test_plot_scalar_tensor_2d(self):
        self._test_plot(CenteredGrid(Noise(), 0, dim1=64, dim2=8, bounds=Box(dim1=1, dim2=1)).values)

    def test_plot_point_tensor(self):
        self._test_plot(math.vec(x=1, y=1))

    def test_plot_collection_tensor(self):
        self._test_plot(math.wrap([(0, 0), (1, 1)], instance('points'), channel(vector='x,y')))

    def test_plot_scalar_2d_batch(self):
        self._test_plot(CenteredGrid(Noise(batch(b=2)), 0, x=64, y=8, bounds=Box(x=1, y=1)))
        self._test_plot(CenteredGrid(Noise(batch(b=2)), 0, x=64, y=8, bounds=Box(x=1, y=1)), row_dims='b', size=(2, 4))

    def test_plot_vector_grid_2d(self):
        self._test_plot(CenteredGrid(Noise(vector='b,a'), extrapolation.ZERO, a=64, b=8, bounds=Box(a=1, b=1)) * 0.1)

    def test_plot_vector_2d_batch(self):
        self._test_plot(CenteredGrid(Noise(batch(b=2), vector=2), extrapolation.ZERO, bounds=Box['x,y', 0:1, 0:1], x=10, y=10))

    def test_plot_staggered_grid_2d(self):
        self._test_plot(StaggeredGrid(Noise(), extrapolation.ZERO, x=16, y=10, bounds=Box(x=1, y=1)) * 0.1)

    def test_plot_multi_grid(self):
        self._test_plot(overlay(CenteredGrid(Noise(), x=10, y=10), CenteredGrid(Noise(), x=10, y=10, bounds=Box(x=(5, 15), y=(5, 15)))))

    def test_plot_circle(self):
        self._test_plot(Sphere(x=1, y=0, radius=.5))

    def test_plot_box(self):
        self._test_plot(Box(x=(1, 2), y=1))

    def test_plot_circles(self):
        spheres = Sphere(wrap([(.2, .4), (.9, .8), (.7, .8)], instance('points'), channel(vector='x,y')), radius=.1)
        self._test_plot(spheres)

    def test_plot_point_cloud_2d(self):
        spheres = PointCloud(Sphere(wrap([(.2, .4), (.9, .8), (.7, .8)], instance('points'), channel(vector='x,y')), radius=.1))
        cells = PointCloud(geom.pack_dims(CenteredGrid(0, 0, x=3, y=3, bounds=Box['x,y', .4:.6, .2:.4]).elements, 'x,y', instance('points')))
        cloud = field.stack([spheres, cells], instance('stack'))
        self._test_plot(cloud, color=wrap(['#994444', '#000000'], instance('stack')))

    def test_plot_point_cloud_2d_large(self):
        spheres = PointCloud(Sphere(wrap([(2, 4), (9, 8), (7, 8)], instance('points'), channel(vector='x,y')), radius=1))
        cells = PointCloud(geom.pack_dims(CenteredGrid(0, 0, x=3, y=3, bounds=Box['x,y', 4:6, 2:4]).elements, 'x,y', instance('points')))
        cloud = field.stack([spheres, cells], instance('stack'))
        self._test_plot(cloud)

    def test_plot_point_cloud_2d_points(self):
        self._test_plot(PointCloud(math.random_normal(instance(points=5), channel(vector='a,b'))))

    def test_plot_point_cloud_2d_bounded(self):
        points = wrap([(.2, .4), (.9, .8)], instance('points'), channel(vector='x,y'))
        self._test_plot(PointCloud(Sphere(points, radius=0.1), bounds=Box(x=1, y=1)))

    def test_plot_point_cloud_vector_field_2d_bounded(self):
        points = math.random_uniform(instance(points='a,b,c,d,e'), channel(vector='x,y'))
        velocity = PointCloud(Sphere(points, radius=.1), bounds=Box(x=1, y=1))
        self._test_plot(velocity * (.05, 0))

    def test_plot_multiple(self):
        grid = CenteredGrid(Noise(batch(b=2)), 0, Box['x,y', 0:1, 0:1], x=50, y=10)
        grid2 = CenteredGrid(grid, 0, Box['x,y', 0:2, 0:1], x=20, y=50)
        points = wrap([(.2, .4), (.9, .8)], instance('points'), channel(vector='x,y'))
        cloud = PointCloud(Sphere(points, radius=0.1), bounds=Box(x=1, y=1))
        titles = math.wrap([['b=0', 'b=0', 'points'], ['b=1', 'b=1', '']], spatial('rows,cols'))
        self._test_plot(grid, grid2, cloud, row_dims='b', title=titles)

    def test_overlay(self):
        grid = CenteredGrid(Noise(), extrapolation.ZERO, x=64, y=8, bounds=Box(x=1, y=1))
        points = wrap([(.2, .4), (.9, .8)], instance('points'), channel(vector='x,y'))
        cloud = PointCloud(Sphere(points, radius=.1))
        self._test_plot(overlay(grid, grid * (0.1, 0.02), cloud), title='Overlay')

    def test_plot_density_3d_batched(self):
        sphere = resample(Sphere(x=.5, y=.5, z=.5, radius=.4), CenteredGrid(0, x=10, y=10, z=10, bounds=Box(x=1, y=1, z=1)), soft=True)
        cylinder = CenteredGrid(geom.infinite_cylinder(x=16, y=16, inf_dim='z', radius=10), x=32, y=32, z=32)
        self._test_plot(sphere, cylinder)

    def test_plot_vector_3d_batched(self):
        sphere = resample(Sphere(x=.5, y=.5, z=.5, radius=.4), CenteredGrid(0, x=10, y=10, z=10, bounds=Box(x=1, y=1, z=1)), soft=True) * (.1, 0, 0)
        cylinder = CenteredGrid(geom.infinite_cylinder(x=16, y=16, inf_dim='z', radius=10), x=32, y=32, z=32) * (0, 0, .1)
        self._test_plot(sphere, cylinder)

    def test_plot_point_cloud_3d(self):
        points = math.random_uniform(instance(points=50), channel(vector='a,b,c'))
        cloud = PointCloud(Sphere(points, radius=.1), bounds=Box(a=2, b=1, c=1))
        self._test_plot(cloud)

    def test_plot_staggered_grid_3d(self):
        self._test_plot(StaggeredGrid(Noise(), extrapolation.ZERO, x=16, y=10, z=10, bounds=Box(x=1, y=1, z=1)) * 0.1)

    def test_plot_point_cloud_3d_points(self):
        self._test_plot(PointCloud(math.random_normal(instance(points=5), channel(vector='x,y,z'))))

    def test_same_scale(self):
        curve1 = vec(x=(0, 1, 2), y=(0, 1, 0))
        curve2 = vec(x=(1, 1, 2), y=(-1, .5, .5))
        try:
            self._test_plot(curve1, curve2, same_scale='y')
        except NotImplementedError:
            pass

    def test_plot_arbitrary_lines(self):
        points = vec(resolution=wrap([0, 1, 4], spatial('line')), error=wrap([0, 1, .5], spatial('line')))
        points = stack([points, points + (0, -1)], instance('disconnected'))
        points = stack([points, points * (1, -1)], channel('categories'))
        try:
            self._test_plot(PointCloud(points), color=wrap([0, 1], channel('categories')))
        except NotImplementedError:
            pass

    def test_plot_error_bars(self):
        self._test_plot(vec(x=[1, 2, 3], y=[0, 2, 1]), err=vec(x=.1, y=[0, 0, .2]))

    def test_plot_line_errors(self):
        data = CenteredGrid(Noise(scale=100), x=100)
        self._test_plot(data, color=1, alpha=1, err=data.values * .1)
        self._test_plot(data, color=1, alpha=1, err=.2)

    def test_plot_error_envelopes(self):
        x = math.linspace(0, 3, spatial(x=100))
        y = math.sin(x)
        try:
            self._test_plot(vec(x=x, y=y), err=vec(x=0, y=abs(y) * .5), color=1, alpha=.9)
        except NotImplementedError:
            pass
        try:
            self._test_plot(vec(x=y, y=x), err=vec(x=abs(y) * .5, y=0), color=1, alpha=.9)
        except NotImplementedError:
            pass

    def test_plot_graph(self):
        points = wrap([(0, 0), (1.2, 0), (2, 0), (2, 1), (.8, 1), (0, 1)], instance('points'), channel(vector='x,y'))
        dense_edges = math.vec_length(math.pairwise_distances(points, 1.5))
        dense_graph = geom.Graph(points, dense_edges, {})
        sparse_edges = math.vec_length(math.pairwise_distances(points, 1.5, format='csr'))
        sparse_graph = geom.Graph(points, sparse_edges, {})
        try:
            self._test_plot(dense_graph, sparse_graph, Field(sparse_graph, lambda x: math.vec_length(x), 0))
        except NotImplementedError:
            pass

    def test_animate(self):
        values = math.random_uniform(batch(time=3), spatial(x=32, y=32))
        anim = plot(values, animate='time', show_color_bar=False, frame_time=100, lib='matplotlib')
        # anim.to_html5_video()
        # anim.save('animation.mp4')

    def test_close(self):
        fig = plot(Sphere(x=0, y=0, radius=5))
        close(fig)
        close()
