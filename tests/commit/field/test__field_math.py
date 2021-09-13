from unittest import TestCase

import numpy

import phi
from phi import math, geom
from phi.field import StaggeredGrid, CenteredGrid, HardGeometryMask
from phi.geom import Box
from phi import field
from phi.math import Solve, extrapolation, instance, channel, spatial, batch
from phi.math.backend import Backend
from phi.physics._boundaries import Domain


BACKENDS = phi.detect_backends()


class TestFieldMath(TestCase):

    def test_spatial_gradient(self):
        domain = Domain(x=4, y=3)
        phi = domain.grid() * (1, 2)
        grad = field.spatial_gradient(phi, stack_dim=channel('spatial_gradient'))
        self.assertEqual(('spatial', 'spatial', 'channel', 'channel'), grad.shape.types)

    def test_spatial_gradient_batched(self):
        bounds = geom.stack([Box[0:1, 0:1], Box[0:10, 0:10]], batch('batch'))
        grid = CenteredGrid(0, extrapolation.ZERO, bounds, x=10, y=10)
        grad = field.spatial_gradient(grid)
        self.assertIsInstance(grad, CenteredGrid)

    def test_laplace_batched(self):
        bounds = geom.stack([Box[0:1, 0:1], Box[0:10, 0:10]], batch('batch'))
        grid = CenteredGrid(0, extrapolation.ZERO, bounds, x=10, y=10)
        lap = field.laplace(grid)
        self.assertIsInstance(lap, CenteredGrid)

    def test_divergence_centered(self):
        v = CenteredGrid(1, extrapolation.ZERO, bounds=Box[0:1, 0:1], x=3, y=3) * (1, 0)  # flow to the right
        div = field.divergence(v).values
        math.assert_close(div.y[0], (1.5, 0, -1.5))

    def test_trace_function(self):
        def f(x: StaggeredGrid, y: CenteredGrid):
            return x + (y @ x)

        ft = field.jit_compile(f)
        domain = Domain(x=4, y=3)
        x = domain.staggered_grid(1)
        y = domain.vector_grid(1)

        res_f = f(x, y)
        res_ft = ft(x, y)
        self.assertEqual(res_f.shape, res_ft.shape)
        field.assert_close(res_f, res_ft)

    def test_gradient_function(self):
        def f(x: StaggeredGrid, y: CenteredGrid):
            pred = x + (y @ x)
            loss = field.l2_loss(pred)
            return loss

        grad = field.functional_gradient(f, get_output=False)
        fgrad = field.functional_gradient(f, (0, 1), get_output=True)

        domain = Domain(x=4, y=3)
        x = domain.staggered_grid(1)
        y = domain.vector_grid(1)

        for backend in BACKENDS:
            if backend.supports(Backend.gradients):
                with backend:
                    dx, = grad(x, y)
                    self.assertIsInstance(dx, StaggeredGrid)
                    loss, (dx, dy) = fgrad(x, y)
                    self.assertIsInstance(loss, math.Tensor)
                    self.assertIsInstance(dx, StaggeredGrid)
                    self.assertIsInstance(dy, CenteredGrid)

    def test_upsample_downsample_centered_1d(self):
        grid = Domain(x=4).scalar_grid([0, 1, 2, 3])
        upsampled = field.upsample2x(grid)
        downsampled = field.downsample2x(upsampled)
        math.assert_close(downsampled.values.x[1:-1], grid.values.x[1:-1])

    def test_downsample_staggered_2d(self):
        grid = Domain(x=32, y=40).staggered_grid(1)
        downsampled = field.downsample2x(grid)
        self.assertEqual(set(spatial(x=16, y=20) & channel(vector=2)), set(downsampled.shape))

    def test_abs(self):
        grid = Domain(x=4, y=3).staggered_grid(-1)
        field.assert_close(field.abs(grid), abs(grid), 1)

    def test_sign(self):
        grid = Domain(x=4, y=3).staggered_grid(0.5)
        field.assert_close(field.sign(grid), 1)

    def test_round_(self):
        grid = Domain(x=4, y=3).staggered_grid(1.7)
        field.assert_close(field.round(grid), 2)

    def test_ceil(self):
        grid = Domain(x=4, y=3).staggered_grid(1.1)
        field.assert_close(field.ceil(grid), 2)

    def test_floor(self):
        grid = Domain(x=4, y=3).staggered_grid(2.8)
        field.assert_close(field.floor(grid), 2)

    def test_sqrt(self):
        grid = Domain(x=4, y=3).staggered_grid(2)
        field.assert_close(field.sqrt(grid), numpy.sqrt(2))

    def test_exp(self):
        grid = Domain(x=4, y=3).staggered_grid(0)
        field.assert_close(field.exp(grid), 1)

    def test_isfinite(self):
        grid = Domain(x=4, y=3).staggered_grid(1)
        field.assert_close(field.isfinite(grid), True)

    def test_real(self):
        grid = Domain(x=4, y=3).staggered_grid(1)
        field.assert_close(field.real(grid), grid)

    def test_imag(self):
        grid = Domain(x=4, y=3).staggered_grid(1)
        field.assert_close(field.imag(grid), 0)

    def test_sin(self):
        grid = Domain(x=4, y=3).staggered_grid(0)
        field.assert_close(field.sin(grid), 0)

    def test_cos(self):
        grid = Domain(x=4, y=3).staggered_grid(0)
        field.assert_close(field.cos(grid), 1)

    def test_convert_grid(self):
        grid = Domain(x=4, y=3).scalar_grid(0)
        for backend in BACKENDS:
            converted = field.convert(grid, backend)
            self.assertEqual(converted.values.default_backend, backend)

    def test_convert_point_cloud(self):
        points = Domain(x=4, y=3).points(math.random_uniform(instance(points=4) & channel(vector=2))).with_values(math.random_normal(instance(points=4) & channel(vector=2)))
        for backend in BACKENDS:
            converted = field.convert(points, backend)
            self.assertEqual(converted.values.default_backend, backend)
            self.assertEqual(converted.elements.center.default_backend, backend)
            self.assertEqual(converted.elements.radius.default_backend, backend)

    def test_center_of_mass(self):
        density = Domain(x=4, y=3).scalar_grid(HardGeometryMask(Box[0:1, 1:2]))
        math.assert_close(field.center_of_mass(density), (0.5, 1.5))
        density = Domain(x=4, y=3).scalar_grid(HardGeometryMask(Box[:, 2:3]))
        math.assert_close(field.center_of_mass(density), (2, 2.5))

    def test_staggered_curl_2d(self):
        pot = Domain(x=4, y=3).scalar_grid(HardGeometryMask(Box[1:2, 1:2]))
        curl = field.curl(pot, type=StaggeredGrid)
        math.assert_close(field.mean(curl), 0)
        math.assert_close(curl.values.vector[0].x[1], (1, -1))
        math.assert_close(curl.values.vector[1].y[1], (-1, 1, 0))

    def test_integrate_all(self):
        grid = CenteredGrid(field.Noise(vector=2), extrapolation.ZERO, x=10, y=10, bounds=Box[0:10, 0:10])
        math.assert_close(field.integrate(grid, grid.bounds), math.sum(grid.values, 'x,y'))
        grid = CenteredGrid(field.Noise(vector=2), extrapolation.ZERO, x=10, y=10, bounds=Box[0:1, 0:1])
        math.assert_close(field.integrate(grid, grid.bounds), math.sum(grid.values, 'x,y') / 100)
