from itertools import product
from unittest import TestCase

import numpy

import phi
from phi import math, geom
from phi.field import StaggeredGrid, CenteredGrid, HardGeometryMask, PointCloud
from phi.geom import Box, Sphere
from phi import field
from phi.math import extrapolation, instance, channel, spatial, batch
from phi.math.backend import Backend

BACKENDS = phi.detect_backends()


class TestFieldMath(TestCase):

    def test_spatial_gradient(self):
        s = CenteredGrid(1, x=4, y=3) * (1, 2)
        grad = field.spatial_gradient(s, stack_dim=channel('spatial_gradient'))
        self.assertEqual(('spatial', 'spatial', 'channel', 'channel'), grad.shape.types)

    def test_spatial_gradient_batched(self):
        bounds = geom.stack([Box['x,y', 0:1, 0:1], Box['x,y', 0:10, 0:10]], batch('batch'))
        grid = CenteredGrid(0, extrapolation.ZERO, bounds, x=10, y=10)
        grad = field.spatial_gradient(grid)
        self.assertIsInstance(grad, CenteredGrid)

    def test_laplace_batched(self):
        bounds = geom.stack([Box['x,y', 0:1, 0:1], Box['x,y', 0:10, 0:10]], batch('batch'))
        grid = CenteredGrid(0, extrapolation.ZERO, bounds, x=10, y=10)
        lap = field.laplace(grid)
        self.assertIsInstance(lap, CenteredGrid)

    def test_divergence_centered(self):
        v = CenteredGrid(1, extrapolation.ZERO, bounds=Box['x,y', 0:1, 0:1], x=3, y=3) * (1, 0)  # flow to the right
        div = field.divergence(v).values
        math.assert_close(div.y[0], [1.5, 0, -1.5])

    def test_trace_function(self):
        def f(x: StaggeredGrid, y: CenteredGrid):
            return x + (y @ x)

        ft = field.jit_compile(f)
        x = StaggeredGrid(1, x=4, y=3)
        y = CenteredGrid((1, 1), x=4, y=3)

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

        x = StaggeredGrid(1, x=4, y=3)
        y = CenteredGrid((1, 1), x=4, y=3)

        for backend in BACKENDS:
            if backend.supports(Backend.jacobian):
                with backend:
                    dx, = grad(x, y)
                    self.assertIsInstance(dx, StaggeredGrid)
                    loss, (dx, dy) = fgrad(x, y)
                    self.assertIsInstance(loss, math.Tensor)
                    self.assertIsInstance(dx, StaggeredGrid)
                    self.assertIsInstance(dy, CenteredGrid)

    def test_upsample_downsample_centered_1d(self):
        grid = CenteredGrid(math.tensor([0, 1, 2, 3], spatial('x')))
        upsampled = field.upsample2x(grid)
        downsampled = field.downsample2x(upsampled)
        math.assert_close(downsampled.values.x[1:-1], grid.values.x[1:-1])

    def test_downsample_staggered_2d(self):
        grid = StaggeredGrid(1, extrapolation.BOUNDARY, x=32, y=40)
        downsampled = field.downsample2x(grid)
        self.assertEqual(set(spatial(x=16, y=20) & channel(vector=2)), set(downsampled.shape))

    def test_abs(self):
        grid = StaggeredGrid(-1, x=4, y=3)
        field.assert_close(field.abs(grid), abs(grid), 1)

    def test_sign(self):
        grid = StaggeredGrid(0.5, x=4, y=3)
        field.assert_close(field.sign(grid), 1)

    def test_round_(self):
        grid = StaggeredGrid(1.7, x=4, y=3)
        field.assert_close(field.round(grid), 2)

    def test_ceil(self):
        grid = StaggeredGrid(1.1, x=4, y=3)
        field.assert_close(field.ceil(grid), 2)

    def test_floor(self):
        grid = StaggeredGrid(2.8, x=4, y=3)
        field.assert_close(field.floor(grid), 2)

    def test_sqrt(self):
        grid = StaggeredGrid(2, x=4, y=3)
        field.assert_close(field.sqrt(grid), numpy.sqrt(2))

    def test_exp(self):
        grid = StaggeredGrid(0, x=4, y=3)
        field.assert_close(field.exp(grid), 1)

    def test_isfinite(self):
        grid = StaggeredGrid(1, x=4, y=3)
        field.assert_close(field.isfinite(grid), True)

    def test_real(self):
        grid = StaggeredGrid(1, x=4, y=3)
        field.assert_close(field.real(grid), grid)

    def test_imag(self):
        grid = StaggeredGrid(1, x=4, y=3)
        field.assert_close(field.imag(grid), 0)

    def test_sin(self):
        grid = StaggeredGrid(0, x=4, y=3)
        field.assert_close(field.sin(grid), 0)

    def test_cos(self):
        grid = StaggeredGrid(0, x=4, y=3)
        field.assert_close(field.cos(grid), 1)

    def test_convert_grid(self):
        grid = CenteredGrid(0, x=4, y=3)
        for backend in BACKENDS:
            converted = field.convert(grid, backend)
            self.assertEqual(converted.values.default_backend, backend)

    def test_convert_point_cloud(self):
        loc = math.random_uniform(instance(points=4), channel(vector='x,y'))
        val = math.random_normal(instance(points=4), channel(vector='x,y'))
        points = PointCloud(Sphere(loc, radius=1), val)
        for backend in BACKENDS:
            converted = field.convert(points, backend)
            self.assertEqual(converted.values.default_backend, backend)
            self.assertEqual(converted.elements.center.default_backend, backend)
            self.assertEqual(converted.elements.radius.default_backend, backend)

    def test_center_of_mass(self):
        density = CenteredGrid(Box(x=1, y=(1, 2)), x=4, y=3)
        math.assert_close(field.center_of_mass(density), (0.5, 1.5))
        density = CenteredGrid(Box(x=None, y=(2, 3)), x=4, y=3)
        math.assert_close(field.center_of_mass(density), (2, 2.5))

    def test_curl_2d_centered_to_staggered(self):
        pot = CenteredGrid(Box['x,y', 1:2, 1:2], x=4, y=3)
        curl = field.curl(pot, type=StaggeredGrid)
        math.assert_close(field.mean(curl), 0)
        math.assert_close(curl.values.vector[0].x[1], [1, -1])
        math.assert_close(curl.values.vector[1].y[1], [-1, 1, 0])

    def test_curl_2d_staggered_to_centered(self):
        velocity = StaggeredGrid((2, 0), extrapolation.BOUNDARY, x=2, y=2)
        curl = field.curl(velocity)
        self.assertEqual(spatial(x=3, y=3), curl.resolution)

    def test_integrate_all(self):
        grid = CenteredGrid(field.Noise(vector=2), extrapolation.ZERO, x=10, y=10, bounds=Box['x,y', 0:10, 0:10])
        math.assert_close(field.integrate(grid, grid.bounds), math.sum(grid.values, 'x,y'))
        grid = CenteredGrid(field.Noise(vector=2), extrapolation.ZERO, x=10, y=10, bounds=Box['x,y', 0:1, 0:1])
        math.assert_close(field.integrate(grid, grid.bounds), math.sum(grid.values, 'x,y') / 100)

    def test_tensor_as_field(self):
        t = math.random_normal(spatial(x=4, y=3), channel(vector='x,y'))
        grid = field.tensor_as_field(t)
        self.assertIsInstance(grid, CenteredGrid)
        math.assert_close(grid.dx, 1)
        math.assert_close(grid.points.x[0].y[0], 0)
        t = math.random_normal(instance(points=5), channel(vector='x,y'))
        points = field.tensor_as_field(t)
        self.assertIsInstance(points, PointCloud)


    def test__periodic_2d_arakawa_poisson_bracket(self):
        """test _periodic_2d_arakawa_poisson_bracket implementation"""
        with math.precision(64):
            # Define parameters to test
            test_params = {
                'grid_size': [(4, 4), (32, 32)],
                'dx': [0.1, 1],
                'gen_func': [lambda grid_size: numpy.random.rand(*grid_size).reshape(grid_size)]
            }
            # Generate test cases as the product
            test_cases = [dict(zip(test_params, v)) for v in product(*test_params.values())]
            for params in test_cases:
                grid_size = params['grid_size']
                d1 = params['gen_func'](grid_size)
                d2 = params['gen_func'](grid_size)
                dx = params['dx']
                padding = extrapolation.PERIODIC
                ref = self.arakawa_reference_implementation(numpy.pad(d1.copy(), 1, mode='wrap'), numpy.pad(d2.copy(), 1, mode='wrap'), dx)[1:-1, 1:-1]
                d1_tensor = field.CenteredGrid(values=math.tensor(d1, spatial('x,y')), bounds=geom.Box(x=grid_size[0], y=grid_size[1]), extrapolation=padding)
                d2_tensor = field.CenteredGrid(values=math.tensor(d2, spatial('x,y')), bounds=geom.Box(x=grid_size[0], y=grid_size[1]), extrapolation=padding)
                val = math._nd._periodic_2d_arakawa_poisson_bracket(d1_tensor.values, d2_tensor.values, dx)
                # try:
                math.assert_close(ref, val, rel_tolerance=1e-14, abs_tolerance=1e-14)
                # except BaseException as e:  # Enable the try/catch to get more info about the deviation
                #     abs_error = math.abs(control - ref)
                #     max_abs_error = math.max(abs_error)
                #     max_rel_error = math.max(math.abs(abs_error / ref))
                #     variation_str = "\n".join([
                #         f"max_absolute_error: {max_abs_error}",
                #         f"max_relative_error: {max_rel_error}",
                #     ])
                #     print(ref)
                #     print(control)
                #     raise AssertionError(e, params, variation_str)

    @staticmethod
    def arakawa_reference_implementation(zeta, psi, d):
        """pure Python exact implementation from paper"""

        def jpp(zeta, psi, d, i, j):
            return ((zeta[i + 1, j] - zeta[i - 1, j]) * (psi[i, j + 1] - psi[i, j - 1])
                    - (zeta[i, j + 1] - zeta[i, j - 1]) * (psi[i + 1, j] - psi[i - 1, j])) / (4 * d ** 2)

        def jpx(zeta, psi, d, i, j):
            return (zeta[i + 1, j] * (psi[i + 1, j + 1] - psi[i + 1, j - 1])
                    - zeta[i - 1, j] * (psi[i - 1, j + 1] - psi[i - 1, j - 1])
                    - zeta[i, j + 1] * (psi[i + 1, j + 1] - psi[i - 1, j + 1])
                    + zeta[i, j - 1] * (psi[i + 1, j - 1] - psi[i - 1, j - 1])) / (4 * d ** 2)

        def jxp(zeta, psi, d, i, j):
            return (zeta[i + 1, j + 1] * (psi[i, j + 1] - psi[i + 1, j])
                    - zeta[i - 1, j - 1] * (psi[i - 1, j] - psi[i, j - 1])
                    - zeta[i - 1, j + 1] * (psi[i, j + 1] - psi[i - 1, j])
                    + zeta[i + 1, j - 1] * (psi[i + 1, j] - psi[i, j - 1])) / (4 * d ** 2)

        val = numpy.zeros_like(zeta)
        for i in range(0, zeta.shape[0] - 1):
            for j in range(0, zeta.shape[1] - 1):
                val[i, j] += (jpp(zeta, psi, d, i, j) + jpx(zeta, psi, d, i, j) + jxp(zeta, psi, d, i, j))
        val = val / 3
        return val


