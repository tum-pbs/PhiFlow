from unittest import TestCase

import numpy

import phi
from phi import math
from phi.field import StaggeredGrid, CenteredGrid
from phi.geom import Box
from phi import field
from phi.math.backend import Backend
from phi.physics import Domain


BACKENDS = phi.detect_backends()


class TestFieldMath(TestCase):

    def test_gradient(self):
        domain = Domain(x=4, y=3)
        phi = domain.grid() * (1, 2)
        grad = field.gradient(phi, stack_dim='gradient')
        self.assertEqual(('spatial', 'spatial', 'channel', 'channel'), grad.shape.types)

    def test_divergence_centered(self):
        v = field.CenteredGrid(math.ones(x=3, y=3), Box[0:1, 0:1], math.extrapolation.ZERO) * (1, 0)  # flow to the right
        div = field.divergence(v).values
        math.assert_close(div.y[0], (1.5, 0, -1.5))

    def test_trace_function(self):
        def f(x: StaggeredGrid, y: CenteredGrid):
            return x + (y >> x)

        ft = field.trace_function(f)
        domain = Domain(x=4, y=3)
        x = domain.staggered_grid(1)
        y = domain.vector_grid(1)

        res_f = f(x, y)
        res_ft = ft(x, y)
        self.assertEqual(res_f.shape, res_ft.shape)
        field.assert_close(res_f, res_ft)

    def test_gradient_function(self):
        def f(x: StaggeredGrid, y: CenteredGrid):
            pred = x + (y >> x)
            loss = field.l2_loss(pred)
            return loss

        domain = Domain(x=4, y=3)
        x = domain.staggered_grid(1)
        y = domain.vector_grid(1)

        for backend in BACKENDS:
            if backend.supports(Backend.gradients):
                with backend:
                    dx, = field.gradient_function(f)(x, y)
                    self.assertIsInstance(dx, StaggeredGrid)
                    loss, dx, dy = field.gradient_function(f, (0, 1), get_output=True)(x, y)
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
        self.assertEqual(math.shape(x=16, y=20, vector=2).alphabetically(), downsampled.shape.alphabetically())

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
