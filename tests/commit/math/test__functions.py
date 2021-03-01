from unittest import TestCase

import numpy as np

import phi
from phi import math
from phi.math import extrapolation
from phi.math.backend import Backend


BACKENDS = phi.detect_backends()


def assert_not_close(*tensors, rel_tolerance, abs_tolerance):
    try:
        math.assert_close(*tensors, rel_tolerance, abs_tolerance)
        raise BaseException(AssertionError('Values are not close'))
    except AssertionError:
        pass


class TestMathFunctions(TestCase):

    def test_assert_close(self):
        math.assert_close(math.zeros(a=10), math.zeros(a=10), math.zeros(a=10), rel_tolerance=0, abs_tolerance=0)
        assert_not_close(math.zeros(a=10), math.ones(a=10), rel_tolerance=0, abs_tolerance=0)
        for scale in (1, 0.1, 10):
            math.assert_close(math.zeros(a=10), math.ones(a=10) * scale, rel_tolerance=0, abs_tolerance=scale * 1.001)
            math.assert_close(math.zeros(a=10), math.ones(a=10) * scale, rel_tolerance=1, abs_tolerance=0)
            assert_not_close(math.zeros(a=10), math.ones(a=10) * scale, rel_tolerance=0.9, abs_tolerance=0)
            assert_not_close(math.zeros(a=10), math.ones(a=10) * scale, rel_tolerance=0, abs_tolerance=0.9 * scale)
        with math.precision(64):
            assert_not_close(math.zeros(a=10), math.ones(a=10) * 1e-100, rel_tolerance=0, abs_tolerance=0)
            math.assert_close(math.zeros(a=10), math.ones(a=10) * 1e-100, rel_tolerance=0, abs_tolerance=1e-15)

    def test_concat(self):
        c = math.concat([math.zeros(b=3, a=2), math.ones(a=2, b=4)], 'b')
        self.assertEqual(2, c.shape.a)
        self.assertEqual(7, c.shape.b)
        math.assert_close(c.b[:3], 0)
        math.assert_close(c.b[3:], 1)

    def test_nonzero(self):
        c = math.concat([math.zeros(b=3, a=2), math.ones(a=2, b=4)], 'b')
        nz = math.nonzero(c)
        self.assertEqual(nz.shape.nonzero, 8)
        self.assertEqual(nz.shape.vector, 2)

    def test_maximum(self):
        v = math.ones(x=4, y=3, vector=2)
        math.assert_close(math.maximum(0, v), 1)
        math.assert_close(math.maximum(0, -v), 0)

    def test_grid_sample(self):
        for backend in BACKENDS:
            with backend:
                grid = math.sum(math.meshgrid(x=[1, 2, 3], y=[0, 3]), 'vector')  # 1 2 3 | 4 5 6
                coords = math.tensor([(0, 0), (0.5, 0), (0, 0.5), (-2, -1)], names=('list', 'vector'))
                interp = math.grid_sample(grid, coords, extrapolation.ZERO)
                math.assert_close(interp, [1, 1.5, 2.5, 0])

    def test_grid_sample_gradient_1d(self):
        grads_grid = []
        grads_coords = []
        for backend in BACKENDS:
            if backend.supports(Backend.gradients):
                print(backend)
                with backend:
                    grid = math.tensor([0., 1, 2, 3], 'x')
                    coords = math.tensor([0.5, 1.5], 'points')
                    with math.record_gradients(grid, coords):
                        sampled = math.grid_sample(grid, coords, extrapolation.ZERO)
                        loss = math.l2_loss(sampled)
                        grad_grid, grad_coords = math.gradients(loss, grid, coords)
                        grads_grid.append(grad_grid)
                        grads_coords.append(grad_coords)
        math.assert_close(*grads_grid, math.tensor([0.125, 0.5, 0.375, 0], 'x'))
        math.assert_close(*grads_coords, math.tensor([0.25, 0.75], 'points'))

    def test_grid_sample_gradient_2d(self):
        grads_grid = []
        grads_coords = []
        for backend in BACKENDS:
            if backend.supports(Backend.gradients):
                with backend:
                    grid = math.tensor([[1., 2, 3], [1, 2, 3]], 'x,y')
                    coords = math.tensor([(0.5, 0.5), (1, 1.1), (-0.8, -0.5)], 'points,vector')
                    with math.record_gradients(grid, coords):
                        sampled = math.grid_sample(grid, coords, extrapolation.ZERO)
                        loss = math.sum(sampled)
                        grad_grid, grad_coords = math.gradients(loss, grid, coords)
                        grads_grid.append(grad_grid)
                        grads_coords.append(grad_coords)
        math.assert_close(*grads_grid)
        math.assert_close(*grads_coords)

    def test_nonzero_batched(self):
        grid = math.tensor([[(0, 1)], [(0, 0)]], 'batch,x,y')
        nz = math.nonzero(grid, list_dim='nonzero', index_dim='vector')
        self.assertEqual(('batch', 'nonzero', 'vector'), nz.shape.names)
        self.assertEqual(1, nz.batch[0].shape.nonzero)
        self.assertEqual(0, nz.batch[1].shape.nonzero)

    def test_sum_collapsed(self):
        ones = math.ones(x=40000, y=30000)
        self.assertEqual(40000 * 30000, math.sum(ones))

    def test_prod_collapsed(self):
        ones = math.ones(x=40000, y=30000)
        math.assert_close(1, math.prod(ones))

    def test_mean_collapsed(self):
        ones = math.ones(x=40000, y=30000)
        data = math.spatial_stack([ones, ones * 2], 'vector')
        self.assertEqual(1.5, math.mean(data))

    def test_std_collapsed(self):
        ones = math.ones(x=40000, y=30000)
        std = math.std(ones)
        self.assertEqual(0, std)

    def test_grid_sample_1d(self):
        grid = math.tensor([0, 1, 2, 3], names='x')
        coords = math.tensor([[0], [1], [0.5]], names='x,vector')
        sampled = math.grid_sample(grid, coords, None)
        math.print(sampled)
        math.assert_close(sampled, [0, 1, 0.5])

    def test_grid_sample_backend_equality_2d(self):
        grid = math.random_normal(y=10, x=7)
        coords = math.random_uniform(mybatch=10, x=3, y=2) * (12, 9)
        grid_ = math.tensor(grid.native('x,y'), 'x,y')
        coords_ = coords.vector.flip()
        for extrap in (extrapolation.ZERO, extrapolation.ONE, extrapolation.BOUNDARY, extrapolation.PERIODIC):
            sampled = []
            for backend in BACKENDS:
                with backend:
                    grid, coords, grid_, coords_ = math.tensors(grid, coords, grid_, coords_)
                    sampled.append(math.grid_sample(grid, coords, extrap))
                    sampled.append(math.grid_sample(grid_, coords_, extrap))
            math.assert_close(*sampled, abs_tolerance=1e-6)

    def test_closest_grid_values_1d(self):
        grid = math.tensor([0, 1, 2, 3], names='x')
        coords = math.tensor([[0.1], [1.9], [0.5], [3.1]], names='x,vector')
        closest = math.closest_grid_values(grid, coords, extrapolation.ZERO)
        math.assert_close(closest, math.tensor([(0, 1), (1, 2), (0, 1), (3, 0)], names='x,closest_x'))

    def test_join_dimensions(self):
        grid = math.random_normal(batch=10, x=4, y=3, vector=2)
        points = math.join_dimensions(grid, grid.shape.spatial, 'points')
        self.assertEqual(('batch', 'points', 'vector'), points.shape.names)
        self.assertEqual(grid.shape.volume, points.shape.volume)
        self.assertEqual(grid.shape.non_spatial, points.shape.non_spatial)

    def test_split_dimension(self):
        grid = math.random_normal(batch=10, x=4, y=3, vector=2)
        points = math.join_dimensions(grid, grid.shape.spatial, 'points')
        split = points.points.split(grid.shape.spatial)
        self.assertEqual(grid.shape, split.shape)
        math.assert_close(grid, split)

    def test_fft(self):
        def get_2d_sine(grid_size, L):
            indices = np.array(np.meshgrid(*list(map(range, grid_size))))
            phys_coord = indices.T * L / (grid_size[0])  # between [0, L)
            x, y = phys_coord.T
            d = np.sin(2 * np.pi * x + 1) * np.sin(2 * np.pi * y + 1)
            return d

        sine_field = get_2d_sine((32, 32), L=2)
        fft_ref_tensor = math.tensor(np.fft.fft2(sine_field), 'x,y')
        with math.precision(64):
            for backend in BACKENDS:
                with backend:
                    sine_tensor = math.tensor(sine_field, 'x,y')
                    fft_tensor = math.fft(sine_tensor)
                    math.assert_close(fft_ref_tensor, fft_tensor, abs_tolerance=1e-12)  # Should usually be more precise. GitHub Actions has larger errors than usual.

    def test_trace_function(self):
        def f(x: math.Tensor, y: math.Tensor):
            return x + y

        for backend in BACKENDS:
            with backend:
                ft = math.trace_function(f)
                args1 = math.ones(x=2), math.ones(y=2)
                args2 = math.ones(x=3), math.ones(y=3)
                res1 = ft(*args1)
                self.assertEqual(math.shape(x=2, y=2), res1.shape)
                math.assert_close(res1, 2)
                res2 = ft(*args2)
                self.assertEqual(math.shape(x=3, y=3), res2.shape)
                math.assert_close(res2, 2)

    def test_gradient_function(self):
        def f(x: math.Tensor, y: math.Tensor):
            pred = x
            loss = math.l2_loss(pred - y)
            return loss, pred

        for backend in BACKENDS:
            if backend.supports(Backend.gradients):
                with backend:
                    x_data = math.tensor(2.)
                    y_data = math.tensor(1.)

                    dx, = math.gradient_function(f)(x_data, y_data)
                    math.assert_close(dx, 1)
                    dx, dy = math.gradient_function(f, [0, 1])(x_data, y_data)
                    math.assert_close(dx, 1)
                    math.assert_close(dy, -1)
                    loss, pred, dx, dy = math.gradient_function(f, [0, 1], get_output=True)(x_data, y_data)
                    math.assert_close(loss, 0.5)
                    math.assert_close(pred, x_data)
                    math.assert_close(dx, 1)
                    math.assert_close(dy, -1)

