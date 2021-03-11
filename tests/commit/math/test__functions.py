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
                math.assert_close(interp, [1, 1.5, 2.5, 0], msg=backend.name)

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

    def test_grid_sample_backend_equality_2d_batched(self):
        grid = math.random_normal(mybatch=10, y=10, x=7)
        coords = math.random_uniform(mybatch=10, x=3, y=2) * (12, 9)
        grid_ = math.tensor(grid.native('mybatch,x,y'), 'mybatch,x,y')
        coords_ = coords.vector.flip()
        for extrap in (extrapolation.ZERO, extrapolation.ONE, extrapolation.BOUNDARY, extrapolation.PERIODIC):
            sampled = []
            for backend in BACKENDS:
                with backend:
                    grid, coords, grid_, coords_ = math.tensors(grid, coords, grid_, coords_)
                    sampled.append(math.grid_sample(grid, coords, extrap))
                    sampled.append(math.grid_sample(grid_, coords_, extrap))
            math.assert_close(*sampled, abs_tolerance=1e-5)

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
        fft_ref_tensor = math.wrap(np.fft.fft2(sine_field), 'x,y')
        with math.precision(64):
            for backend in BACKENDS:
                if backend.name != 'Jax':  # TODO Jax casts to float32 / complex64 on GitHub Actions
                    with backend:
                        sine_tensor = math.tensor(sine_field, 'x,y')
                        fft_tensor = math.fft(sine_tensor)
                        self.assertEqual(fft_tensor.dtype, math.DType(complex, 128), msg=backend.name)
                        math.assert_close(fft_ref_tensor, fft_tensor, abs_tolerance=1e-12, msg=backend.name)  # Should usually be more precise. GitHub Actions has larger errors than usual.

    def test_ifft(self):
        dimensions = 'xyz'
        for backend in BACKENDS:
            with backend:
                for d in range(1, len(dimensions) + 1):
                    x = math.random_normal(**{dim: 6 for dim in dimensions[:d]}) + math.tensor((0, 1), 'batch')
                    k = math.fft(x)
                    x_ = math.ifft(k)
                    math.assert_close(x, x_, abs_tolerance=1e-5, msg=backend.name)

    def test_trace_function(self):
        def f(x: math.Tensor, y: math.Tensor):
            return x + y

        for backend in BACKENDS:
            with backend:
                ft = math.jit_compile(f)
                args1 = math.ones(x=2), math.ones(batch=2)
                args2 = math.ones(x=3), math.ones(batch=3)
                res1 = ft(*args1)
                self.assertEqual(math.shape(batch=2, x=2), res1.shape, msg=backend.name)
                math.assert_close(res1, 2, msg=backend.name)
                res2 = ft(*args2)
                self.assertEqual(math.shape(batch=3, x=3), res2.shape, msg=backend.name)
                math.assert_close(res2, 2, msg=backend.name)

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
                    dx, = math.functional_gradient(f)(x_data, y_data)
                    math.assert_close(dx, 1, msg=backend.name)
                    dx, dy = math.functional_gradient(f, [0, 1])(x_data, y_data)
                    math.assert_close(dx, 1, msg=backend.name)
                    math.assert_close(dy, -1, msg=backend.name)
                    loss, pred, dx, dy = math.functional_gradient(f, [0, 1], get_output=True)(x_data, y_data)
                    math.assert_close(loss, 0.5, msg=backend.name)
                    math.assert_close(pred, x_data, msg=backend.name)
                    math.assert_close(dx, 1, msg=backend.name)
                    math.assert_close(dy, -1, msg=backend.name)

    def test_dot_vector(self):
        for backend in BACKENDS:
            with backend:
                a = math.ones(a=4)
                b = math.ones(b=4)
                dot = math.dot(a, 'a', b, 'b')
                self.assertEqual(0, dot.rank, msg=backend.name)
                math.assert_close(dot, 4, a.a * b.b, msg=backend.name)
                math.assert_close(math.dot(a, 'a', a, 'a'), 4, msg=backend.name)

    def test_dot_matrix(self):
        for backend in BACKENDS:
            with backend:
                a = math.ones(x=2, a=4, batch=10)
                b = math.ones(y=3, b=4)
                dot = math.dot(a, 'a', b, 'b')
                self.assertEqual(math.shape(x=2, batch=10, y=3).alphabetically(), dot.shape.alphabetically(), msg=backend.name)
                math.assert_close(dot, 4, msg=backend.name)

    def test_dot_batched_vector(self):
        for backend in BACKENDS:
            with backend:
                a = math.ones(batch=10, a=4)
                b = math.ones(batch=10, b=4)
                dot = math.dot(a, 'a', b, 'b')
                self.assertEqual(math.shape(batch=10), dot.shape, msg=backend.name)
                math.assert_close(dot, 4, a.a * b.b, msg=backend.name)
                dot = math.dot(a, 'a', a, 'a')
                self.assertEqual(math.shape(batch=10), dot.shape, msg=backend.name)
                math.assert_close(dot, 4, a.a * a.a, msg=backend.name)
                # more dimensions
                a = math.ones(batch=10, a=4, x=2)
                b = math.ones(batch=10, y=3, b=4)
                dot = math.dot(a, 'a', b, 'b')
                self.assertEqual(math.shape(x=2, batch=10, y=3).alphabetically(), dot.shape.alphabetically(), msg=backend.name)
                math.assert_close(dot, 4, msg=backend.name)

    def test_range(self):
        for backend in BACKENDS:
            with backend:
                math.assert_close(math.range(1, 5), [1, 2, 3, 4], msg=backend.name)
                math.assert_close(math.range(1), [0], msg=backend.name)

    def test_boolean_mask_1d(self):
        for backend in BACKENDS:
            with backend:
                x = math.range(4)
                mask = math.tensor([True, False, True, False], 'range')
                math.assert_close(math.boolean_mask(x, 'range', mask), [0, 2], msg=backend.name)
                math.assert_close(x.range[mask], [0, 2], msg=backend.name)

    def test_boolean_mask_batched(self):
        for backend in BACKENDS:
            with backend:
                x = math.expand(math.range(4, dim='x'), batch=2) * math.tensor([1, -1])
                mask = math.tensor([[True, False, True, False], [False, True, False, False]], 'batch,x')
                selected = math.boolean_mask(x, 'x', mask)
                expected_0 = math.tensor([(0, -0), (2, -2)], 'x,vector')
                expected_1 = math.tensor([(1, -1)], 'x,vector')
                math.assert_close(selected.batch[0], expected_0, msg=backend.name)
                math.assert_close(selected.batch[1], expected_1, msg=backend.name)
                math.assert_close(selected, x.x[mask], msg=backend.name)

    def test_boolean_mask_semi_batched(self):
        for backend in BACKENDS:
            with backend:
                x = math.range(4, dim='x')
                mask = math.tensor([[True, False, True, False], [False, True, False, False]], 'batch,x')
                selected = math.boolean_mask(x, 'x', mask)
                self.assertEqual(3, selected.shape.volume, msg=backend.name)

    def test_boolean_mask_dim_missing(self):
        for backend in BACKENDS:
            with backend:
                x = math.random_uniform(x=2)
                mask = math.tensor([True, False, True, True], 'selection')
                selected = math.boolean_mask(x, 'selection', mask)
                self.assertEqual(math.shape(x=2, selection=3).alphabetically(), selected.shape.alphabetically(), msg=backend.name)

    def test_minimize(self):
        def loss(x):
            return math.l1_loss(x - 1)

        x0 = math.zeros(x=4)
        for backend in BACKENDS:
            with backend:
                converged, x, iterations = math.minimize(loss, x0, math.Solve(None, 0, 1e-3))
                math.assert_close(x, 1, abs_tolerance=1e-3, msg=backend.name)

    def test_custom_gradient_scalar(self):
        def f(x):
            return x

        def grad(_x, _y, df):
            return df * 0,

        for backend in BACKENDS:
            if backend.supports(Backend.gradients):
                with backend:
                    normal_gradient, = math.functional_gradient(f)(math.ones())
                    math.assert_close(normal_gradient, 1)
                    f_custom_grad = math.custom_gradient(f, grad)
                    custom_gradient, = math.functional_gradient(f_custom_grad)(math.ones())
                    math.assert_close(custom_gradient, 0)

    def test_custom_gradient_vector(self):
        def f(x):
            return x.x[:2]

        def grad(_x, _y, df):
            return math.flatten(math.expand(df * 0, tmp=2)),

        def loss(x):
            fg = math.custom_gradient(f, grad)
            y = fg(x)
            return math.l1_loss(y)

        for backend in BACKENDS:
            if backend.supports(Backend.custom_gradient) and backend.name != 'Jax':
                with backend:
                    custom_loss_grad, = math.functional_gradient(loss)(math.ones(x=4))
                    math.assert_close(custom_loss_grad, 0, msg=backend.name)

    def test_scatter_1d(self):
        for backend in BACKENDS:
            if backend.name in ('NumPy', 'Jax'):
                with backend:
                    base = math.ones(x=4)
                    indices = math.wrap([1, 2], 'points')
                    values = math.wrap([11, 12], 'points')
                    updated = math.scatter(base, indices, values, 'points', mode='update', outside_handling='undefined')
                    math.assert_close(updated, [1, 11, 12, 1])
                    updated = math.scatter(base, indices, values, 'points', mode='add', outside_handling='undefined')
                    math.assert_close(updated, [1, 12, 13, 1])
                    # with vector dim
                    indices = math.expand_channel(indices, vector=1)
                    updated = math.scatter(base, indices, values, 'points', mode='update', outside_handling='undefined')
                    math.assert_close(updated, [1, 11, 12, 1])

    def test_scatter_update_1d_batched(self):
        for backend in BACKENDS:
            if backend.name in ('NumPy', 'Jax'):
                with backend:
                    # Only base batched
                    base = math.zeros(x=4) + math.tensor([0, 1])
                    indices = math.wrap([1, 2], 'points')
                    values = math.wrap([11, 12], 'points')
                    updated = math.scatter(base, indices, values, 'points', mode='update', outside_handling='undefined')
                    math.assert_close(updated, math.tensor([(0, 1), (11, 11), (12, 12), (0, 1)], 'x,vector'))
                    # Only values batched
                    base = math.ones(x=4)
                    indices = math.wrap([1, 2], 'points')
                    values = math.wrap([[11, 12], [-11, -12]], 'batch,points')
                    updated = math.scatter(base, indices, values, 'points', mode='update', outside_handling='undefined')
                    math.assert_close(updated, math.tensor([[1, 11, 12, 1], [1, -11, -12, 1]], 'batch,x'))
                    # Only indices batched
                    base = math.ones(x=4)
                    indices = math.wrap([[0, 1], [1, 2]], 'batch,points')
                    values = math.wrap([11, 12], 'points')
                    updated = math.scatter(base, indices, values, 'points', mode='update', outside_handling='undefined')
                    math.assert_close(updated, math.tensor([[11, 12, 1, 1], [1, 11, 12, 1]], 'batch,x'))
                    # Everything batched
                    base = math.zeros(x=4) + math.tensor([0, 1], 'batch')
                    indices = math.wrap([[0, 1], [1, 2]], 'batch,points')
                    values = math.wrap([[11, 12], [-11, -12]], 'batch,points')
                    updated = math.scatter(base, indices, values, 'points', mode='update', outside_handling='undefined')
                    math.assert_close(updated, math.tensor([[11, 12, 0, 0], [1, -11, -12, 1]], 'batch,x'))

    def test_scatter_update_2d(self):
        base = math.ones(x=3, y=2)
        indices = math.wrap([(0, 0), (0, 1), (2, 1)], 'points,vector')
        values = math.wrap([11, 12, 13], 'points')
        updated = math.scatter(base, indices, values, 'points', mode='update', outside_handling='undefined')
        math.assert_close(updated, math.tensor([[11, 1, 1], [12, 1, 13]], 'y,x'))

    def test_scatter_add_2d(self):
        base = math.ones(x=3, y=2)
        indices = math.wrap([(0, 0), (0, 0), (0, 1), (2, 1)], 'points,vector')
        values = math.wrap([11, 11, 12, 13], 'points')
        updated = math.scatter(base, indices, values, 'points', mode='add', outside_handling='undefined')
        math.assert_close(updated, math.tensor([[23, 1, 1], [13, 1, 14]], 'y,x'))

    def test_scatter_2d_clamp(self):
        base = math.ones(x=3, y=2)
        indices = math.wrap([(-1, 0), (0, 2), (4, 3)], 'points,vector')
        values = math.wrap([11, 12, 13], 'points')
        updated = math.scatter(base, indices, values, 'points', mode='update', outside_handling='clamp')
        math.assert_close(updated, math.tensor([[11, 1, 1], [12, 1, 13]], 'y,x'))

    def test_scatter_2d_discard(self):
        base = math.ones(x=3, y=2)
        indices = math.wrap([(-1, 0), (0, 1), (3, 1)], 'points,vector')
        values = math.wrap([11, 12, 13], 'points')
        updated = math.scatter(base, indices, values, 'points', mode='update', outside_handling='discard')
        math.assert_close(updated, math.tensor([[1, 1, 1], [12, 1, 1]], 'y,x'))

    def test_sin(self):
        for backend in BACKENDS:
            with backend:
                math.assert_close(math.sin(math.zeros(x=4)), 0, abs_tolerance=1e-6, msg=backend.name)
                math.assert_close(math.sin(math.tensor(math.PI / 2)), 1, abs_tolerance=1e-6, msg=backend.name)
                math.assert_close(math.sin(math.tensor(math.PI)), 0, abs_tolerance=1e-6, msg=backend.name)
                math.assert_close(math.sin(math.tensor(math.PI * 3 / 2)), -1, abs_tolerance=1e-6, msg=backend.name)

    def test_cos(self):
        for backend in BACKENDS:
            with backend:
                math.assert_close(math.cos(math.zeros(x=4)), 1, abs_tolerance=1e-6, msg=backend.name)
                math.assert_close(math.cos(math.tensor(math.PI / 2)), 0, abs_tolerance=1e-6, msg=backend.name)
                math.assert_close(math.cos(math.tensor(math.PI)), -1, abs_tolerance=1e-6, msg=backend.name)
                math.assert_close(math.cos(math.tensor(math.PI * 3 / 2)), 0, abs_tolerance=1e-6, msg=backend.name)

    def test_any(self):
        for backend in BACKENDS:
            with backend:
                math.assert_close(math.any(math.tensor([[False, True], [False, False]], 'y,x'), dim='x'), [True, False])
                math.assert_close(math.any(math.tensor([[False, True], [False, False]], 'y,x'), dim='x,y'), True)
                math.assert_close(math.any(math.tensor([[False, True], [False, False]], 'y,x')), True)

    def test_all(self):
        for backend in BACKENDS:
            with backend:
                math.assert_close(math.all(math.tensor([[False, True], [True, True]], 'y,x'), dim='x'), [False, True])
                math.assert_close(math.all(math.tensor([[False, True], [True, True]], 'y,x'), dim='x,y'), False)
                math.assert_close(math.all(math.tensor([[False, True], [True, True]], 'y,x')), False)

    def test_imag(self):
        for backend in BACKENDS:
            with backend:
                math.assert_close(math.imag(math.ones(x=4)), 0, msg=backend.name)
                math.assert_close(math.imag(math.ones(x=4) * 1j), 1, msg=backend.name)

    def test_real(self):
        for backend in BACKENDS:
            with backend:
                math.assert_close(math.real(math.ones(x=4)), 1, msg=backend.name)
                math.assert_close(math.real(math.ones(x=4) * 1j), 0, msg=backend.name)

    def test_convolution_1d_scalar(self):
        for backend in BACKENDS:
            with backend:
                x = math.tensor([1, 2, 3, 4], 'x')
                identity_kernel1 = math.ones(x=1)
                identity_kernel2 = math.tensor([0, 1], 'x')
                identity_kernel3 = math.tensor([0, 1, 0], 'x')
                shift_kernel3 = math.tensor([0, 0, 1], 'x')
                # no padding
                math.assert_close(x, math.convolve(x, identity_kernel1), msg=backend.name)
                math.assert_close(x.x[1:-1], math.convolve(x, identity_kernel3), msg=backend.name)
                math.assert_close(x.x[1:], math.convolve(x, identity_kernel2), msg=backend.name)
                math.assert_close(x.x[2:], math.convolve(x, shift_kernel3), msg=backend.name)
                # zero-padding
                math.assert_close(x, math.convolve(x, identity_kernel1, math.extrapolation.ZERO), msg=backend.name)
                math.assert_close(x, math.convolve(x, identity_kernel3, math.extrapolation.ZERO), msg=backend.name)
                math.assert_close(x, math.convolve(x, identity_kernel2, math.extrapolation.ZERO), msg=backend.name)
                math.assert_close([2, 3, 4, 0], math.convolve(x, shift_kernel3, math.extrapolation.ZERO), msg=backend.name)
                # periodic padding
                math.assert_close(x, math.convolve(x, identity_kernel1, math.extrapolation.PERIODIC), msg=backend.name)
                math.assert_close(x, math.convolve(x, identity_kernel3, math.extrapolation.PERIODIC), msg=backend.name)
                math.assert_close(x, math.convolve(x, identity_kernel2, math.extrapolation.PERIODIC), msg=backend.name)
                math.assert_close([2, 3, 4, 1], math.convolve(x, shift_kernel3, math.extrapolation.PERIODIC), msg=backend.name)

    def test_convolution_1d_batched(self):
        for backend in BACKENDS:
            with backend:
                # only values batched
                x = math.tensor([[1, 2, 3], [11, 12, 13]], 'batch,x') * (2, -1)
                identity_kernel1 = math.ones(x=1)
                identity_kernel2 = math.tensor([0, 1], 'x')
                identity_kernel3 = math.tensor([0, 1, 0], 'x')
                shift_kernel3 = math.tensor([0, 0, 1], 'x')
                # no padding
                math.assert_close(math.convolve(x, identity_kernel1), math.tensor([[1, 2, 3], [11, 12, 13]], 'batch,x'), msg=backend.name)
                math.assert_close(math.convolve(x, identity_kernel2), math.tensor([[2, 3], [12, 13]], 'batch,x'), msg=backend.name)
                math.assert_close(math.convolve(x, identity_kernel3), math.tensor([[2], [12]], 'batch,x'), msg=backend.name)
                math.assert_close(math.convolve(x, shift_kernel3), math.tensor([[3], [13]], 'batch,x'), msg=backend.name)
                # # zero-padding
                math.assert_close(math.convolve(x, identity_kernel1, math.extrapolation.ZERO), math.tensor([[1, 2, 3], [11, 12, 13]], 'batch,x'), msg=backend.name)
                math.assert_close(math.convolve(x, identity_kernel2, math.extrapolation.ZERO), math.tensor([[1, 2, 3], [11, 12, 13]], 'batch,x'), msg=backend.name)
                math.assert_close(math.convolve(x, identity_kernel3, math.extrapolation.ZERO), math.tensor([[1, 2, 3], [11, 12, 13]], 'batch,x'), msg=backend.name)
                math.assert_close(math.convolve(x, shift_kernel3, math.extrapolation.ZERO), math.tensor([[2, 3, 0], [12, 13, 0]], 'batch,x'), msg=backend.name)
                # # periodic padding
                math.assert_close(math.convolve(x, identity_kernel1, math.extrapolation.PERIODIC), math.tensor([[1, 2, 3], [11, 12, 13]], 'batch,x'), msg=backend.name)
                math.assert_close(math.convolve(x, identity_kernel2, math.extrapolation.PERIODIC), math.tensor([[1, 2, 3], [11, 12, 13]], 'batch,x'), msg=backend.name)
                math.assert_close(math.convolve(x, identity_kernel3, math.extrapolation.PERIODIC), math.tensor([[1, 2, 3], [11, 12, 13]], 'batch,x'), msg=backend.name)
                math.assert_close(math.convolve(x, shift_kernel3, math.extrapolation.PERIODIC), math.tensor([[2, 3, 1], [12, 13, 11]], 'batch,x'), msg=backend.name)
                # values and filters batched
                mixed_kernel = math.tensor([[0, 1, 0], [0, 0, 1]], 'batch,x')
                math.assert_close(math.convolve(x, mixed_kernel, math.extrapolation.ZERO), math.tensor([[1, 2, 3], [12, 13, 0]], 'batch,x'), msg=backend.name)
                math.assert_close(math.convolve(x, mixed_kernel, math.extrapolation.PERIODIC), math.tensor([[1, 2, 3], [12, 13, 11]], 'batch,x'), msg=backend.name)
                # with output channels
                out_matrix = math.tensor([[1, 0], [0, 1], [1, 1]], 'out,vector').out.as_channel()
                kernel = identity_kernel3 * out_matrix
                expected = math.tensor([
                    [[2, 4, 6], [22, 24, 26]],
                    [[-1, -2, -3], [-11, -12, -13]],
                    [[1, 2, 3], [11, 12, 13]]], 'out,batch,x')
                math.assert_close(math.convolve(x, kernel, math.extrapolation.ZERO), expected, msg=backend.name)

    # def test_convolution_2d(self):  # TODO
    #     pass

    def test_reshaped_native(self):
        a = math.random_uniform(vector=2, x=4, y=3)
        nat = math.reshaped_native(a, ['batch', a.shape.spatial, 'vector'], force_expand=False)
        self.assertEqual((1, 12, 2), nat.shape)
        nat = math.reshaped_native(a, [math.shape(batch=10), a.shape.spatial, math.shape(vector=2)], force_expand=False)
        self.assertEqual((1, 12, 2), nat.shape)
        nat = math.reshaped_native(a, [math.shape(batch=10), a.shape.spatial, math.shape(vector=2)], force_expand=['x'])
        self.assertEqual((1, 12, 2), nat.shape)
        nat = math.reshaped_native(a, [math.shape(batch=10), a.shape.spatial, math.shape(vector=2)], force_expand=True)
        self.assertEqual((10, 12, 2), nat.shape)
        nat = math.reshaped_native(a, [math.shape(batch=10), a.shape.spatial, math.shape(vector=2)], force_expand=['batch'])
        self.assertEqual((10, 12, 2), nat.shape)
        nat = math.reshaped_native(a, [a.shape.spatial, math.shape(vector=2, v2=2)], force_expand=False)
        self.assertEqual((12, 4), nat.shape)
        try:
            math.reshaped_native(a, [math.shape(vector=2, v2=2)], force_expand=False)
        except AssertionError as err:
            print(err)
            pass

    def test_native(self):
        nat = np.zeros(4)
        self.assertIs(math.native(nat), nat)
        self.assertIs(math.native(math.tensor(nat)), nat)

    def test_numpy(self):
        nat = np.zeros(4)
        self.assertIs(math.numpy(nat), nat)
        self.assertIs(math.numpy(math.tensor(nat)), nat)
