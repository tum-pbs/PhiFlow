from unittest import TestCase

import numpy as np

import phi
from phi import math
from phi.math import extrapolation, spatial, channel, instance, batch
from phi.math.backend import Backend


BACKENDS = phi.detect_backends()


def assert_not_close(*tensors, rel_tolerance, abs_tolerance):
    try:
        math.assert_close(*tensors, rel_tolerance, abs_tolerance)
        raise Exception(AssertionError('Values are not close'))
    except AssertionError:
        pass


class TestMathFunctions(TestCase):

    def test_assert_close(self):
        a = spatial(a=10)
        math.assert_close(math.zeros(a), math.zeros(a), math.zeros(a), rel_tolerance=0, abs_tolerance=0)
        assert_not_close(math.zeros(a), math.ones(a), rel_tolerance=0, abs_tolerance=0)
        for scale in (1, 0.1, 10):
            math.assert_close(math.zeros(a), math.ones(a) * scale, rel_tolerance=0, abs_tolerance=scale * 1.001)
            math.assert_close(math.zeros(a), math.ones(a) * scale, rel_tolerance=1, abs_tolerance=0)
            assert_not_close(math.zeros(a), math.ones(a) * scale, rel_tolerance=0.9, abs_tolerance=0)
            assert_not_close(math.zeros(a), math.ones(a) * scale, rel_tolerance=0, abs_tolerance=0.9 * scale)
        with math.precision(64):
            assert_not_close(math.zeros(a), math.ones(a) * 1e-100, rel_tolerance=0, abs_tolerance=0)
            math.assert_close(math.zeros(a), math.ones(a) * 1e-100, rel_tolerance=0, abs_tolerance=1e-15)

    def test_concat(self):
        c = math.concat([math.zeros(spatial(b=3, a=2)), math.ones(spatial(a=2, b=4))], spatial('b'))
        self.assertEqual(2, c.shape.get_size('a'))
        self.assertEqual(7, c.shape.get_size('b'))
        math.assert_close(c.b[:3], 0)
        math.assert_close(c.b[3:], 1)

    def test_nonzero(self):
        c = math.concat([math.zeros(spatial(b=3, a=2)), math.ones(spatial(a=2, b=4))], spatial('b'))
        nz = math.nonzero(c)
        self.assertEqual(nz.shape.get_size('nonzero'), 8)
        self.assertEqual(nz.shape.get_size('vector'), 2)

    def test_nonzero_batched(self):
        grid = math.tensor([[(0, 1)], [(0, 0)]], batch('batch'), spatial('x,y'))
        nz = math.nonzero(grid)
        self.assertEqual(('batch', 'nonzero', 'vector'), nz.shape.names)
        self.assertEqual(1, nz.batch[0].shape.get_size('nonzero'))
        self.assertEqual(0, nz.batch[1].shape.get_size('nonzero'))

    def test_maximum(self):
        v = math.ones(spatial(x=4, y=3) & channel(vector=2))
        math.assert_close(math.maximum(0, v), 1)
        math.assert_close(math.maximum(0, -v), 0)

    def test_sum_collapsed(self):
        ones = math.ones(spatial(x=40000, y=30000))
        math.assert_close(40000 * 30000, math.sum(ones))

    def test_prod_collapsed(self):
        ones = math.ones(spatial(x=40000, y=30000))
        math.assert_close(1, math.prod(ones))

    def test_mean_collapsed(self):
        ones = math.ones(spatial(x=40000, y=30000))
        data = math.stack([ones, ones * 2], spatial('vector'))
        math.assert_close(1.5, math.mean(data))

    def test_std_collapsed(self):
        ones = math.ones(spatial(x=40000, y=30000))
        std = math.std(ones)
        math.assert_close(0, std)

    def test_sum_by_type(self):
        a = math.ones(spatial(x=3, y=4), batch(b=10), instance(i=2), channel(vector=2))
        math.assert_close(math.sum(a, spatial), 12)

    def test_grid_sample(self):
        for backend in BACKENDS:
            with backend:
                grid = math.sum(math.meshgrid(x=[1, 2, 3], y=[0, 3]), 'vector')  # 1 2 3 | 4 5 6
                coords = math.tensor([(0, 0), (0.5, 0), (0, 0.5), (-2, -1)], instance('list'), channel('vector'))
                interp = math.grid_sample(grid, coords, extrapolation.ZERO)
                math.assert_close(interp, [1, 1.5, 2.5, 0], msg=backend.name)

    def test_grid_sample_1d(self):
        grid = math.tensor([0, 1, 2, 3], spatial('x'))
        coords = math.tensor([[0], [1], [0.5]], spatial('x'), channel('vector'))
        sampled = math.grid_sample(grid, coords, None)
        math.print(sampled)
        math.assert_close(sampled, [0, 1, 0.5])

    def test_grid_sample_backend_equality_2d(self):
        grid = math.random_normal(spatial(y=10, x=7))
        coords = math.random_uniform(batch(mybatch=10) & spatial(x=3, y=2)) * (12, 9)
        grid_ = math.tensor(grid.native('x,y'), spatial('x,y'))
        coords_ = coords.vector.flip()
        for extrap in (extrapolation.ZERO, extrapolation.ONE, extrapolation.BOUNDARY, extrapolation.PERIODIC):
            sampled = []
            for backend in BACKENDS:
                with backend:
                    grid = math.tensor(grid)
                    coords = math.tensor(coords)
                    grid_ = math.tensor(grid_)
                    coords_ = math.tensor(coords_)
                    sampled.append(math.grid_sample(grid, coords, extrap))
                    sampled.append(math.grid_sample(grid_, coords_, extrap))
            math.assert_close(*sampled, abs_tolerance=1e-6)

    def test_grid_sample_backend_equality_2d_batched(self):
        grid = math.random_normal(batch(mybatch=10) & spatial(y=10, x=7))
        coords = math.random_uniform(batch(mybatch=10) & spatial(x=3, y=2)) * (12, 9)
        grid_ = math.tensor(grid.native('mybatch,x,y'), batch('mybatch'), spatial('x,y'))
        coords_ = coords.vector.flip()
        for extrap in (extrapolation.ZERO, extrapolation.ONE, extrapolation.BOUNDARY, extrapolation.PERIODIC):
            sampled = []
            for backend in BACKENDS:
                with backend:
                    grid = math.tensor(grid)
                    coords = math.tensor(coords)
                    grid_ = math.tensor(grid_)
                    coords_ = math.tensor(coords_)
                    sampled.append(math.grid_sample(grid, coords, extrap))
                    sampled.append(math.grid_sample(grid_, coords_, extrap))
            math.assert_close(*sampled, abs_tolerance=1e-5)

    def test_grid_sample_gradient_1d(self):
        for backend in BACKENDS:
            if backend.supports(Backend.gradients):
                with backend:
                    grid = math.tensor([0., 1, 2, 3], spatial('x'))
                    coords = math.tensor([0.5, 1.5], instance('points'))
                    with math.record_gradients(grid, coords):
                        sampled = math.grid_sample(grid, coords, extrapolation.ZERO)
                        loss = math.mean(math.l2_loss(sampled)) / 2
                        grad_grid, grad_coords = math.gradients(loss, grid, coords)
                    math.assert_close(grad_grid, math.tensor([0.125, 0.5, 0.375, 0], spatial('x')), msg=backend)
                    math.assert_close(grad_coords, math.tensor([0.25, 0.75], instance('points')), msg=backend)

    def test_grid_sample_gradient_2d(self):
        grads_grid = []
        grads_coords = []
        for backend in BACKENDS:
            if backend.supports(Backend.gradients):
                with backend:
                    grid = math.tensor([[1., 2, 3], [1, 2, 3]], spatial('x,y'))
                    coords = math.tensor([(0.5, 0.5), (1, 1.1), (-0.8, -0.5)], instance('points'), channel('vector'))
                    with math.record_gradients(grid, coords):
                        sampled = math.grid_sample(grid, coords, extrapolation.ZERO)
                        loss = math.sum(sampled) / 3
                        grad_grid, grad_coords = math.gradients(loss, grid, coords)
                        grads_grid.append(grad_grid)
                        grads_coords.append(grad_coords)
        math.assert_close(*grads_grid)
        math.assert_close(*grads_coords)

    def test_closest_grid_values_1d(self):
        grid = math.tensor([0, 1, 2, 3], spatial('x'))
        coords = math.tensor([[0.1], [1.9], [0.5], [3.1]], spatial('x'), channel('vector'))
        closest = math.closest_grid_values(grid, coords, extrapolation.ZERO)
        math.assert_close(closest, math.tensor([(0, 1), (1, 2), (0, 1), (3, 0)], spatial('x'), channel('closest_x')))

    def test_join_dimensions(self):
        grid = math.random_normal(batch(batch=10) & spatial(x=4, y=3) & channel(vector=2))
        points = math.pack_dims(grid, grid.shape.spatial, instance('points'))
        self.assertEqual(('batch', 'points', 'vector'), points.shape.names)
        self.assertEqual(grid.shape.volume, points.shape.volume)
        self.assertEqual(grid.shape.non_spatial, points.shape.non_instance)

    def test_split_dimension(self):
        grid = math.random_normal(batch(batch=10) & spatial(x=4, y=3) & channel(vector=2))
        points = math.pack_dims(grid, grid.shape.spatial, instance('points'))
        split = points.points.split(grid.shape.spatial)
        self.assertEqual(grid.shape, split.shape)
        math.assert_close(grid, split)

    def test_cumulative_sum(self):
        t = math.tensor([(0, 1, 2, 3), (-1, -2, -3, -4)], spatial('y,x'))
        for backend in BACKENDS:
            with backend:
                t_ = math.tensor(t)
                x_ = math.cumulative_sum(t_, 'x')
                math.assert_close(x_, [(0, 1, 3, 6), (-1, -3, -6, -10)], msg=backend.name)
                y_ = math.cumulative_sum(t_, t.shape[0])
                math.assert_close(y_, [(0, 1, 2, 3), (-1, -1, -1, -1)], msg=backend.name)

    def test_quantile(self):
        for backend in BACKENDS:
            if backend.name != "TensorFlow":  # TensorFlow probability import problem
                with backend:
                    t = math.tensor([(1, 2, 3, 4), (1, 2, 3, 4), (6, 7, 8, 9)], batch('batch'), instance('list'))
                    q = math.quantile(t, 0.5)
                    math.assert_close(q, (2.5, 2.5, 7.5), msg=backend.name)
                    q = math.quantile(t, [0.5, 0.6])
                    math.assert_close(q, [(2.5, 2.5, 7.5), (2.8, 2.8, 7.8)], msg=backend.name)

    def test_median(self):
        t = math.tensor([(1, 2, 3, 10), (0, 1, 3, 10)], batch('batch'), instance('list'))
        math.assert_close(math.median(t), [2.5, 2])

    def test_fft(self):
        def get_2d_sine(grid_size, L):
            indices = np.array(np.meshgrid(*list(map(range, grid_size))))
            phys_coord = indices.T * L / (grid_size[0])  # between [0, L)
            x, y = phys_coord.T
            d = np.sin(2 * np.pi * x + 1) * np.sin(2 * np.pi * y + 1)
            return d

        sine_field = get_2d_sine((32, 32), L=2)
        fft_ref_tensor = math.wrap(np.fft.fft2(sine_field), spatial('x,y'))
        with math.precision(64):
            for backend in BACKENDS:
                if backend.name != 'Jax':  # TODO Jax casts to float32 / complex64 on GitHub Actions
                    with backend:
                        sine_tensor = math.tensor(sine_field, spatial('x,y'))
                        fft_tensor = math.fft(sine_tensor)
                        self.assertEqual(fft_tensor.dtype, math.DType(complex, 128), msg=backend.name)
                        math.assert_close(fft_ref_tensor, fft_tensor, abs_tolerance=1e-12, msg=backend.name)  # Should usually be more precise. GitHub Actions has larger errors than usual.

    def test_ifft(self):
        dimensions = 'xyz'
        for backend in BACKENDS:
            with backend:
                for d in range(1, len(dimensions) + 1):
                    x = math.random_normal(spatial(**{dim: 6 for dim in dimensions[:d]})) + math.tensor((0, 1), batch('batch'))
                    k = math.fft(x)
                    x_ = math.ifft(k)
                    math.assert_close(x, x_, abs_tolerance=1e-5, msg=backend.name)

    def test_fft_dims(self):
        for backend in BACKENDS:
            with backend:
                x = math.random_normal(batch(x=8, y=6, z=4))
                k3 = math.fft(x, 'x,y,z')
                k = x
                for dim in 'xyz':
                    k = math.fft(k, dim)
                math.assert_close(k, k3, abs_tolerance=1e-5, msg=backend.name)

    def test_dot_vector(self):
        for backend in BACKENDS:
            with backend:
                a = math.ones(spatial(a=4))
                b = math.ones(spatial(b=4))
                dot = math.dot(a, 'a', b, 'b')
                self.assertEqual(0, dot.rank, msg=backend.name)
                math.assert_close(dot, 4, a.a * b.b, msg=backend.name)
                math.assert_close(math.dot(a, 'a', a, 'a'), 4, msg=backend.name)

    def test_dot_matrix(self):
        for backend in BACKENDS:
            with backend:
                a = math.ones(spatial(x=2, a=4) & batch(batch=10))
                b = math.ones(spatial(y=3, b=4))
                dot = math.dot(a, 'a', b, 'b')
                self.assertEqual(set(spatial(x=2, y=3) & batch(batch=10)), set(dot.shape), msg=backend.name)
                math.assert_close(dot, 4, msg=backend.name)

    def test_dot_batched_vector(self):
        for backend in BACKENDS:
            with backend:
                a = math.ones(batch(batch=10) & spatial(a=4))
                b = math.ones(batch(batch=10) & spatial(b=4))
                dot = math.dot(a, 'a', b, 'b')
                self.assertEqual(batch(batch=10), dot.shape, msg=backend.name)
                math.assert_close(dot, 4, a.a * b.b, msg=backend.name)
                dot = math.dot(a, 'a', a, 'a')
                self.assertEqual(batch(batch=10), dot.shape, msg=backend.name)
                math.assert_close(dot, 4, a.a * a.a, msg=backend.name)
                # more dimensions
                a = math.ones(batch(batch=10) & spatial(a=4, x=2))
                b = math.ones(batch(batch=10) & spatial(y=3, b=4))
                dot = math.dot(a, 'a', b, 'b')
                self.assertEqual(set(spatial(x=2, y=3) & batch(batch=10)), set(dot.shape), msg=backend.name)
                math.assert_close(dot, 4, msg=backend.name)

    def test_range(self):
        for backend in BACKENDS:
            with backend:
                math.assert_close(math.range(spatial('x'), 1, 5), [1, 2, 3, 4], msg=backend.name)
                math.assert_close(math.range(spatial('x'), 1), [0], msg=backend.name)

    def test_boolean_mask_1d(self):
        for backend in BACKENDS:
            with backend:
                x = math.range(spatial('range'), 4)
                mask = math.tensor([True, False, True, False], spatial('range'))
                math.assert_close(math.boolean_mask(x, 'range', mask), [0, 2], msg=backend.name)
                math.assert_close(x.range[mask], [0, 2], msg=backend.name)

    def test_boolean_mask_batched(self):
        for backend in BACKENDS:
            with backend:
                x = math.expand(math.range(spatial('x'), 4), batch(batch=2)) * math.tensor([1, -1])
                mask = math.tensor([[True, False, True, False], [False, True, False, False]], batch('batch'), spatial('x'))
                selected = math.boolean_mask(x, 'x', mask)
                expected_0 = math.tensor([(0, -0), (2, -2)], spatial('x'), channel('vector'))
                expected_1 = math.tensor([(1, -1)], spatial('x'), channel('vector'))
                math.assert_close(selected.batch[0], expected_0, msg=backend.name)
                math.assert_close(selected.batch[1], expected_1, msg=backend.name)
                math.assert_close(selected, x.x[mask], msg=backend.name)

    def test_boolean_mask_semi_batched(self):
        for backend in BACKENDS:
            with backend:
                x = math.range(spatial('x'), 4)
                mask = math.tensor([[True, False, True, False], [False, True, False, False]], batch('batch'), spatial('x'))
                selected = math.boolean_mask(x, 'x', mask)
                self.assertEqual(3, selected.shape.volume, msg=backend.name)

    def test_boolean_mask_dim_missing(self):
        for backend in BACKENDS:
            with backend:
                x = math.random_uniform(spatial(x=2))
                mask = math.tensor([True, False, True, True], batch('selection'))
                selected = math.boolean_mask(x, 'selection', mask)
                self.assertEqual(set(spatial(x=2) & batch(selection=3)), set(selected.shape), msg=backend.name)

    def test_scatter_1d(self):
        for backend in BACKENDS:
            with backend:
                base = math.ones(spatial(x=4))
                indices = math.wrap([1, 2], instance('points'))
                values = math.wrap([11, 12], instance('points'))
                updated = math.scatter(base, indices, values, mode='update', outside_handling='undefined')
                math.assert_close(updated, [1, 11, 12, 1])
                updated = math.scatter(base, indices, values, mode='add', outside_handling='undefined')
                math.assert_close(updated, [1, 12, 13, 1])
                # with vector dim
                indices = math.expand(indices, channel(vector=1))
                updated = math.scatter(base, indices, values, mode='update', outside_handling='undefined')
                math.assert_close(updated, [1, 11, 12, 1])

    def test_scatter_update_1d_batched(self):
        for backend in BACKENDS:
            with backend:
                # Only base batched
                base = math.zeros(spatial(x=4)) + math.tensor([0, 1])
                indices = math.wrap([1, 2], instance('points'))
                values = math.wrap([11, 12], instance('points'))
                updated = math.scatter(base, indices, values, mode='update', outside_handling='undefined')
                math.assert_close(updated, math.tensor([(0, 1), (11, 11), (12, 12), (0, 1)], spatial('x'), channel('vector')), msg=backend.name)
                # Only values batched
                base = math.ones(spatial(x=4))
                indices = math.wrap([1, 2], instance('points'))
                values = math.wrap([[11, 12], [-11, -12]], batch('batch'), instance('points'))
                updated = math.scatter(base, indices, values, mode='update', outside_handling='undefined')
                math.assert_close(updated, math.tensor([[1, 11, 12, 1], [1, -11, -12, 1]], batch('batch'), spatial('x')), msg=backend.name)
                # Only indices batched
                base = math.ones(spatial(x=4))
                indices = math.wrap([[0, 1], [1, 2]], batch('batch'), instance('points'))
                values = math.wrap([11, 12], instance('points'))
                updated = math.scatter(base, indices, values, mode='update', outside_handling='undefined')
                math.assert_close(updated, math.tensor([[11, 12, 1, 1], [1, 11, 12, 1]], batch('batch'), spatial('x')), msg=backend.name)
                # Everything batched
                base = math.zeros(spatial(x=4)) + math.tensor([0, 1], batch('batch'))
                indices = math.wrap([[0, 1], [1, 2]], batch('batch'), instance('points'))
                values = math.wrap([[11, 12], [-11, -12]], batch('batch'), instance('points'))
                updated = math.scatter(base, indices, values, mode='update', outside_handling='undefined')
                math.assert_close(updated, math.tensor([[11, 12, 0, 0], [1, -11, -12, 1]], batch('batch'), spatial('x')), msg=backend.name)

    def test_scatter_update_2d(self):
        for backend in BACKENDS:
            with backend:
                base = math.ones(spatial(x=3, y=2))
                indices = math.wrap([(0, 0), (0, 1), (2, 1)], instance('points'), channel('vector'))
                values = math.wrap([11, 12, 13], instance('points'))
                updated = math.scatter(base, indices, values, mode='update', outside_handling='undefined')
                math.assert_close(updated, math.tensor([[11, 1, 1], [12, 1, 13]], spatial('y,x')))

    def test_scatter_add_2d(self):
        for backend in BACKENDS:
            with backend:
                base = math.ones(spatial(x=3, y=2))
                indices = math.wrap([(0, 0), (0, 0), (0, 1), (2, 1)], instance('points'), channel('vector'))
                values = math.wrap([11, 11, 12, 13], instance('points'))
                updated = math.scatter(base, indices, values, mode='add', outside_handling='undefined')
                math.assert_close(updated, math.tensor([[23, 1, 1], [13, 1, 14]], spatial('y,x')))

    def test_scatter_2d_clamp(self):
        base = math.ones(spatial(x=3, y=2))
        indices = math.wrap([(-1, 0), (0, 2), (4, 3)], instance('points'), channel('vector'))
        values = math.wrap([11, 12, 13], instance('points'))
        updated = math.scatter(base, indices, values, mode='update', outside_handling='clamp')
        math.assert_close(updated, math.tensor([[11, 1, 1], [12, 1, 13]], spatial('y,x')))

    def test_scatter_2d_discard(self):
        base = math.ones(spatial(x=3, y=2))
        indices = math.wrap([(-1, 0), (0, 1), (3, 1)], instance('points'), channel('vector'))
        values = math.wrap([11, 12, 13], instance('points'))
        updated = math.scatter(base, indices, values, mode='update', outside_handling='discard')
        math.assert_close(updated, math.tensor([[1, 1, 1], [12, 1, 1]], spatial('y,x')))

    def test_sin(self):
        for backend in BACKENDS:
            with backend:
                math.assert_close(math.sin(math.zeros(spatial(x=4))), 0, abs_tolerance=1e-6, msg=backend.name)
                math.assert_close(math.sin(math.tensor(math.PI / 2)), 1, abs_tolerance=1e-6, msg=backend.name)
                math.assert_close(math.sin(math.tensor(math.PI)), 0, abs_tolerance=1e-6, msg=backend.name)
                math.assert_close(math.sin(math.tensor(math.PI * 3 / 2)), -1, abs_tolerance=1e-6, msg=backend.name)

    def test_cos(self):
        for backend in BACKENDS:
            with backend:
                math.assert_close(math.cos(math.zeros(spatial(x=4))), 1, abs_tolerance=1e-6, msg=backend.name)
                math.assert_close(math.cos(math.tensor(math.PI / 2)), 0, abs_tolerance=1e-6, msg=backend.name)
                math.assert_close(math.cos(math.tensor(math.PI)), -1, abs_tolerance=1e-6, msg=backend.name)
                math.assert_close(math.cos(math.tensor(math.PI * 3 / 2)), 0, abs_tolerance=1e-6, msg=backend.name)

    def test_any(self):
        for backend in BACKENDS:
            with backend:
                math.assert_close(math.any(math.tensor([[False, True], [False, False]], spatial('y,x')), dim='x'), [True, False])
                math.assert_close(math.any(math.tensor([[False, True], [False, False]], spatial('y,x')), dim='x,y'), True)
                math.assert_close(math.any(math.tensor([[False, True], [False, False]], spatial('y,x'))), True)

    def test_all(self):
        for backend in BACKENDS:
            with backend:
                math.assert_close(math.all(math.tensor([[False, True], [True, True]], spatial('y,x')), dim='x'), [False, True])
                math.assert_close(math.all(math.tensor([[False, True], [True, True]], spatial('y,x')), dim='x,y'), False)
                math.assert_close(math.all(math.tensor([[False, True], [True, True]], spatial('y,x'))), False)

    def test_imag(self):
        for backend in BACKENDS:
            with backend:
                math.assert_close(math.imag(math.ones(spatial(x=4))), 0, msg=backend.name)
                math.assert_close(math.imag(math.ones(spatial(x=4)) * 1j), 1, msg=backend.name)

    def test_real(self):
        for backend in BACKENDS:
            with backend:
                math.assert_close(math.real(math.ones(spatial(x=4))), 1, msg=backend.name)
                math.assert_close(math.real(math.ones(spatial(x=4)) * 1j), 0, msg=backend.name)

    def test_conjugate(self):
        for backend in BACKENDS:
            with backend:
                math.assert_close(math.conjugate(1 + 1j), 1 - 1j, msg=backend.name)
                math.assert_close(math.conjugate(1j * math.ones()), -1j, msg=backend.name)

    def test_convolution_1d_scalar(self):
        for backend in BACKENDS:
            with backend:
                x = math.tensor([1, 2, 3, 4], spatial('x'))
                identity_kernel1 = math.ones(spatial(x=1))
                identity_kernel2 = math.tensor([0, 1], spatial('x'))
                identity_kernel3 = math.tensor([0, 1, 0], spatial('x'))
                shift_kernel3 = math.tensor([0, 0, 1], spatial('x'))
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
                x = math.tensor([[1, 2, 3], [11, 12, 13]], batch('batch'), spatial('x')) * (2, -1)
                identity_kernel1 = math.ones(spatial(x=1))
                identity_kernel2 = math.tensor([0, 1], spatial('x'))
                identity_kernel3 = math.tensor([0, 1, 0], spatial('x'))
                shift_kernel3 = math.tensor([0, 0, 1], spatial('x'))
                # no padding
                math.assert_close(math.convolve(x, identity_kernel1), math.tensor([[1, 2, 3], [11, 12, 13]], batch('batch'), spatial('x')), msg=backend.name)
                math.assert_close(math.convolve(x, identity_kernel2), math.tensor([[2, 3], [12, 13]], batch('batch'), spatial('x')), msg=backend.name)
                math.assert_close(math.convolve(x, identity_kernel3), math.tensor([[2], [12]], batch('batch'), spatial('x')), msg=backend.name)
                math.assert_close(math.convolve(x, shift_kernel3), math.tensor([[3], [13]], batch('batch'), spatial('x')), msg=backend.name)
                # # zero-padding
                math.assert_close(math.convolve(x, identity_kernel1, math.extrapolation.ZERO), math.tensor([[1, 2, 3], [11, 12, 13]], batch('batch'), spatial('x')), msg=backend.name)
                math.assert_close(math.convolve(x, identity_kernel2, math.extrapolation.ZERO), math.tensor([[1, 2, 3], [11, 12, 13]], batch('batch'), spatial('x')), msg=backend.name)
                math.assert_close(math.convolve(x, identity_kernel3, math.extrapolation.ZERO), math.tensor([[1, 2, 3], [11, 12, 13]], batch('batch'), spatial('x')), msg=backend.name)
                math.assert_close(math.convolve(x, shift_kernel3, math.extrapolation.ZERO), math.tensor([[2, 3, 0], [12, 13, 0]], batch('batch'), spatial('x')), msg=backend.name)
                # # periodic padding
                math.assert_close(math.convolve(x, identity_kernel1, math.extrapolation.PERIODIC), math.tensor([[1, 2, 3], [11, 12, 13]], batch('batch'), spatial('x')), msg=backend.name)
                math.assert_close(math.convolve(x, identity_kernel2, math.extrapolation.PERIODIC), math.tensor([[1, 2, 3], [11, 12, 13]], batch('batch'), spatial('x')), msg=backend.name)
                math.assert_close(math.convolve(x, identity_kernel3, math.extrapolation.PERIODIC), math.tensor([[1, 2, 3], [11, 12, 13]], batch('batch'), spatial('x')), msg=backend.name)
                math.assert_close(math.convolve(x, shift_kernel3, math.extrapolation.PERIODIC), math.tensor([[2, 3, 1], [12, 13, 11]], batch('batch'), spatial('x')), msg=backend.name)
                # values and filters batched
                mixed_kernel = math.tensor([[0, 1, 0], [0, 0, 1]], batch('batch'), spatial('x'))
                math.assert_close(math.convolve(x, mixed_kernel, math.extrapolation.ZERO), math.tensor([[1, 2, 3], [12, 13, 0]], batch('batch'), spatial('x')), msg=backend.name)
                math.assert_close(math.convolve(x, mixed_kernel, math.extrapolation.PERIODIC), math.tensor([[1, 2, 3], [12, 13, 11]], batch('batch'), spatial('x')), msg=backend.name)
                # with output channels
                out_matrix = math.tensor([[1, 0], [0, 1], [1, 1]], channel('out'), channel('vector')).out.as_channel()
                kernel = identity_kernel3 * out_matrix
                expected = math.tensor([
                    [[2, 4, 6], [22, 24, 26]],
                    [[-1, -2, -3], [-11, -12, -13]],
                    [[1, 2, 3], [11, 12, 13]]], channel('out'), batch('batch'), spatial('x'))
                math.assert_close(math.convolve(x, kernel, math.extrapolation.ZERO), expected, msg=backend.name)

    # def test_convolution_2d(self):  # TODO
    #     pass

    def test_reshaped_native(self):
        a = math.random_uniform(channel(vector=2) & spatial(x=4, y=3))
        nat = math.reshaped_native(a, ['batch', a.shape.spatial, 'vector'], force_expand=False)
        self.assertEqual((1, 12, 2), nat.shape)
        nat = math.reshaped_native(a, [batch(batch=10), a.shape.spatial, channel(vector=2)], force_expand=False)
        self.assertEqual((1, 12, 2), nat.shape)
        nat = math.reshaped_native(a, [batch(batch=10), a.shape.spatial, channel(vector=2)], force_expand=['x'])
        self.assertEqual((1, 12, 2), nat.shape)
        nat = math.reshaped_native(a, [batch(batch=10), a.shape.spatial, channel(vector=2)], force_expand=True)
        self.assertEqual((10, 12, 2), nat.shape)
        nat = math.reshaped_native(a, [batch(batch=10), a.shape.spatial, channel(vector=2)], force_expand=['batch'])
        self.assertEqual((10, 12, 2), nat.shape)
        nat = math.reshaped_native(a, [a.shape.spatial, channel(vector=2, v2=2)], force_expand=False)
        self.assertEqual((12, 4), nat.shape)
        try:
            math.reshaped_native(a, [channel(vector=2, v2=2)], force_expand=False)
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

    def test_sparse(self):
        i = [[0, 1, 1],
             [2, 0, 2]]
        v = [3, 4, 5]
        shape = (2, 3)
        for backend in BACKENDS:
            if backend.supports(Backend.sparse_tensor):
                with backend:
                    matrix = backend.sparse_tensor(i, v, shape)
                    i_, v_ = backend.coordinates(matrix)
                    self.assertIsInstance(i_, tuple, msg=backend.name)
                    assert len(i_) == 2
                    assert len(v) == 3

    def test_rename_dims(self):
        t = math.zeros(spatial(x=5, y=4))
        self.assertEqual(math.rename_dims(t, 'x', 'z').shape.get_type('z'), 'spatial')
        self.assertEqual(math.rename_dims(t, ['x'], ['z']).shape.get_type('z'), 'spatial')
        self.assertEqual(math.rename_dims(t, ['x'], channel('z')).shape.get_type('z'), 'channel')
