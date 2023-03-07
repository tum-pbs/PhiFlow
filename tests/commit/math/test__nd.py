from itertools import product
from unittest import TestCase
from phi import math
from phi.math import wrap, extrapolation, PI, tensor, batch, spatial, instance, channel, NAN, vec

import numpy as np
import os


REF_DATA = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'reference_data')


class TestMathNDNumpy(TestCase):

    def test_gradient_scalar(self):
        ones = tensor(np.ones([2, 4, 3]), batch('batch'), spatial('x,y'))
        cases = dict(difference=('central', 'forward', 'backward'),
                     padding=(extrapolation.ONE, extrapolation.BOUNDARY, extrapolation.PERIODIC, extrapolation.SYMMETRIC))
        for case_dict in [dict(zip(cases, v)) for v in product(*cases.values())]:
            scalar_grad = math.spatial_gradient(ones, dx=0.1, **case_dict)
            math.assert_close(scalar_grad, 0)
            self.assertEqual(scalar_grad.shape.names, ('batch', 'x', 'y', 'gradient'))
            ref_shape = (2, 4, 3, 2) if case_dict['padding'] is not None else ((2, 2, 1, 2) if case_dict['difference'] == 'central' else (2, 3, 2, 2))
            self.assertEqual(scalar_grad.shape.sizes, ref_shape)

    def test_gradient_vector(self):
        meshgrid = math.meshgrid(x=4, y=3)
        cases = dict(difference=('central', 'forward', 'backward'),
                     padding=(extrapolation.ONE, extrapolation.BOUNDARY, extrapolation.PERIODIC, extrapolation.SYMMETRIC),
                     dx=(0.1, 1),
                     dims=(spatial, ('x', 'y'), ))
        for case_dict in [dict(zip(cases, v)) for v in product(*cases.values())]:
            grad = math.spatial_gradient(meshgrid, **case_dict)
            inner = grad.x[1:-1].y[1:-1]
            math.assert_close(inner.gradient[0].vector[1], 0)
            math.assert_close(inner.gradient[1].vector[0], 0)
            math.assert_close(inner.gradient[0].vector[0], 1 / case_dict['dx'])
            math.assert_close(inner.gradient[1].vector[1], 1 / case_dict['dx'])
            self.assertEqual(grad.shape.get_size('vector'), 2)
            self.assertEqual(grad.shape.get_size('gradient'), 2)
            ref_shape = (4, 3) if case_dict['padding'] is not None else ((2, 1) if case_dict['difference'] == 'central' else (3, 2))
            self.assertEqual((grad.shape.get_size('x'), grad.shape.get_size('y')), ref_shape)

    def test_gradient_1d_vector(self):
        a = tensor([(0,), (1,), (2,)], spatial('x'), channel('vector'))
        math.assert_close(tensor([0.5, 1, 0.5], spatial('x')), math.spatial_gradient(a))

    def test_vector_laplace(self):
        meshgrid = math.meshgrid(x=(0, 1, 2, 3), y=(0, -1))
        cases = dict(padding=(extrapolation.ZERO, extrapolation.ONE, extrapolation.BOUNDARY, extrapolation.PERIODIC, extrapolation.SYMMETRIC),
                     dx=(0.1, 1),
                     dims=(spatial, ('x',), ('y',), ('x', 'y')))
        for case_dict in [dict(zip(cases, v)) for v in product(*cases.values())]:
            laplace = math.laplace(meshgrid, **case_dict)

    # Fourier Poisson

    def test_downsample2x(self):
        meshgrid = math.meshgrid(x=(0, 1, 2, 3), y=(0, -1, -2))
        half_size = math.downsample2x(meshgrid, extrapolation.BOUNDARY)
        math.print(meshgrid, 'Full size')
        math.print(half_size, 'Half size')
        math.assert_close(half_size.vector[0], wrap([[0.5, 2.5], [0.5, 2.5]], spatial('y,x')))
        math.assert_close(half_size.vector[1], wrap([[-0.5, -0.5], [-2, -2]], spatial('y,x')))

    def test_upsample2x(self):
        meshgrid = math.meshgrid(x=(0, 1, 2, 3), y=(0, -1, -2))
        double_size = math.upsample2x(meshgrid, extrapolation.BOUNDARY)
        same_size = math.downsample2x(double_size)
        math.print(meshgrid, 'Normal size')
        math.print(double_size, 'Double size')
        math.print(same_size, 'Same size')
        math.assert_close(meshgrid.x[1:-1].y[1:-1], same_size.x[1:-1].y[1:-1])

    def test_finite_fill_3x3_sanity(self):
        values = tensor([[NAN, NAN, NAN],
                         [NAN, 1,   NAN],
                         [NAN, NAN, NAN]], spatial('x, y'))
        math.assert_close(math.ones(spatial(x=3, y=3)), math.finite_fill(values, distance=1, diagonal=True))
        math.assert_close(math.ones(spatial(x=3, y=3)), math.finite_fill(values, distance=2, diagonal=False))
        values = tensor([[1, 1, 1], [1, NAN, 1], [1, 1, 1]], spatial('x,y'))
        math.assert_close(math.ones(spatial(x=3, y=3)), math.finite_fill(values, distance=1, diagonal=False))
        math.assert_close(math.ones(spatial(x=3, y=3)), math.finite_fill(values, distance=1, diagonal=True))

    def test_finite_fill_3x3(self):
        values = tensor([[NAN, NAN, NAN],
                         [NAN, NAN, 4  ],
                         [NAN, 2,   NAN]], spatial('x, y'))
        expected_diag = tensor([[NAN, 4,   4],
                                [2,   3,   4],
                                [2,   2,   3]], spatial('x, y'))
        math.assert_close(expected_diag, math.finite_fill(values, distance=1, diagonal=True))
        expected = tensor([[NAN, 3.5, 4],
                           [2.5, 3,   4],
                           [2,   2,   3]], spatial('x, y'))
        math.assert_close(expected, math.finite_fill(values, distance=2, diagonal=False))

    def test_extrapolate_valid_3x3_sanity(self):
        values = tensor([[0, 0, 0],
                         [0, 1, 0],
                         [0, 0, 0]], spatial('x, y'))
        valid = values
        extrapolated_values, extrapolated_valid = math.masked_fill(values, valid)
        expected_values = math.ones(spatial(x=3, y=3))
        expected_valid = extrapolated_values
        math.assert_close(extrapolated_values, expected_values)
        math.assert_close(extrapolated_valid, expected_valid)

    def test_extrapolate_valid_3x3(self):
        valid = tensor([[0, 0, 0],
                        [0, 0, 1],
                        [1, 0, 0]], spatial('x, y'))
        values = tensor([[1, 0, 2],
                         [0, 0, 4],
                         [2, 0, 0]], spatial('x, y'))
        expected_valid = tensor([[0, 1, 1],
                                 [1, 1, 1],
                                 [1, 1, 1]], spatial('x, y'))
        expected_values = tensor([[1, 4, 4],
                                  [2, 3, 4],
                                  [2, 3, 4]], spatial('x, y'))
        extrapolated_values, extrapolated_valid = math.masked_fill(values, valid)
        math.assert_close(extrapolated_values, expected_values)
        math.assert_close(extrapolated_valid, expected_valid)

    def test_extrapolate_valid_4x4(self):
        valid = tensor([[0, 0, 0, 0],
                        [0, 0, 1, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 0]], spatial('x, y'))
        values = tensor([[1, 0, 0, 0],
                         [0, 0, 4, 0],
                         [2, 0, 0, 0],
                         [0, 0, 0, 1]], spatial('x, y'))
        expected_valid = tensor([[1, 1, 1, 1],
                                 [1, 1, 1, 1],
                                 [1, 1, 1, 1],
                                 [1, 1, 1, 1]], spatial('x, y'))
        expected_values = tensor([[3, 4, 4, 4],
                                  [2, 3, 4, 4],
                                  [2, 3, 4, 4],
                                  [2, 2, 3.25, 4]], spatial('x, y'))
        extrapolated_values, extrapolated_valid = math.masked_fill(values, valid, 2)
        math.assert_close(extrapolated_values, expected_values)
        math.assert_close(extrapolated_valid, expected_valid)

    def test_extrapolate_valid_3D_3x3x3_1(self):
        valid = tensor([[[0, 0, 0],
                         [0, 0, 0],
                         [0, 0, 0]],
                        [[0, 0, 1],
                         [0, 0, 0],
                         [1, 0, 0]],
                        [[0, 0, 0],
                         [0, 0, 0],
                         [0, 0, 0]]], spatial('x, y, z'))
        values = tensor([[[0, 0, 0],
                          [0, 0, 0],
                          [0, 0, 0]],
                         [[1, 0, 4],
                          [0, 0, 0],
                          [2, 0, 0]],
                         [[0, 0, 0],
                          [0, 0, 0],
                          [0, 0, 0]]], spatial('x, y, z'))
        expected_valid = tensor([[[0, 1, 1],
                                  [1, 1, 1],
                                  [1, 1, 0]],
                                 [[0, 1, 1],
                                  [1, 1, 1],
                                  [1, 1, 0]],
                                 [[0, 1, 1],
                                  [1, 1, 1],
                                  [1, 1, 0]]], spatial('x, y, z'))
        expected_values = tensor([[[0, 4, 4],
                                   [2, 3, 4],
                                   [2, 2, 0]],
                                  [[1, 4, 4],
                                   [2, 3, 4],
                                   [2, 2, 0]],
                                  [[0, 4, 4],
                                   [2, 3, 4],
                                   [2, 2, 0]]], spatial('x, y, z'))
        extrapolated_values, extrapolated_valid = math.masked_fill(values, valid, 1)
        math.assert_close(extrapolated_values, expected_values)
        math.assert_close(extrapolated_valid, expected_valid)

    def test_extrapolate_valid_3D_3x3x3_2(self):
        valid = tensor([[[0, 0, 0],
                         [0, 0, 0],
                         [0, 0, 0]],
                        [[0, 0, 1],
                         [0, 0, 0],
                         [1, 0, 0]],
                        [[0, 0, 0],
                         [0, 0, 0],
                         [0, 0, 0]]], spatial('x, y, z'))
        values = tensor([[[0, 0, 0],
                          [0, 0, 0],
                          [0, 0, 0]],
                         [[1, 0, 4],
                          [0, 0, 0],
                          [2, 0, 0]],
                         [[0, 0, 0],
                          [0, 0, 0],
                          [0, 0, 0]]], spatial('x, y, z'))
        expected_valid = math.ones(spatial(x=3, y=3, z=3))
        expected_values = tensor([[[3, 4, 4],
                                   [2, 3, 4],
                                   [2, 2, 3]],
                                  [[3, 4, 4],
                                   [2, 3, 4],
                                   [2, 2, 3]],
                                  [[3, 4, 4],
                                   [2, 3, 4],
                                   [2, 2, 3]]], spatial('x, y, z'))
        extrapolated_values, extrapolated_valid = math.masked_fill(values, valid, 2)
        math.assert_close(extrapolated_values, expected_values)
        math.assert_close(extrapolated_valid, expected_valid)

    def test_fourier_laplace_2d_periodic(self):
        """test for convergence of the laplace operator"""
        test_params = {
            'size': [16, 32, 40],
            'L': [1, 2, 3],  # NOTE: Cannot test with less than 1 full wavelength
        }
        test_cases = [dict(zip(test_params, v)) for v in product(*test_params.values())]
        for params in test_cases:
            vec = math.meshgrid(x=params['size'], y=params['size'])
            sine_field = math.prod(math.sin(2 * PI * params['L'] * vec / params['size'] + 1), 'vector')
            sin_lap_ref = - 2 * (2 * PI * params['L'] / params['size']) ** 2 * sine_field  # leading 2 from from x-y cross terms
            sin_lap = math.fourier_laplace(sine_field, 1)
            # try:
            math.assert_close(sin_lap, sin_lap_ref, rel_tolerance=0, abs_tolerance=1e-5)
            # except BaseException as e:  # Enable the try/catch to get more info about the deviation
            #     abs_error = math.abs(sin_lap - sin_lap_ref)
            #     max_abs_error = math.max(abs_error)
            #     max_rel_error = math.max(math.abs(abs_error / sin_lap_ref))
            #     variation_str = "\n".join(
            #         [
            #             f"max_absolute_error: {max_abs_error}",
            #             f"max_relative_error: {max_rel_error}",
            #         ]
            #     )
            #     print(f"{variation_str}\n{params}")
            #     raise AssertionError(e, f"{variation_str}\n{params}")

    def test_vector_length(self):
        v = tensor([(0, 0), (1, 1), (-1, 0)], instance('values'), channel('vector'))
        le = math.vec_length(v)
        math.assert_close(le, [0, 1.41421356237, 1])
        le = math.vec_length(v, eps=0.01)
        math.assert_close(le, [1e-1, 1.41421356237, 1])

    def test_dim_mask(self):
        math.assert_close((1, 0, 0), math.dim_mask(spatial('x,y,z'), 'x'))
        math.assert_close((1, 0, 1), math.dim_mask(spatial('x,y,z'), 'x,z'))

    def test_vec_expand(self):
        v = math.vec(x=0, y=math.linspace(0, 1, instance(points=10)))
        self.assertEqual(set(instance(points=10) & channel(vector='x,y')), set(v.shape))

    def test_vec_sequence(self):
        size = vec(batch('size'), 4, 8, 16, 32)
        self.assertEqual(batch(size='4,8,16,32'), size.shape)
        math.assert_close([4, 8, 16, 32], size)
        size = vec(batch('size'), [4, 8, 16, 32])
        self.assertEqual(batch(size='4,8,16,32'), size.shape)
        math.assert_close([4, 8, 16, 32], size)

    def test_vec_component_sequence(self):
        math.assert_close(wrap([(0, 1), (0, 2)], spatial('sequence'), channel(vector='x,y')), vec(x=0, y=(1, 2)))
        math.assert_close(wrap([(0, 1), (0, 2)], instance('sequence'), channel(vector='x,y')), vec(x=0, y=[1, 2]))
