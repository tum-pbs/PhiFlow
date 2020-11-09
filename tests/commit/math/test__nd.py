from itertools import product
from unittest import TestCase
from phi import math, field, geom
from phi.math import tensor, extrapolation, Tensor, PI

import numpy as np
import os


REF_DATA = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'reference_data')


def get_2d_sine(grid_size, L):
    indices = np.array(np.meshgrid(*list(map(range, grid_size))))
    phys_coord = indices.T * L / (grid_size[0])  # between [0, L)
    x, y = phys_coord.T
    d = np.sin(2 * np.pi * x + 1) * np.sin(2 * np.pi * y + 1)
    return d


class AbstractTestMathND(TestCase):

    def _test_scalar_gradient(self, ones: Tensor):
        cases = dict(difference=('central', 'forward', 'backward'),
                     padding=(extrapolation.ONE, extrapolation.BOUNDARY, extrapolation.PERIODIC, extrapolation.SYMMETRIC))
        for case_dict in [dict(zip(cases, v)) for v in product(*cases.values())]:
            try:
                scalar_grad = math.gradient(ones, dx=0.1, **case_dict)
                math.assert_close(scalar_grad, 0)
                self.assertEqual(scalar_grad.shape.names, ('batch', 'x', 'y', 'gradient'))
                self.assertEqual(scalar_grad.shape.sizes, (2, 4, 3, 2))
            except BaseException as e:
                raise AssertionError(e, case_dict)

    def _test_vector_gradient(self, meshgrid: Tensor, save=False):
        """test math.gradient for all differences and padding/extrapolations combinations against reference

        :param meshgrid: Tensor to be performed on
        :type meshgrid: Tensor
        :param save: save reference values
        :type save: bool, optional
        """
        cases = dict(difference=('central', 'forward', 'backward'),
                     padding=(extrapolation.ONE, extrapolation.BOUNDARY, extrapolation.PERIODIC, extrapolation.SYMMETRIC),
                     dx=(0.1, 1),
                     axes=(None, 0, 1))
        for case_dict in [dict(zip(cases, v)) for v in product(*cases.values())]:
            file = os.path.join(REF_DATA, 'vector_grad_%s.npy' % '_'.join(f'{key}={value}' for key, value in case_dict.items()))
            grad = math.gradient(meshgrid, **case_dict)
            if save:
                math.print(meshgrid, 'Base')
                np.save(file, grad.numpy())
                math.print(grad, str(case_dict))
            else:
                ref = np.load(file)
                math.assert_close(grad, ref)

    def _test_vector_laplace(self, meshgrid: Tensor, save=False):
        """test math.laplace for all differences and padding/extrapolations combinations against reference

        :param meshgrid: Tensor to be performed on
        :type meshgrid: Tensor
        :param save: save reference values
        :type save: bool, optional
        """
        cases = dict(padding=(extrapolation.ZERO, extrapolation.ONE, extrapolation.BOUNDARY, extrapolation.PERIODIC, extrapolation.SYMMETRIC),
                     dx=(0.1, 1),
                     axes=(None, 1, (0, 1)))
        for case_dict in [dict(zip(cases, v)) for v in product(*cases.values())]:
            file = os.path.join(REF_DATA, 'laplace_%s.npy' % '_'.join(f'{key}={value}' for key, value in case_dict.items()))
            laplace = math.laplace(meshgrid, **case_dict)
            if save:
                math.print(meshgrid, 'Base')
                np.save(file, laplace.numpy())
                math.print(laplace, str(case_dict))
            else:
                ref = np.load(file)
                math.assert_close(laplace, ref)


class TestMathNDNumpy(AbstractTestMathND):

    def test_gradient_scalar(self):
        scalar = tensor(np.ones([2, 4, 3, 1]))
        self._test_scalar_gradient(scalar)

    def test_gradient_vector(self):
        meshgrid = math.meshgrid(x=(0, 1, 2, 3), y=(0, -1))
        self._test_vector_gradient(meshgrid, save=False)

    def test_vector_laplace(self):
        meshgrid = math.meshgrid(x=(0, 1, 2, 3), y=(0, -1))
        self._test_vector_laplace(meshgrid, save=False)

    # Fourier Poisson

    def test_fourier_poisson_2d_periodic_convergence(self):
        """test for convergence of the laplace operator"""
        test_params = {
            'grid_sizes': [(16, 16), (32, 32), (64, 64), (128, 128), (256, 256)],
            'L': [1],  # FFT requires L>1
            'padding': [extrapolation.PERIODIC]  # Tests against sine requires padding
        }
        test_cases = [dict(zip(test_params, v)) for v in product(*test_params.values())]
        results = []
        for params in test_cases:
            N = params['grid_sizes'][0]
            L = params['L']
            padding = params['padding']
            dx = L / N
            sine_field = get_2d_sine((N, N), L=L)
            input_field = 8 * np.pi**2 * sine_field
            ref = -sine_field
            input_tensor = field.CenteredGrid(values=math.tensor(input_field, names=['x', 'y']), bounds=geom.Box([0,0], [N,N]), extrapolation=padding)
            val = math.fourier_poisson(input_tensor.values, dx)
            abs_diff = math.abs(val - ref)
            results.append({'abs_tolerance': math.max(abs_diff), 'rel_tolerance': math.max(abs_diff / ref), **params})
        abs_differences = [d['abs_tolerance'] for d in results]
        assert all(np.diff(np.array(abs_differences)) < 0), f"fourier_poisson not converging: {abs_differences}"

    def test_downsample2x(self):
        meshgrid = math.meshgrid(x=(0, 1, 2, 3), y=(0, -1, -2))
        half_size = math.downsample2x(meshgrid, extrapolation.BOUNDARY)
        math.print(meshgrid, 'Full size')
        math.print(half_size, 'Half size')
        math.assert_close(half_size.vector[0], tensor([[0.5, 2.5], [0.5, 2.5]], names='y,x'))
        math.assert_close(half_size.vector[1], tensor([[-0.5, -0.5], [-2, -2]], names='y,x'))

    def test_upsample2x(self):
        meshgrid = math.meshgrid(x=(0, 1, 2, 3), y=(0, -1, -2))
        double_size = math.upsample2x(meshgrid, extrapolation.BOUNDARY)
        same_size = math.downsample2x(double_size)
        math.print(meshgrid, 'Normal size')
        math.print(double_size, 'Double size')
        math.print(same_size, 'Same size')
        math.assert_close(meshgrid.x[1:-1].y[1:-1], same_size.x[1:-1].y[1:-1])

    def test_fourier_poisson_2d_periodic(self):
        test_params = {
            'size': [16, 32, 40],
            'L': [0.5, 1],
        }
        test_cases = [dict(zip(test_params, v)) for v in product(*test_params.values())]
        for params in test_cases:
            vec = math.meshgrid(x=params['size'], y=params['size'])
            sine_field = math.prod(math.sin(2 * PI * params['L'] * vec / 40 + 1), 'vector')
            sin_lap_ref = - 2 * (2 * PI / params['size']) ** 2 * params['L'] ** 2 * sine_field  # leading 2 from from x-y cross terms
            sin_lap = math.fourier_laplace(sine_field, 1)
            math.assert_close(sin_lap, sin_lap_ref, rel_tolerance=0, abs_tolerance=1e-5)

            N = params['grid_sizes'][0]
            L = params['L']
            padding = params['padding']
            dx = L / N
            sine_field = get_2d_sine((N, N), L=L)
            input_field = 8 * np.pi**2 * sine_field
            ref = -sine_field
            input_tensor = field.CenteredGrid(values=math.tensor(input_field, names=['x', 'y']), bounds=geom.Box([0,0], [N,N]), extrapolation=padding)
            val = math.fourier_poisson(input_tensor.values, dx)
            try:
                math.assert_close(val, ref, rel_tolerance=1e-5)
            except BaseException as e:
                abs_error = math.abs(ref - val)
                max_abs_error = math.max(abs_error)
                max_rel_error = math.max(math.abs(abs_error / ref))
                variation_str = "\n".join([
                    f"max_absolute_error: {max_abs_error}",
                    f"max_relative_error: {max_rel_error}",
                ])
                print(ref)
                print(val.values)
                raise AssertionError(e, f"{variation_str}\n{params}")

    # Fourier Laplace

    def test_fourier_laplace_2d_periodic_convergence(self):
        """test for convergence of the laplace operator"""
        test_params = {
            'size': [16, 32, 40],
            'L': [1],
        }
        test_cases = [dict(zip(test_params, v)) for v in product(*test_params.values())]
        for params in test_cases:
            vec = math.meshgrid(x=params['size'], y=params['size'])
            sine_field = math.prod(math.sin(2 * PI * params['L'] * vec / params['size'] + 1), 'vector')
            sin_lap_ref = - 2 * (2 * PI / params['size']) ** 2 * sine_field  # leading 2 from from x-y cross terms
            sin_lap = math.fourier_laplace(sine_field, 1)
            math.assert_close(sin_lap, sin_lap_ref, rel_tolerance=0, abs_tolerance=1e-5)

    def test_fourier_laplace_2d_periodic(self):
        test_params = {
            'grid_sizes': [(32, 32), (64, 64)],
            'L': [1, 3],  # FFT requires L>1
            'padding': [extrapolation.PERIODIC]  # Tests against sine requires padding
        }
        test_cases = [dict(zip(test_params, v)) for v in product(*test_params.values())]
        for params in test_cases:
            N = params['grid_sizes'][0]
            L = params['L']
            padding = params['padding']
            dx = L / N
            sine_field = get_2d_sine((N, N), L=L)
            ref = 8 * np.pi**2 * sine_field
            input_field = -sine_field
            input_tensor = field.CenteredGrid(values=math.tensor(input_field, names=['x', 'y']), bounds=geom.Box([0,0], [N,N]), extrapolation=padding)
            val = math.fourier_laplace(input_tensor.values, dx)
            try:
                math.assert_close(val, ref, rel_tolerance=1e-5)
            except BaseException as e:
                abs_error = math.abs(val - ref)
                max_abs_error = math.max(abs_error)
                max_rel_error = math.max(math.abs(abs_error / ref))
                variation_str = "\n".join([
                    f"max_absolute_error: {max_abs_error}",
                    f"max_relative_error: {max_rel_error}",
                ])
                raise AssertionError(e, f"{variation_str}\n{params}")

    # Arakawa

    def test_poisson_bracket(self):
        """test poisson_bracket wrapper"""
        return

    @staticmethod
    def arakawa_reference_implementation(zeta, psi, d):
        """pure Python exact implementation from paper"""
        def jpp(zeta, psi, d, i, j):
            return ((zeta[i + 1, j] - zeta[i - 1, j]) * (psi[i, j + 1] - psi[i, j - 1])
                    - (zeta[i, j + 1] - zeta[i, j - 1]) * (psi[i + 1, j] - psi[i - 1, j])) / (4 * d**2)

        def jpx(zeta, psi, d, i, j):
            return (zeta[i + 1, j] * (psi[i + 1, j + 1] - psi[i + 1, j - 1])
                    - zeta[i - 1, j] * (psi[i - 1, j + 1] - psi[i - 1, j - 1])
                    - zeta[i, j + 1] * (psi[i + 1, j + 1] - psi[i - 1, j + 1])
                    + zeta[i, j - 1] * (psi[i + 1, j - 1] - psi[i - 1, j - 1])) / (4 * d**2)

        def jxp(zeta, psi, d, i, j):
            return (zeta[i + 1, j + 1] * (psi[i, j + 1] - psi[i + 1, j])
                    - zeta[i - 1, j - 1] * (psi[i - 1, j] - psi[i, j - 1])
                    - zeta[i - 1, j + 1] * (psi[i, j + 1] - psi[i - 1, j])
                    + zeta[i + 1, j - 1] * (psi[i + 1, j] - psi[i, j - 1])) / (4 * d**2)
        val = np.zeros_like(zeta)
        for i in range(0, zeta.shape[0] - 1):
            for j in range(0, zeta.shape[1] - 1):
                val[i, j] += (jpp(zeta, psi, d, i, j) + jpx(zeta, psi, d, i, j) + jxp(zeta, psi, d, i, j))
        val = val / 3
        return val

    def test__periodic_2d_arakawa_poisson_bracket(self):
        """test _periodic_2d_arakawa_poisson_bracket implementation"""
        test_params = {
            'grid_size': [(4, 4), (32, 32)],
            'dx': [0.1, 1],
            'gen_func': [lambda grid_size: np.random.rand(*grid_size).reshape(grid_size)]
        }
        test_cases = [dict(zip(test_params, v)) for v in product(*test_params.values())]
        for params in test_cases:
            grid_size = params['grid_size']
            d1 = params['gen_func'](grid_size)
            d2 = params['gen_func'](grid_size)
            dx = params['dx']
            padding = extrapolation.PERIODIC
            ref = self.arakawa_reference_implementation(np.pad(d1.copy(), 1, mode='wrap'), np.pad(d2.copy(), 1, mode='wrap'), dx)[1:-1, 1:-1]
            d1_tensor = field.CenteredGrid(values=math.tensor(d1, names=['x', 'y']), bounds=geom.Box([0, 0], list(grid_size)), extrapolation=padding)
            d2_tensor = field.CenteredGrid(values=math.tensor(d2, names=['x', 'y']), bounds=geom.Box([0, 0], list(grid_size)), extrapolation=padding)
            val = math._nd._periodic_2d_arakawa_poisson_bracket(d1_tensor.values, d2_tensor.values, dx)
            try:
                math.assert_close(ref, val, rel_tolerance=1e-5, abs_tolerance=0)
            except BaseException as e:
                abs_error = math.abs(val - ref)
                max_abs_error = math.max(abs_error)
                max_rel_error = math.max(math.abs(abs_error / ref))
                variation_str = "\n".join([
                    f"max_absolute_error: {max_abs_error}",
                    f"max_relative_error: {max_rel_error}",
                ])
                print(ref)
                print(val)
                raise AssertionError(e, params, variation_str)
