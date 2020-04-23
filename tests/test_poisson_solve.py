from unittest import TestCase

import numpy as np
from phi import math

from phi.flow import CLOSED, PERIODIC, OPEN, Domain, poisson_solve, Noise
from phi.physics.pressuresolver.geom import GeometricCG
from phi.physics.pressuresolver.sparse import SparseCG, SparseSciPy
from phi.physics.pressuresolver.fourier import FourierSolver
from phi.physics.field import CenteredGrid
from phi.geom.geometry import AABox


def _generate_examples():
    # --- Example 1 ---
    ex1 = np.tile(np.linspace(1, 0, 5), [4, 1])
    ex1 = math.expand_dims(math.expand_dims(ex1, -1), 0) - math.mean(ex1)
    # --- Example 2 ---
    ex2 = np.zeros([1, 4, 5, 1])
    ex2[0, :, 2, 0] = 1
    ex2 -= math.mean(ex2)
    # --- Stack examples to batch ---
    return math.concat([ex1, ex2], axis=0)


def _test_solve_no_obstacles(domain, solver):
    print('Testing domain with boundaries: %s' % (domain.boundaries,))
    data_in = _generate_examples()
    p = poisson_solve(domain.centered_grid(data_in), domain, solver=solver)[0]
    np.testing.assert_almost_equal(p.laplace().data, data_in, decimal=5)
    if domain.boundaries is CLOSED:
        np.testing.assert_almost_equal(p.laplace().data, data_in, decimal=5)
    # rows = math.unstack(p.data, 1)
    # for row in rows[1:]:
    #     np.testing.assert_almost_equal(row, rows[0], decimal=5)


def _test_random_closed(solver):
    domain = Domain([40, 32], boundaries=CLOSED)
    div = domain.centered_grid(Noise())
    div_ = poisson_solve(div, domain, solver)[0].laplace()
    np.testing.assert_almost_equal(div.data, div_.data, decimal=3)


def _test_random_open(solver):
    domain = Domain([40, 32], boundaries=OPEN)
    div = domain.centered_grid(Noise())
    div_ = poisson_solve(div, domain, solver)[0].laplace()
    np.testing.assert_almost_equal(div.data, div_.data, decimal=3)


def _test_random_periodic(solver):
    domain = Domain([40, 32], boundaries=PERIODIC)
    div = domain.centered_grid(Noise())
    div_ = poisson_solve(div, domain, solver)[0].laplace()
    np.testing.assert_almost_equal(div.data, div_.data, decimal=3)


def _test_all(solver):
    for domain in DOMAINS:
        _test_solve_no_obstacles(domain, solver)
    _test_random_closed(solver)
    _test_random_open(solver)
    _test_random_periodic(solver)


DOMAINS = [
    Domain([4, 5], boundaries=CLOSED),
    Domain([4, 5], boundaries=OPEN),
    Domain([4, 5], boundaries=PERIODIC),
    Domain([4, 5], boundaries=[CLOSED, PERIODIC]),
    Domain([4, 5], boundaries=[CLOSED, OPEN]),
    Domain([4, 5], boundaries=[PERIODIC, OPEN]),
]


class TestPoissonSolve(TestCase):

    def test_equal_results(self):
        data_in = _generate_examples()
        for domain in DOMAINS:
            pressure_fields = [poisson_solve(domain.centered_grid(data_in), domain, solver=solver)[0].data
                               for solver in [SparseCG(), GeometricCG()]]
            for field in pressure_fields[1:]:
                np.testing.assert_almost_equal(field, pressure_fields[0], decimal=4)

    def test_sparse_cg(self):
        _test_all(SparseCG())

    def test_sparse_scipy(self):
        _test_all(SparseSciPy())

    def test_geometric_cg(self):
        _test_all(GeometricCG())


# def _run_higher_order_fft_reconstruction(in_field, set_accuracy, tolerance=20, order=2):
#     # Higher Order FFT test
#     mean = math.mean(in_field).data
#     centered_field = in_field - mean
#     fft_poisson = math.fourier_poisson(in_field, times=order)
#     fft_poisson += mean
#     fft_reconst = math.fourier_laplace(fft_poisson)
#     for _ in range(order - 1):
#         fft_reconst = math.fourier_laplace(fft_reconst)
#     error = (in_field - fft_reconst) / in_field
#     max_error = np.max(np.abs(error.data))
#     passed = max_error < tolerance * set_accuracy
#     print("{:.2g} vs. {:.2g}".format(max_error, tolerance * set_accuracy))
#     #assert passed, "{}^2 reconstruction not within set accuracy. {:.2g} vs. {:.2g}".format('FFT*2', max_error, tolerance * set_accuracy)
#
#
# def _test_reconstruction_first_order(in_field, solve_func, laplace_func):
#     # Test Reconstruction
#     mean = math.mean(in_field).data
#     centered_field = in_field - mean
#     ret = solve_func(centered_field)
#     try:
#         solved_field, it = ret
#     except:
#         solved_field = ret
#     reconst1 = laplace_func(solved_field) + mean  # Reconstruct Input
#     np.testing.assert_almost_equal(reconst1.data, in_field.data, decimal=3)
#
#
# def _test_reconstruction_second_order(in_field, solve_func, laplace_func):
#     # Calculate 1st order
#     mean = math.mean(in_field).data
#     centered_field = in_field - mean
#     ret = solve_func(centered_field)
#     try:
#         solved_field, it = ret
#     except:
#         solved_field = ret
#     # Calculate 2nd order
#     ret2 = solve_func(solved_field)
#     try:
#         solved_field2, it = ret2
#     except:
#         solved_field2 = ret2
#     reconst2 = laplace_func(laplace_func(solved_field2)) + mean
#     np.testing.assert_almost_equal(reconst2.data, in_field.data, decimal=1)
#
#
# class TestReconstruction(TestCase):
#
#     def test_reconst(self, set_accuracy=1e-5, shape=[40, 40], first_order_tolerance=3, second_order_tolerance=40,
#                      boundary_list=[PERIODIC, OPEN, CLOSED]):
#         for boundary in boundary_list:
#             domain = Domain(shape, boundaries=(boundary, boundary))
#             solver_list = [
#                 ('SparseCG', lambda field: poisson_solve(field, domain, SparseCG(accuracy=set_accuracy)), lambda field: field.laplace()),
#                 ('GeometricCG', lambda field: poisson_solve(field, domain, GeometricCG(accuracy=set_accuracy)), lambda field: field.laplace()),
#                 #('SparseSciPy', lambda field: poisson_solve(field, domain, SparseSciPy()), lambda field: field.laplace()),
#                 # ('Fourier', lambda field: poisson_solve(field, domain, Fourier()))]  # TODO: poisson_solve() causes resolution to be empty
#                 ('FFT', math.fourier_poisson, math.fourier_laplace)]
#             in_data = CenteredGrid.sample(Noise(), domain)
#             sloped_data = (np.array([np.arange(shape[1]) for _ in range(shape[0])]).reshape([1] + shape + [1]) / 10 + 1)
#             in_data = in_data.copied_with(data=sloped_data)
#             for name, solver, laplace in solver_list:
#                 print('Testing {} boundary with {} solver... '.format(boundary, name)),
#                 _test_reconstruction_first_order(in_data, solver, laplace)
#                 _test_reconstruction_second_order(in_data, solver, laplace)
#             print('Testing {} boundary with {} solver... '.format(boundary, 'higher order FFT')),
#             _run_higher_order_fft_reconstruction(in_data, set_accuracy, order=2, tolerance=second_order_tolerance)
