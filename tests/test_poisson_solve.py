from unittest import TestCase

import numpy as np
from phi import math

from phi.flow import CLOSED, PERIODIC, OPEN, Domain, poisson_solve, Noise
from phi.physics.pressuresolver.geom import GeometricCG
from phi.physics.pressuresolver.sparse import SparseCG, SparseSciPy
from phi.physics.field import CenteredGrid


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


def run_second_order_fft_reconstruction(in_field, set_accuracy, second_order_tolerance=20):
    # Second Order FFT test
    mean = math.mean(in_field).data
    centered_field = in_field - mean
    fft_poisson2 = math.fourier_poisson(in_field, times=2)
    fft_poisson2 += mean
    fft_reconst2 = math.fourier_laplace(math.fourier_laplace(fft_poisson2))
    error2 = (in_field - fft_reconst2)/in_field
    max_error2 = np.max(np.abs(error2.data))
    assert max_error2 < second_order_tolerance*set_accuracy, "{}^2 reconstruction not within set accuracy. {:.2g} vs. {:.2g}".format('FFT*2', max_error2, second_order_tolerance*set_accuracy)

def test_reconstruction_first_order(in_field, solve_func, laplace_func, set_accuracy, name, first_order_tolerance=2):
    # Test Reconstruction
    mean = math.mean(in_field).data
    centered_field = in_field - mean
    ret = solve_func(centered_field)
    try:
        solved_field, it = ret
    except:
        solved_field = ret
    reconst1 = laplace_func(solved_field) + mean  # Reconstruct Input
    error = (in_field - reconst1)/in_field
    max_error = np.max(np.abs(error.data))
    print("{:.2g}/{:.2g}".format(max_error, first_order_tolerance*set_accuracy))
    assert max_error < first_order_tolerance*set_accuracy, "{} reconstruction not within set accuracy. {:.2g} vs. {:.2g}".format(name, max_error, first_order_tolerance*set_accuracy)

def test_reconstruction_second_order(in_field, solve_func, laplace_func, set_accuracy, name, second_order_tolerance=20):
    # Calculate 1st order
    mean = math.mean(in_field).data
    centered_field = in_field - mean
    ret = solve_func(centered_field)
    try:
        solved_field, it = ret
    except:
        solved_field = ret
    # Calculate 2nd order
    ret2 = solve_func(solved_field)
    try:
        solved_field2, it = ret2
    except:
        solved_field2 = ret2
    reconst2 = laplace_func(laplace_func(solved_field2)) + mean
    error2 = (in_field - reconst2)/in_field
    max_error2 = np.max(np.abs(error2.data))
    assert max_error2 < second_order_tolerance*set_accuracy, "{}^2 reconstruction not within set accuracy. {:.2g} vs. {:.2g}".format(name, max_error2, second_order_tolerance*set_accuracy)

class TestReconstruction(TestCase):

    def test_reconst(self, set_accuracy=1e-5, shape=[40, 32], first_order_tolerance=2, second_order_tolerance=20,
                 boundary_list=[PERIODIC, OPEN, CLOSED]):
        for boundary in boundary_list:
            domain = Domain(shape, boundaries=boundary)
            solver_list=[('SparseCG', lambda field: poisson_solve(field, domain, SparseCG(accuracy=set_accuracy)), lambda x: x.laplace()),
                         ('GeometricCG', lambda field: poisson_solve(field, domain, GeometricCG(accuracy=set_accuracy)), lambda x: x.laplace()),
                         ('SparseSciPy', lambda field: poisson_solve(field, domain, SparseSciPy()), lambda x: x.laplace()),
                         ('FFT', math.fourier_poisson, math.fourier_laplace)]
            in_data = CenteredGrid.sample(Noise(), domain)
            sloped_data = (np.array([np.arange(shape[1]) for _ in range(shape[0])]).reshape(1, *shape, 1)/10+1)
            in_data = in_data.copied_with(data=sloped_data)
            for name, solver, laplace in solver_list:
                print('Testing {} boundary with {} solver... '.format(boundary, name), end='')
                test_reconstruction_first_order(in_data, solver, laplace, set_accuracy, name, first_order_tolerance=first_order_tolerance)
                #test_reconstruction_second_order(in_data, solver, laplace, set_accuracy, name, second_order_tolerance=second_order_tolerance)
            #print('Testing {} boundary with {} solver'.format(boundary, 'higher order FFT'))
            #run_second_order_fft_reconstruction(in_data, set_accuracy, second_order_tolerance=second_order_tolerance)

