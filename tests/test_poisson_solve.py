from unittest import TestCase

import numpy as np
from phi import math

from phi.flow import CLOSED, PERIODIC, OPEN, Domain, poisson_solve, Noise
from phi.physics.pressuresolver.geom import GeometricCG
from phi.physics.pressuresolver.sparse import SparseCG, SparseSciPy


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
        ]


class TestPoissonSolve(TestCase):

    def test_equal_results(self):
        data_in = _generate_examples()
        for domain in DOMAINS:
            pressure_fields = [poisson_solve(domain.centered_grid(data_in), domain, solver=solver)[0].data for solver in [SparseCG(), GeometricCG()]]
            for field in pressure_fields[1:]:
                np.testing.assert_almost_equal(field, pressure_fields[0], decimal=4)

    def test_sparse_cg(self):
        _test_all(SparseCG())

    def test_sparse_scipy(self):
        _test_all(SparseSciPy())

    def test_geometric_cg(self):
        _test_all(GeometricCG())
