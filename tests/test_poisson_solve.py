from unittest import TestCase

import numpy as np
from phi import math

from phi.flow import CLOSED, PERIODIC, OPEN, Domain, poisson_solve
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
    np.testing.assert_almost_equal(p.laplace().data[:, 1:-1, 1:-1, :], data_in[:, 1:-1, 1:-1, :], decimal=5)
    if domain.boundaries is CLOSED:
        np.testing.assert_almost_equal(p.laplace().data, data_in, decimal=5)
    # rows = math.unstack(p.data, 1)
    # for row in rows[1:]:
    #     np.testing.assert_almost_equal(row, rows[0], decimal=5)


DOMAINS = [
            Domain([4, 5], boundaries=CLOSED),
            Domain([4, 5], boundaries=OPEN),
            Domain([4, 5], boundaries=PERIODIC),
            Domain([4, 5], boundaries=[PERIODIC, CLOSED]),
            Domain([4, 5], boundaries=[CLOSED, OPEN]),
        ]

SOLVERS = [
    SparseCG(), GeometricCG()
]


class TestPoissonSolve(TestCase):

    def test_equal_results(self):
        data_in = _generate_examples()
        for domain in DOMAINS:
            pressure_fields = [poisson_solve(domain.centered_grid(data_in), domain, solver=solver)[0].data for solver in SOLVERS]
            for field in pressure_fields[1:]:
                np.testing.assert_almost_equal(field, pressure_fields[0], decimal=4)

    def test_sparse_cg(self):
        solver = SparseCG()
        for domain in DOMAINS:
            _test_solve_no_obstacles(domain, solver)

    # def test_sparse_scipy(self):
    #     solver = SparseSciPy()
    #     for domain in DOMAINS:
    #         _test_solve_no_obstacles(domain, solver)

    def test_geometric_cg(self):
        solver = GeometricCG()
        for domain in DOMAINS:
            _test_solve_no_obstacles(domain, solver)
