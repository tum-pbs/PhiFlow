from functools import partial
from unittest import TestCase

import phi
from phi import math
from phi.math import Solve, Diverged, tensor, SolveTape, extrapolation, spatial, batch, channel
from phi.math.backend import Backend

BACKENDS = phi.detect_backends()


class TestOptimize(TestCase):

    def test_minimize(self):
        def loss(x, y):
            return math.l2_loss(x - 1) + math.l2_loss(y + 1)

        for backend in BACKENDS:
            if backend.supports(Backend.jacobian):
                with backend:
                    x0 = tensor([0, 0, 0], spatial('x')), tensor([-1, -1, -1], spatial('y'))
                    x, y = math.minimize(loss, math.Solve('L-BFGS-B', 0, 1e-3, x0=x0))
                    math.assert_close(x, 1, abs_tolerance=1e-3, msg=backend.name)
                    math.assert_close(y, -1, abs_tolerance=1e-3, msg=backend.name)

                    x0 = tensor([[0, 0, 0], [1, 1, 1]], batch('batch'), spatial('x')), tensor([[0, 0, 0], [-1, -1, -1]], batch('batch'), spatial('y'))
                    x, y = math.minimize(loss, math.Solve('L-BFGS-B', 0, 1e-3, x0=x0))
                    math.assert_close(x, 1, abs_tolerance=1e-3, msg=backend.name)
                    math.assert_close(y, -1, abs_tolerance=1e-3, msg=backend.name)

                    with math.SolveTape() as solves:
                        x, y = math.minimize(loss, math.Solve('L-BFGS-B', 0, 1e-3, x0=x0))
                    math.assert_close(x, 1, abs_tolerance=1e-3, msg=backend.name)
                    math.assert_close(y, -1, abs_tolerance=1e-3, msg=backend.name)
                    math.assert_close(solves[0].residual, 0, abs_tolerance=1e-4)
                    assert (solves[0].iterations <= [4, 0]).all
                    assert (solves[0].function_evaluations <= [30, 1]).all

                    with math.SolveTape(record_trajectories=True) as trajectories:
                        x, y = math.minimize(loss, math.Solve('L-BFGS-B', 0, 1e-3, x0=x0))
                    math.assert_close(x, 1, abs_tolerance=1e-3, msg=backend.name)
                    math.assert_close(y, -1, abs_tolerance=1e-3, msg=backend.name)
                    math.assert_close(trajectories[0].residual.trajectory[-1], 0, abs_tolerance=1e-4)
                    assert (trajectories[0].iterations == solves[0].iterations).all
                    assert trajectories[0].residual.trajectory.size == trajectories[0].x[0].trajectory.size
                    assert trajectories[0].residual.trajectory.size > 1

    def test_solve_linear_matrix(self):
        for backend in BACKENDS:
            with backend:
                y = math.ones(spatial(x=3))
                x0 = math.zeros(spatial(x=3))
                for method in ['CG', 'CG-adaptive', 'biCG-stab(1)']:
                    solve = math.Solve(method, 0, 1e-3, x0=x0, max_iterations=100)
                    x = math.solve_linear(math.jit_compile_linear(partial(math.laplace, padding=extrapolation.ZERO)), y, solve)
                    math.assert_close(x, [-1.5, -2, -1.5], abs_tolerance=1e-3, msg=backend)

    def test_linear_solve_matrix_batched(self):  # TODO also test batched matrix
        y = math.ones(spatial(x=3)) * math.vec(x=1, y=2)
        x0 = math.zeros(spatial(x=3))
        for method in ['CG', 'CG-adaptive', 'auto']:
            solve = math.Solve(method, 0, 1e-3, x0=x0, max_iterations=100)
            x = math.solve_linear(math.jit_compile_linear(partial(math.laplace, padding=extrapolation.ZERO)), y, solve)
            math.assert_close(x, [[-1.5, -2, -1.5], [-3, -4, -3]], abs_tolerance=1e-3)

    def test_linear_solve_matrix_jit(self):
        @math.jit_compile
        def solve(y, method):
            solve = math.Solve(method, 0, 1e-3, x0=x0, max_iterations=100)
            return math.solve_linear(math.jit_compile_linear(partial(math.laplace, padding=extrapolation.ZERO)), y, solve)

        for backend in BACKENDS:
            with backend:
                x0 = math.zeros(spatial(x=3))
                for method in ['CG']:
                    x = solve(math.zeros(spatial(x=3)), method=method)
                    math.assert_close(x, 0, abs_tolerance=1e-3)
                    x = solve(math.ones(spatial(x=3)), method=method)
                    math.assert_close(x, [-1.5, -2, -1.5], abs_tolerance=1e-3)

    def test_linear_solve_matrix_tape(self):
        y = math.ones(spatial(x=3)) * math.vec(x=1, y=2)
        x0 = math.zeros(spatial(x=3))
        for method in ['CG', 'CG-adaptive', 'auto']:
            solve = math.Solve(method, 0, 1e-3, x0=x0, max_iterations=100)
            with math.SolveTape() as solves:
                x = math.solve_linear(math.jit_compile_linear(partial(math.laplace, padding=extrapolation.ZERO)), y, solve)
            math.assert_close(x, [[-1.5, -2, -1.5], [-3, -4, -3]], abs_tolerance=1e-3)
            assert len(solves) == 1
            assert solves[0] == solves[solve]
            math.assert_close(solves[solve].residual, 0, abs_tolerance=1e-3)
            assert math.close(solves[solve].iterations, 2) or math.close(solves[solve].iterations, -1)
            with math.SolveTape(record_trajectories=True) as solves:
                x = math.solve_linear(math.jit_compile_linear(partial(math.laplace, padding=extrapolation.ZERO)), y, solve)
            math.assert_close(x, [[-1.5, -2, -1.5], [-3, -4, -3]], abs_tolerance=1e-3)
            assert solves[solve].x.trajectory.size >= 3
            math.assert_close(solves[solve].residual.trajectory[-1], 0, abs_tolerance=1e-3)
            # math.print(solves[solve].x.vector[1])

    def test_solve_linear_function_batched(self):
        y = math.ones(spatial(x=3)) * math.vec(x=1, y=2)
        x0 = math.zeros(spatial(x=3))
        for method in ['CG', 'CG-adaptive', 'auto']:
            solve = math.Solve(method, 0, 1e-3, x0=x0, max_iterations=100)
            x = math.solve_linear(math.jit_compile_linear(partial(math.laplace, padding=extrapolation.ZERO)), y, solve)
            math.assert_close(x, math.wrap([[-1.5, -2, -1.5], [-3, -4, -3]], channel('vector'), spatial('x')), abs_tolerance=1e-3)
            with math.SolveTape() as solves:
                x = math.solve_linear(math.jit_compile_linear(partial(math.laplace, padding=extrapolation.ZERO)), y, solve)
            math.assert_close(x, math.wrap([[-1.5, -2, -1.5], [-3, -4, -3]], channel('vector'), spatial('x')), abs_tolerance=1e-3)
            assert len(solves) == 1
            assert solves[0] == solves[solve]
            math.assert_close(solves[solve].residual, 0, abs_tolerance=1e-3)

    def test_solve_diverge(self):
        y = math.ones(spatial(x=2)) * [1, 2]
        x0 = math.zeros(spatial(x=2))
        for method in ['CG']:
            solve = Solve(method, 0, 1e-3, x0=x0, max_iterations=100)
            try:
                math.solve_linear(math.jit_compile_linear(math.laplace), y, solve)
                assert False
            except Diverged:
                pass
            with math.SolveTape(record_trajectories=True) as solves:
                try:
                    math.solve_linear(math.jit_compile_linear(math.laplace), y, solve)  # impossible
                    assert False
                except Diverged:
                    pass

    def test_solve_linear_matrix_dirichlet(self):
        for backend in BACKENDS:
            with backend:
                y = math.ones(spatial(x=3))
                x0 = math.zeros(spatial(x=3))
                solve = math.Solve('CG', 0, 1e-3, x0=x0, max_iterations=100)
                x_ref = math.solve_linear(partial(math.laplace, padding=extrapolation.ONE), y, solve)
                x_jit = math.solve_linear(math.jit_compile_linear(partial(math.laplace, padding=extrapolation.ONE)), y, solve)
                math.assert_close(x_ref, x_jit, [-0.5, -1, -0.5], abs_tolerance=1e-3, msg=backend)

    def test_jit_solves(self):
        @math.jit_compile
        def solve(y, method):
            print(f"Tracing {method} with {backend}...")
            solve = math.Solve(method, 0, 1e-3, x0=x0, max_iterations=100)
            with SolveTape() as solves:
                x = math.solve_linear(math.jit_compile_linear(partial(math.laplace, padding=extrapolation.ZERO)), y, solve)
            return x

        for backend in BACKENDS:
            with backend:
                x0 = math.zeros(spatial(x=3))

                for method in ['CG', 'CG-adaptive', 'auto']:
                    x = solve(math.zeros(spatial(x=3)), method=method)
                    math.assert_close(x, 0, abs_tolerance=1e-3)
                    x = solve(math.ones(spatial(x=3)), method=method)
                    math.assert_close(x, [-1.5, -2, -1.5], abs_tolerance=1e-3)

    def test_gradient_descent_minimize(self):
        def loss(x):
            return x ** 2

        for backend in BACKENDS:
            if backend.supports(Backend.jacobian):
                with backend:
                    result = math.minimize(loss, Solve('GD', 0, 1e-5, x0=3, max_iterations=20))
                    math.assert_close(result, 0, abs_tolerance=1e-5, msg=backend.name)

    def test_solve_dense(self):
        @math.jit_compile_linear
        def f(x):
            return math.laplace(x, padding=extrapolation.ZERO)

        for backend in BACKENDS:
            with backend:
                matrix, bias = math.matrix_from_function(f, math.ones(spatial(x=3)))
                dense_matrix = math.dense(matrix)

                @math.jit_compile
                def solve(y):
                    return math.solve_linear(dense_matrix, y, Solve('CG', x0=y * 0))

                x = solve(math.ones(spatial(x=3)))
                math.assert_close([-1.5, -2, -1.5], x)
