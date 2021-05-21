from unittest import TestCase

import phi
from phi import math, field
from phi.math.backend import Backend

BACKENDS = phi.detect_backends()


class TestTrace(TestCase):

    def test_jit_compile(self):
        @math.jit_compile
        def scalar_mul(x, fac=1):
            return x * fac

        for backend in BACKENDS:
            with backend:
                x = math.ones(x=4)
                trace_count_0 = len(scalar_mul.traces)
                math.assert_close(scalar_mul(x, fac=1), 1, msg=backend)
                if backend.supports(Backend.jit_compile):
                    self.assertEqual(len(scalar_mul.traces), trace_count_0 + 1)
                math.assert_close(scalar_mul(x, fac=1), 1, msg=backend)
                if backend.supports(Backend.jit_compile):
                    self.assertEqual(len(scalar_mul.traces), trace_count_0 + 1)
                math.assert_close(scalar_mul(x, fac=2), 2, msg=backend)
                if backend.supports(Backend.jit_compile):
                    self.assertEqual(len(scalar_mul.traces), trace_count_0 + 2)
                math.assert_close(scalar_mul(math.zeros(x=4), fac=2), 0, msg=backend)
                if backend.supports(Backend.jit_compile):
                    self.assertEqual(len(scalar_mul.traces), trace_count_0 + 2)
                math.assert_close(scalar_mul(math.zeros(y=4), fac=2), 0, msg=backend)
                if backend.supports(Backend.jit_compile):
                    self.assertEqual(len(scalar_mul.traces), trace_count_0 + 3)

    def test_jit_compile_linear(self):
        math.GLOBAL_AXIS_ORDER.x_last()
        x = math.random_normal(batch=3, x=4, y=3)  # , vector=2

        def linear_function(val):
            val = -val
            val *= 2
            val = math.pad(val, {'x': (2, 0), 'y': (0, 1)}, math.extrapolation.PERIODIC)
            val = val.x[:-2].y[1:] + val.x[2:].y[:-1]
            val = math.pad(val, {'x': (0, 0), 'y': (0, 1)}, math.extrapolation.ZERO)
            val = math.pad(val, {'x': (2, 2), 'y': (0, 1)}, math.extrapolation.BOUNDARY)
            return math.sum([val, val], dim=0) - val

        functions = [
            linear_function,
            lambda val: math.spatial_gradient(val, difference='forward', padding=math.extrapolation.ZERO, dims='x').gradient[0],
            lambda val: math.spatial_gradient(val, difference='backward', padding=math.extrapolation.PERIODIC, dims='x').gradient[0],
            lambda val: math.spatial_gradient(val, difference='central', padding=math.extrapolation.BOUNDARY, dims='x').gradient[0],
        ]
        for f in functions:
            direct_result = f(x)
            jit_f = math.jit_compile_linear(f)
            jit_result = jit_f(x)
            math.assert_close(direct_result, jit_result)

    def test_functional_gradient(self):
        def f(x: math.Tensor, y: math.Tensor):
            pred = x
            loss = math.l2_loss(pred - y)
            return loss, pred

        for backend in BACKENDS:
            if backend.supports(Backend.functional_gradient):
                with backend:
                    x_data = math.tensor(2.)
                    y_data = math.tensor(1.)
                    dx, = math.functional_gradient(f, wrt=[0], get_output=False)(x_data, y_data)
                    math.assert_close(dx, 1, msg=backend.name)
                    dx, dy = math.functional_gradient(f, [0, 1], get_output=False)(x_data, y_data)
                    math.assert_close(dx, 1, msg=backend.name)
                    math.assert_close(dy, -1, msg=backend.name)
                    (loss, pred), (dx, dy) = math.functional_gradient(f, [0, 1], get_output=True)(x_data, y_data)
                    math.assert_close(loss, 0.5, msg=backend.name)
                    math.assert_close(pred, x_data, msg=backend.name)
                    math.assert_close(dx, 1, msg=backend.name)
                    math.assert_close(dy, -1, msg=backend.name)

    def test_custom_gradient_scalar(self):
        def f(x):
            return x

        def grad(_x, _y, df):
            return df * 0,

        for backend in BACKENDS:
            if backend.supports(Backend.functional_gradient):
                with backend:
                    normal_gradient, = math.functional_gradient(f, get_output=False)(math.ones())
                    math.assert_close(normal_gradient, 1)
                    f_custom_grad = math.custom_gradient(f, grad)
                    custom_gradient, = math.functional_gradient(f_custom_grad, get_output=False)(math.ones())
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
                    custom_loss_grad, = math.functional_gradient(loss, get_output=False)(math.ones(x=4))
                    math.assert_close(custom_loss_grad, 0, msg=backend.name)

    def test_minimize(self):
        def loss(x, y):
            return math.l1_loss(x - 1) + math.l1_loss(y + 1)

        x0 = math.zeros(x=4), math.zeros(y=4)
        for backend in BACKENDS:
            if backend.supports(Backend.functional_gradient):
                print(backend)
                with backend:
                    result = math.minimize(loss, math.Solve('L-BFGS-B', 0, 1e-3, x0=x0))
                    x, y = result.x
                    math.assert_close(x, 1, abs_tolerance=1e-3, msg=backend.name)
                    math.assert_close(y, -1, abs_tolerance=1e-3, msg=backend.name)

    def test_solve_linear_matrix(self):
        for backend in BACKENDS:
            with backend:
                y = field.grid(1, math.shape(x=3))
                x0 = field.grid(0, math.shape(x=3))
                for method in ['CG', 'CG-adaptive', 'auto']:
                    solve = math.Solve(method, 0, 1e-3, x0=x0, max_iterations=100)
                    x = field.solve_linear(math.jit_compile_linear(field.laplace), y, solve)
                    math.assert_close(x.values, [-1.5, -2, -1.5], abs_tolerance=1e-3, msg=backend)

    def test_linear_solve_matrix_batched(self):
        y = field.grid(1, math.shape(x=3)) * (1, 2)
        x0 = field.grid(0, math.shape(x=3))
        for method in ['CG', 'CG-adaptive', 'auto']:
            solve = math.Solve(method, 0, 1e-3, x0=x0, max_iterations=100)
            x = field.solve_linear(math.jit_compile_linear(field.laplace), y, solve)
            math.assert_close(x.values, [[-1.5, -2, -1.5], [-3, -4, -3]], abs_tolerance=1e-3)

    def test_linear_solve_matrix_tape(self):
        y = field.grid(1, math.shape(x=3)) * (1, 2)
        x0 = field.grid(0, math.shape(x=3))
        for method in ['CG', 'CG-adaptive', 'auto']:
            solve = math.Solve(method, 0, 1e-3, x0=x0, max_iterations=100)
            with math.SolveTape() as solves:
                x = field.solve_linear(math.jit_compile_linear(field.laplace), y, solve)
            math.assert_close(x.values, [[-1.5, -2, -1.5], [-3, -4, -3]], abs_tolerance=1e-3)
            assert len(solves) == 1
            assert solves[0] == solves[solve]
            math.assert_close(solves[solve].residual.values, 0, abs_tolerance=1e-3)
            math.assert_close(solves[solve].iterations, 2)
            with math.SolveTape(record_trajectories=True) as solves:
                x = field.solve_linear(math.jit_compile_linear(field.laplace), y, solve)
            math.assert_close(x.values, [[-1.5, -2, -1.5], [-3, -4, -3]], abs_tolerance=1e-3)
            assert solves[solve].x.trajectory.size == 3
            math.assert_close(solves[solve].residual.trajectory[-1].values, 0, abs_tolerance=1e-3)
            # math.print(solves[solve].x.vector[1])

    def test_solve_linear_function_batched(self):
        y = field.grid(1, math.shape(x=3)) * (1, 2)
        x0 = field.grid(0, math.shape(x=3))
        for method in ['CG', 'CG-adaptive', 'auto']:
            solve = math.Solve(method, 0, 1e-3, x0=x0, max_iterations=100)
            x = field.solve_linear(math.jit_compile_linear(field.laplace), y, solve)
            math.assert_close(x.values, math.wrap([[-1.5, -2, -1.5], [-3, -4, -3]], 'vector,x'), abs_tolerance=1e-3)
            with math.SolveTape() as solves:
                x = field.solve_linear(math.jit_compile_linear(field.laplace), y, solve)
            math.assert_close(x.values, math.wrap([[-1.5, -2, -1.5], [-3, -4, -3]], 'vector,x'), abs_tolerance=1e-3)
            assert len(solves) == 1
            assert solves[0] == solves[solve]
            math.assert_close(solves[solve].residual.values, 0, abs_tolerance=1e-3)

