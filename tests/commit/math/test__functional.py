from functools import partial
from unittest import TestCase

import phi
from phi import math
from phi.math import Solve, Diverged, tensor, SolveTape, extrapolation, spatial, batch, channel
from phi.math.backend import Backend

BACKENDS = phi.detect_backends()


class TestFunctional(TestCase):

    def test_jit_compile(self):
        @math.jit_compile
        def scalar_mul(x, fac=1):
            return x * fac

        for backend in BACKENDS:
            with backend:
                x = math.ones(spatial(x=4))
                trace_count_0 = len(scalar_mul.traces)
                math.assert_close(scalar_mul(x, fac=1), 1, msg=backend)
                if backend.supports(Backend.jit_compile):
                    self.assertEqual(len(scalar_mul.traces), trace_count_0 + 1)
                math.assert_close(scalar_mul(x, fac=1), 1, msg=backend)
                if backend.supports(Backend.jit_compile):
                    self.assertEqual(len(scalar_mul.traces), trace_count_0 + 1)
                math.assert_close(scalar_mul(x, fac=2), 2, msg=backend)
                if backend.supports(Backend.jit_compile):
                    self.assertEqual(len(scalar_mul.traces), trace_count_0 + 1)
                math.assert_close(scalar_mul(math.zeros(spatial(x=4)), fac=2), 0, msg=backend)
                if backend.supports(Backend.jit_compile):
                    self.assertEqual(len(scalar_mul.traces), trace_count_0 + 1)
                math.assert_close(scalar_mul(math.zeros(spatial(y=4)), fac=2), 0, msg=backend)
                if backend.supports(Backend.jit_compile):
                    self.assertEqual(len(scalar_mul.traces), trace_count_0 + 2)

    def test_jit_compile_aux(self):
        @partial(math.jit_compile, auxiliary_args='fac')
        def scalar_mul(x, fac):
            return x * fac
        math.assert_close(6, scalar_mul(2, 3))

    def test_jit_compile_with_native(self):
        @math.jit_compile
        def scalar_mul(x, fac=1):
            return x * fac

        for backend in BACKENDS:
            with backend:
                x = backend.ones([3, 2])
                math.assert_close(scalar_mul(x, fac=1), 1, msg=backend)

    def test_jit_compile_linear(self):
        x = math.random_normal(batch(batch=3) & spatial(x=4, y=3))  # , vector=2

        def linear_function(val):
            val = -val
            val *= 2
            val = math.pad(val, {'x': (2, 0), 'y': (0, 1)}, math.extrapolation.PERIODIC)
            val = val.x[:-2].y[1:] + val.x[2:].y[:-1]
            val = math.pad(val, {'x': (0, 0), 'y': (0, 1)}, math.extrapolation.ZERO)
            val = math.pad(val, {'x': (2, 2), 'y': (0, 1)}, math.extrapolation.BOUNDARY)
            return math.sum([val, val], dim='0') - val

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
            assert isinstance(x, math.Tensor)
            assert isinstance(y, math.Tensor)
            pred = x
            loss = math.l2_loss(pred - y)
            return loss, pred

        for backend in BACKENDS:
            if backend.supports(Backend.jacobian):
                with backend:
                    x_data = math.tensor(2.)
                    y_data = math.tensor(1.)
                    dx = math.functional_gradient(f, wrt=0, get_output=False)(x_data, y_data)
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

        def grad(_inputs, _y, df):
            return {'x': df * 0}

        for backend in BACKENDS:
            if backend.supports(Backend.jacobian):
                with backend:
                    normal_gradient, = math.functional_gradient(f, get_output=False)(math.ones())
                    math.assert_close(normal_gradient, 1)
                    f_custom_grad = math.custom_gradient(f, grad)
                    custom_gradient, = math.functional_gradient(f_custom_grad, get_output=False)(math.ones())
                    math.assert_close(custom_gradient, 0)

    def test_custom_gradient_vector(self):
        def f(x):
            return x.x[:2]

        def grad(_inputs, _y, df):
            return {'x': math.flatten(math.expand(df * 0, batch(tmp=2)))}

        def loss(x):
            fg = math.custom_gradient(f, grad)
            y = fg(x)
            return math.l1_loss(y)

        for backend in BACKENDS:
            if backend.supports(Backend.custom_gradient):  # and backend.name != 'Jax':
                with backend:
                    custom_loss_grad, = math.functional_gradient(loss, get_output=False)(math.ones(spatial(x=4)))
                    math.assert_close(custom_loss_grad, 0, msg=backend.name)

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
                for method in ['CG', 'CG-adaptive', 'auto']:
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
            assert solves[solve].x.trajectory.size == 3
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
                    result = math.minimize(loss, Solve('GD', 0, 1e-5, 20, x0=3))
                    math.assert_close(result, 0, abs_tolerance=1e-5, msg=backend.name)

    def test_map_types(self):
        def f(x, y):
            assert x.shape.batch.names == ('batch', 'x', 'y')
            assert x.shape.channel.names == ('vector',)
            assert y.shape == x.shape
            return x, y

        for f_ in [
            # math.map_types(f, 'x,y', batch),
            # math.map_types(f, spatial('x,y'), batch),
            math.map_types(f, spatial, batch),
        ]:
            x = math.random_uniform(batch(batch=10), spatial(x=4, y=3), channel(vector=2))
            x_, y_ = f_(x, x)
            assert x_.shape == x.shape
            math.assert_close(x, x_)

    def test_hessian(self):
        def f(x, y):
            return math.l1_loss(x ** 2 * y), x, y
        
        eval_hessian = math.hessian(f, wrt='x', get_output=True, get_gradient=True, dim_suffixes=('1', '2'))

        for backend in BACKENDS:
            if backend.supports(Backend.hessian):
                with backend:
                    x = math.tensor([(0.01, 1, 2)], channel('vector', 'v'))
                    y = math.tensor([1., 2.], batch('batch'))
                    (L, x, y), g, H, = eval_hessian(x, y)
                    math.assert_close(L, [5.0001, 10.0002], msg=backend.name)
                    math.assert_close(g.batch[0].vector[0], (0.02, 2, 4), msg=backend.name)
                    math.assert_close(g.batch[1].vector[0], (0.04, 4, 8), msg=backend.name)
                    math.assert_close(2, H.v1[0].v2[0].batch[0], H.v1[1].v2[1].batch[0], H.v1[2].v2[2].batch[0], msg=backend.name)
                    math.assert_close(4, H.v1[0].v2[0].batch[1], H.v1[1].v2[1].batch[1], H.v1[2].v2[2].batch[1], msg=backend.name)

    def test_sparse_matrix(self):
        for backend in BACKENDS:
            with backend:
                for f in ['csr', 'csc', 'coo']:
                    matrix = math.jit_compile_linear(math.laplace).sparse_matrix(math.zeros(spatial(x=5)), format=f)
                    self.assertEqual(f, matrix.indexing_type)
                    self.assertEqual((5, 5), matrix.shape)

    def test_loss_batch_not_reduced(self):
        def loss_function(x):
            return math.l2_loss(x)

        gradient_function = math.functional_gradient(loss_function, wrt=0)

        for backend in BACKENDS:
            if backend.supports(Backend.jacobian):
                with backend:
                    x_test = tensor([0, 1], batch('examples'))
                    loss_direct = loss_function(x_test)
                    loss_g, _ = gradient_function(x_test)
                    math.assert_close([0, 0.5], loss_g, loss_direct)

    def test_iterate(self):
        def f(x, fac):
            return x * fac

        self.assertEqual(4, math.iterate(f, 2, 1, f_kwargs=dict(fac=2.)))
        math.assert_close([1, 2, 4], math.iterate(f, batch(trajectory=2), 1, f_kwargs=dict(fac=2.)))
