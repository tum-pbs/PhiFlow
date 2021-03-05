from unittest import TestCase

from phi import math, field
from phi.geom import Box, Sphere
from phi.field import StaggeredGrid, CenteredGrid, divergence, Noise
from phi.physics import Domain, CLOSED, fluid, OPEN
from phi.tf import TF_BACKEND
from phi.torch import TORCH_BACKEND

import tensorflow as tf
if tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)  # prevent Blas GEMM launch failed


class FluidTest(TestCase):

    # Backend-independent abstract tests

    def _test_make_incompressible(self, grid_type):
        DOMAIN = Domain(x=16, y=16, boundaries=CLOSED, bounds=Box[0:100, 0:100])
        smoke = DOMAIN.scalar_grid(Sphere(center=(50, 10), radius=5))
        velocity = DOMAIN.vector_grid(0, grid_type)
        for _ in range(2):
            velocity += smoke * (0, 0.1) >> velocity
            velocity, pressure, _, _ = fluid.make_incompressible(velocity, DOMAIN)
        math.assert_close(divergence(velocity).values, 0, abs_tolerance=2e-5)
        return velocity.values

    def _test_make_incompressible_batched(self, grid_type):
        DOMAIN = Domain(x=16, y=16, boundaries=CLOSED, bounds=Box[0:100, 0:100])
        smoke = DOMAIN.scalar_grid(Sphere(center=(math.random_uniform(batch=2) * 100, 10), radius=5))
        velocity = DOMAIN.vector_grid(0, grid_type)
        for _ in range(2):
            velocity += smoke * (0, 0.1) >> velocity
            velocity, pressure, _, _ = fluid.make_incompressible(velocity, DOMAIN)
        math.assert_close(divergence(velocity).values, 0, abs_tolerance=2e-5)
        return velocity.values

    # Backend-specific testing

    def test_make_incompressible_staggered_scipy(self):
        with math.NUMPY_BACKEND:
            self._test_make_incompressible(StaggeredGrid)
            self._test_make_incompressible_batched(StaggeredGrid)

    def test_make_incompressible_centered_scipy(self):
        with math.NUMPY_BACKEND:
            self._test_make_incompressible(CenteredGrid)
            self._test_make_incompressible_batched(CenteredGrid)

    def test_make_incompressible_staggered_tensorflow(self):
        with TF_BACKEND:
            self._test_make_incompressible(StaggeredGrid)
            self._test_make_incompressible_batched(StaggeredGrid)

    def test_make_incompressible_centered_tensorflow(self):
        with TF_BACKEND:
            self._test_make_incompressible(CenteredGrid)
            self._test_make_incompressible_batched(CenteredGrid)

    def test_make_incompressible_staggered_pytorch(self):
        with TORCH_BACKEND:
            self._test_make_incompressible(StaggeredGrid)
            self._test_make_incompressible_batched(StaggeredGrid)

    def test_make_incompressible_centered_pytorch(self):
        with TORCH_BACKEND:
            self._test_make_incompressible(CenteredGrid)
            self._test_make_incompressible_batched(CenteredGrid)

    def test_make_incompressible_np_equal_tf(self):
        with math.NUMPY_BACKEND:
            v_np = self._test_make_incompressible(StaggeredGrid)
        with TF_BACKEND:
            v_tf = self._test_make_incompressible(StaggeredGrid)
        math.assert_close(v_np, v_tf, abs_tolerance=1e-5)

    def test_make_incompressible_np_equal_torch(self):
        import sys
        print(sys.getrecursionlimit())
        with math.NUMPY_BACKEND:
            v_np = self._test_make_incompressible(StaggeredGrid)
        with TORCH_BACKEND:
            v_to = self._test_make_incompressible(StaggeredGrid)
        math.assert_close(v_np, v_to, abs_tolerance=1e-5)

    def test_make_incompressible_gradients_equal_tf_torch(self):
        DOMAIN = Domain(x=16, y=16, boundaries=OPEN, bounds=Box[0:100, 0:100])  # TODO CLOSED solve fails because div is not subtracted from dx
        velocity0 = DOMAIN.staggered_grid(Noise(vector=2))
        grads = []
        for backend in [TF_BACKEND, TORCH_BACKEND]:
            with backend:
                velocity = param = velocity0.with_(values=math.tensor(velocity0.values))
                with math.record_gradients(param.values):
                    solve = math.LinearSolve()
                    velocity, _, _, _ = fluid.make_incompressible(velocity, DOMAIN, solve_params=solve)
                    loss = field.l2_loss(velocity)
                    assert math.isfinite(loss)
                    grad = math.gradients(loss, param.values)
                    assert math.all(math.isfinite(grad))
                    grads.append(grad)
        math.assert_close(*grads, abs_tolerance=1e-5)
